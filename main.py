###################################################################################################################################################################################################################
#  _____          _______      __  _____           _           _   
# |_   _|   /\   / ____\ \    / / |  __ \         (_)         | |  
#   | |    /  \ | |     \ \  / /  | |__) | __ ___  _  ___  ___| |_ 
#   | |   / /\ \| |      \ \/ /   |  ___/ '__/ _ \| |/ _ \/ __| __|
#  _| |_ / ____ \ |____   \  /    | |   | | | (_) | |  __/ (__| |_ 
# |_____/_/    \_\_____|   \/     |_|   |_|  \___/| |\___|\___|\__|
#                                                _/ |              
#                                               |__/               
# Project developed by Paolo Riva, Michelangelo Stasi, Mihai-Viorel Grecu c/o Politecnico di Milano
# Course: Image Analysis and Computer Vison - A.A. 2023/24
# This Computer Vision project aims at detecting data in a tennis match through the widely-used Human Pose Detection method and the TRACE methon for ball detection.
#
# The program focuses specifically on the following tasks:
# 0. Identify the field lines and, knowing the field measures, find yhe homography H from field to image.
# 1. Use the well-known Human Pose Estimation  method (based on Deep Learning) to identify the articulated segments of the player.
# 2. Select the feet (end points of the leg segments) and their position Pleft and Pright in each image
# 3. Check whether the feet are static or they are moving (by checking if H^-1 Pleft and/or H^-1 Pright are constant along a short sequence).
#    If a foot is static, assume that it is placed on the ground.
# 4. Collect the time-sequence of the step points: i.e., the positions H^-1P of the feet in the instances when they were static.
# 5. In parallel, try to select the time instants when the player hits the ball with the rackets, and try to compute statistics on the short runs between consecutive hits
# 
###################################################################################################################################################################################################################

## MODULES ########################################################
import sys
import cv2
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import threading
#from mediapipe import solutions
#from cv2 import cvtColor, COLOR_BGR2RGB, COLOR_RGB2BGR
import mediapipe as mp
import time
from TRACE.BallDetection import BallDetector
from TRACE.BallMapping import euclideanDistance, withinCircle
from moviepy.editor import VideoFileClip
from scipy.optimize import minimize

from player_detection.playerDetection import PlayersDetections
##################################################################

def computePoseAndAnkles(cropped_frame, static_centers_queue, mpPose, pose, mpDraw, hom_matrix, prev_right_ankle, prev_left_ankle, threshold, x_offset, y_offset, rect_img, rightwristbuffer, leftwristbuffer, heightbuffer):

    imgRGB = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    #print(results.pose_landmarks)
    right_ankle, left_ankle = (0,0),(0,0)
    Pright_image, Pleft_image = (0,0),(0,0)
    #global right_wrist # Required for Racket Hit detection
    #global left_wrist # Required for Racket Hit detection
    
    head_y, _ , _ = cropped_frame.shape
    head_y *= 0.1 # By default, the difference between feet and head (height of the player) will be 1/10 of the height of the video [in pixels]
    feet_y = 0
    detected_height = head_y

    if results.pose_landmarks:
        mpDraw.draw_landmarks(cropped_frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w,c = cropped_frame.shape
            #print(id, lm)
            if id == 28 : ## RIGHT ANKLE
                right_ankle = (int(lm.x*w), int(lm.y*h))
                cv2.circle(cropped_frame, right_ankle, 5, (0,0,255), cv2.FILLED)

                #right_ankle but in the context of the full frame (not the cropped one)
                right_ankle_real = (right_ankle[0] + x_offset, right_ankle[1] + y_offset, 0)
                right_ankle_real = np.array([[right_ankle_real[0], right_ankle_real[1]]], dtype=np.float32)
                right_ankle_real = np.reshape(right_ankle_real, (1,1,2))
                #standard function to get the resulting point by applying the homography to the point of the image
                Pright_image = cv2.perspectiveTransform(right_ankle_real, hom_matrix)                     
                # Approximation to avoid displaying all the decimals      
                Pright_image = (round(Pright_image[0][0][0]), round(Pright_image[0][0][1]))
                # Display of the right foot field real coordinates values at image coordinates, with slight offset on the X axis to avoid overlapping with the actual foot
                cv2.putText(cropped_frame, f"{Pright_image}", (right_ankle[0] + 10, right_ankle [1]), font, font_scale, color, thickness, cv2.LINE_AA)        
            elif id == 27 : ## LEFT ANKLE
                left_ankle = (int(lm.x*w), int(lm.y*h))
                cv2.circle(cropped_frame, left_ankle, 5, (0,255,0), cv2.FILLED)
                
                #left_ankle but in the context of the full frame (not the cropped one)
                left_ankle_real = (left_ankle[0] + x_offset, left_ankle[1] + y_offset, 0)
                left_ankle_real = np.array([[left_ankle_real[0], left_ankle_real[1]]], dtype=np.float32)
                left_ankle_real = np.reshape(left_ankle_real, (1,1,2))
                #standard function to get the resulting point by applying the homography to the point of the image
                Pleft_image = cv2.perspectiveTransform(left_ankle_real, hom_matrix)
                # Approximation to avoid displaying all the decimals      
                Pleft_image = (round(Pleft_image[0][0][0]),round(Pleft_image[0][0][1]))
                # Display of the left foot field real coordinates values at image coordinates, with slight offset on the X axis to avoid overlapping with the actual foot
                cv2.putText(cropped_frame, f"{Pleft_image}", (left_ankle[0] + 10, left_ankle [1] + 20), font, font_scale, color, thickness, cv2.LINE_AA)
            elif id == 15: ## LEFT WRIST
                cx, cy = int(lm.x*w), int(lm.y*h)
                leftwristbuffer.append((cx, cy))
                cv2.circle(cropped_frame, (cx, cy), 5, (255,0,255), cv2.FILLED)
            elif id == 16: ## RIGHT WRIST
                cx, cy = int(lm.x*w), int(lm.y*h)
                rightwristbuffer.append((cx, cy))
                cv2.circle(cropped_frame, (cx, cy), 5, (255,0,255), cv2.FILLED)
            elif id == 0: ## HEAD/NOSE
                cx, cy = int(lm.x*w), int(lm.y*h)
                cv2.circle(cropped_frame, (cx, cy), 5, (255,0,255), cv2.FILLED)
                head_y = cy # Set new head height
            else : ## GENERIC BODY POINT
                cx, cy = int(lm.x*w), int(lm.y*h)
                cv2.circle(cropped_frame, (cx, cy), 5, (255,0,0), cv2.FILLED)
                rightwristbuffer.append((cx, cy)) # will be (0,0)
                leftwristbuffer.append((cx, cy))# will be (0,0)

    # Feet Movement Detection
    if prev_right_ankle is not None and prev_left_ankle is not None:
        # Euclidean distance computation between the current Left foot position and its position in the previous frame, all compared to the chosen threshold            
        left_foot_moved = np.linalg.norm(np.array(Pleft_image) - np.array(prev_left_ankle)) > threshold 
        # Euclidean distance computation between the current Right foot position and its position in the previous frame, all compared to the chosen threshold                
        right_foot_moved = np.linalg.norm(np.array(Pright_image) - np.array(prev_right_ankle)) > threshold
    
        # Check if the left ankle's point has been detected 
        if left_ankle !=(0,0):                                                                                                                                         
            # Check if the left foot has moved 
            if left_foot_moved:                                                                     
                # Display "(LFoot) Moving" under the player's left foot using the image coordinates of the left foot with an offset  
                cv2.putText(cropped_frame, "(LFoot) Moving " + str(Pleft_image[0])+ ", " + str(Pleft_image[1]), (left_ankle[0] +10, left_ankle[1] + 40), font, font_scale, color, thickness, cv2.LINE_AA)            
            else:
                #print(str(Pleft_image) + " " + str(prev_left_ankle))
                # Display "(LFoot) Static" under the player's left foot using the image coordinates of the left foot with an offset  
                cv2.putText(cropped_frame, "(LFoot) Static " + str(Pleft_image[0])+ ", " + str(Pleft_image[1]), (left_ankle[0] +10, left_ankle[1] + 40), font, font_scale, color, thickness, cv2.LINE_AA)            
        # Check if the right ankle's point has been detected
        if right_ankle!=(0,0):
            # Check if the right foot has moved
            if  right_foot_moved:
                # Display "(RFoot) Moving" under the player's right foot using the image coordinates of the left foot with an offset  
                cv2.putText(cropped_frame, "(RFoot) Moving " + str(Pright_image[0])+ ", " + str(Pright_image[1]), (right_ankle[0] +10, right_ankle[1] -20 ), font, font_scale, color, thickness, cv2.LINE_AA)          
            else:
                # Display "(RFoot) Static" under the player's right foot using the image coordinates of the right foot with an offset
                #print(str(Pright_image) + " " + str(prev_right_ankle))
                cv2.putText(cropped_frame, "(RFoot) Static "  + str(Pright_image[0])+ ", " + str(Pright_image[1]),  (right_ankle[0] +10, right_ankle[1] -20 ), font, font_scale, color, thickness, cv2.LINE_AA)          
    
    print(str(computeMoving))
    if(prev_left_ankle is None or computeMoving) : 
        # Update the values of the field coordinates of the feet from the previous frame  with the current ones
        prev_left_ankle[0] = Pleft_image[0]
        prev_left_ankle[1] = Pleft_image[1]                                                                                                                 
    if(prev_right_ankle is None or computeMoving) :  
        # Update the values of the field coordinates of the feet from the previous frame  with the current ones
        prev_right_ankle[0] = Pright_image[0]
        prev_right_ankle[1] = Pright_image[1]
    
    # Computing the center position of the player in the real field (top view)
    center_real = tuple((int((Pright_image[0] + Pleft_image[0])/2), int((Pright_image[1] + Pleft_image[1])/2)))  
    
    # Player Height Calculation
    center_inframe = (int((right_ankle[0] + left_ankle[0])/2), int((right_ankle[1] + left_ankle[1])/2))
    height = int(abs(head_y - center_inframe[1]))
    heightbuffer.append(height)

    # Collecting static positions
    if right_ankle != (0,0) and left_ankle != (0,0) and left_foot_moved != True and right_foot_moved != True : 
        static_centers_queue.append(center_real)
    #displaying live positions of the player
    center_real = (round(center_real[0]), round(center_real[1]))
    cv2.circle(rect_img, center_real , 5, (255, 255, 0), cv2.FILLED)

def processBallTrajectory (BallDetector, frame, positions_stack):
    global pos_counter
    global prevpos
    global lastvalidpos
    global beginning

    ball_detector.detect_ball(frame) # Detection of the ball

    # Valid Position Detected by the model
    if ball_detector.xy_coordinates[-1][0] is not None and ball_detector.xy_coordinates[-1][1] is not None:
        center_x = ball_detector.xy_coordinates[-1][0]
        center_y = ball_detector.xy_coordinates[-1][1]
        currpos = (center_x, center_y)

        #if pos_counter < 3:
        #    positions_stack.append(currpos)
        #    prevpos = currpos
        #    pos_counter += 1
        #    return
        #else: sequence = True

        # Head of sequence detected
        if pos_counter < 3: #pos_counter keeps track of the head of each non-zero sequence, to avoid worst-case scenarios where the first new sequence is a mistake by the deep learning model (e.g. detecting an ankle multiple times instead of the ball)
            if not beginning and (abs(currpos[0]-lastvalidpos[0]) > 200 or abs(currpos[1]-lastvalidpos[1]) > 200): #If the ball wasn't detected last 3 frames (counter was put to 0) we check that the first value detected isn't an error (being 200 pixel off the last confirmed position). Here we avoid the beginning case, where every sample would have a big coordinate gap from any static "starting" value of lastvalidpos
                positions_stack.append((0,0)) #If there's a big difference between the last valid position when beginning a new sequence, we append 0,0 to avoid appending wrong information to the ball position array
                prevpos = (0,0)
                pos_counter += 1
                return
            else: #beginning of the detection: we accept the first 3 samples since there's no check on validity wrt previous ones we can perform
                positions_stack.append((currpos))
                prevpos = currpos
                lastvalidpos = currpos
                pos_counter += 1
                return
        else: 
            sequence = True #After 3 valid samples, we go on as a sequence
            beginning = False #After the first 3 valid samples we pass the beginning phase     

        # In-sequence processing
        if (abs(currpos[0]-prevpos[0]) > 100 or abs(currpos[1]-prevpos[1]) > 100) and sequence : #Detection happens during a sequence, but with noise: we put a zero value to allow interpolation to best estimate it from neighbor samples
            positions_stack.append((0,0))
            prevpos = (0,0)
            sequence = False
            return
        else: #Valid detection during sequence
            lastvalidpos = currpos
            positions_stack.append((currpos))
            prevpos = currpos
    
    #No detection in current frame
    else: 
        positions_stack.append((0,0))
        prevpos = (0,0)
        pos_counter = 0
        sequence = False

def determinant(a, b):
    return a[0] * b[1] - a[1] * b[0]
    
def findIntersection(line1, line2, xStart, yStart, xEnd, yEnd):
    xDiff = (line1[0][0]-line1[1][0],line2[0][0]-line2[1][0])
    yDiff = (line1[0][1]-line1[1][1],line2[0][1]-line2[1][1])
    div = determinant(xDiff, yDiff)
    if div == 0:
        return None
    d = (determinant(*line1), determinant(*line2))
    x = int(determinant(d, xDiff) / div)
    y = int(determinant(d, yDiff) / div)
    if (x<xStart) or (x>xEnd):
        return None
    if (y<yStart) or (y>yEnd):
        return None
    return x,y

def autoComputeHomography(video, frm, NtopLeftP, NtopRightP, NbottomLeftP, NbottomRightP):

    #global field_points_2D
    width = int(video.get(3))
    height = int(video.get(4))

    threshold = 10

    # Setting reference frame lines
    extraLen = width/3

    class axis:
        top = [[-extraLen,0],[width+extraLen,0]]
        right = [[width+extraLen,0],[width+extraLen,height]]
        bottom = [[-extraLen,height],[width+extraLen,height]]
        left = [[-extraLen,0],[-extraLen,height]]

        
    hasFrame, frame = cap.read()
    if hasFrame:
         # Apply filters that removes noise and simplifies image
        gry = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bw = cv2.threshold(gry, 156, 255, cv2.THRESH_BINARY)[1]
        canny = cv2.Canny(bw, 100, 200)
        # Copy edges to the images that will display the results in BGR
        cdst = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
        cdstP = np.copy(cdst)                  
        # Using hough lines probablistic to find lines with most intersections
        hPLines = cv2.HoughLinesP(canny, 1, np.pi/180, threshold=150, minLineLength=100, maxLineGap=10)
        intersectNum = np.zeros((len(hPLines),2))  
         # Draw the lines
        if hPLines is not None:
            for i in range(0, len(hPLines)):
                l = hPLines[i][0]
                cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
        i = 0
        for hPLine1 in hPLines:
            Line1x1, Line1y1, Line1x2, Line1y2 = hPLine1[0]
            Line1 = [[Line1x1,Line1y1],[Line1x2,Line1y2]]
            for hPLine2 in hPLines:
                Line2x1, Line2y1, Line2x2, Line2y2 = hPLine2[0]
                Line2 = [[Line2x1,Line2y1],[Line2x2,Line2y2]]
                if Line1 is Line2:
                    continue
                if Line1x1>Line1x2:
                    temp = Line1x1
                    Line1x1 = Line1x2
                    Line1x2 = temp

                if Line1y1>Line1y2:
                    temp = Line1y1
                    Line1y1 = Line1y2
                    Line1y2 = temp

                intersect = findIntersection(Line1, Line2, Line1x1-200, Line1y1-200, Line1x2+200, Line1y2+200)
                if intersect is not None:
                    intersectNum[i][0] += 1
            intersectNum[i][1] = i
            i += 1

             # Lines with most intersections get a fill mask command on them
        i = p = 0
        dilation = cv2.dilate(bw, np.ones((5, 5), np.uint8), iterations=1)
        nonRectArea = dilation.copy()
        intersectNum = intersectNum[(-intersectNum)[:, 0].argsort()]
        for hPLine in hPLines:
            x1,y1,x2,y2 = hPLine[0]
            # line(frame, (x1,y1), (x2,y2), (255, 255, 0), 2)
            for p in range(8):
                if (i==intersectNum[p][1]) and (intersectNum[i][0]>0):
                    #cv2.line(frame, (x1,y1), (x2,y2), (0, 0, 255), 2)
                    cv2.floodFill(nonRectArea, np.zeros((height+2, width+2), np.uint8), (x1, y1), 1) 
                    cv2.floodFill(nonRectArea, np.zeros((height+2, width+2), np.uint8), (x2, y2), 1) 
            i+=1


        dilation[np.where(nonRectArea == 255)] = 0
        dilation[np.where(nonRectArea == 1)] = 255
        eroded = cv2.erode(dilation, np.ones((5, 5), np.uint8)) 
        cannyMain = cv2.Canny(eroded, 90, 100)

         # Extreme lines found every frame
        xOLeft = width + extraLen
        xORight = 0 - extraLen
        xFLeft = width + extraLen
        xFRight = 0 - extraLen

        yOTop = height
        yOBottom = 0
        yFTop = height
        yFBottom = 0

        # Finding all lines then allocate them to specified extreme variables
        hLines = cv2.HoughLines(cannyMain, 2, np.pi/180, 300)
        for hLine in hLines:
            for rho,theta in hLine:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + width*(-b))
                y1 = int(y0 + width*(a))
                x2 = int(x0 - width*(-b))
                y2 = int(y0 - width*(a))

                # Furthest intersecting point at every axis calculations done here
                intersectxF = findIntersection(axis.bottom, [[x1,y1],[x2,y2]], -extraLen, 0, width+extraLen, height)
                intersectyO = findIntersection(axis.left, [[x1,y1],[x2,y2]], -extraLen, 0, width+extraLen, height)
                intersectxO = findIntersection(axis.top, [[x1,y1],[x2,y2]], -extraLen, 0, width+extraLen, height)
                intersectyF = findIntersection(axis.right, [[x1,y1],[x2,y2]], -extraLen, 0, width+extraLen, height)

                if (intersectxO is None) and (intersectxF is None) and (intersectyO is None) and (intersectyF is None):
                    continue
                
                if intersectxO is not None:
                    if intersectxO[0] < xOLeft:
                        xOLeft = intersectxO[0]
                        xOLeftLine = [[x1,y1],[x2,y2]]
                    if intersectxO[0] > xORight:
                        xORight = intersectxO[0]
                        xORightLine = [[x1,y1],[x2,y2]]
                if intersectyO is not None:
                    if intersectyO[1] < yOTop:
                        yOTop = intersectyO[1]
                        yOTopLine = [[x1,y1],[x2,y2]]
                    if intersectyO[1] > yOBottom:
                        yOBottom = intersectyO[1]
                        yOBottomLine = [[x1,y1],[x2,y2]]

                if intersectxF is not None:
                    if intersectxF[0] < xFLeft:
                        xFLeft = intersectxF[0]
                        xFLeftLine = [[x1,y1],[x2,y2]]
                    if intersectxF[0] > xFRight:
                        xFRight = intersectxF[0]
                        xFRightLine = [[x1,y1],[x2,y2]]
                if intersectyF is not None:
                    if intersectyF[1] < yFTop:
                        yFTop = intersectyF[1]
                        yFTopLine = [[x1,y1],[x2,y2]]
                    if intersectyF[1] > yFBottom:
                        yFBottom = intersectyF[1]
                        yFBottomLine = [[x1,y1],[x2,y2]]
       # Top line has margin of error that effects all courtmapped outputs 
        yOTopLine[0][1] = yOTopLine[0][1]+4
        yOTopLine[1][1] = yOTopLine[1][1]+4

        yFTopLine[0][1] = yFTopLine[0][1]+4
        yFTopLine[1][1] = yFTopLine[1][1]+4

        # Find four corners of the court and display it
        topLeftP = findIntersection(xOLeftLine, yOTopLine, -extraLen, 0, width+extraLen, height)
        topRightP = findIntersection(xORightLine, yFTopLine, -extraLen, 0, width+extraLen, height)
        bottomLeftP = findIntersection(xFLeftLine, yOBottomLine, -extraLen, 0, width+extraLen, height)
        bottomRightP = findIntersection(xFRightLine, yFBottomLine, -extraLen, 0, width+extraLen, height)

        # If all corner points are different or something not found, rerun print
        if (not(topLeftP == NtopLeftP)) and (not(topRightP == NtopRightP)) and (not(bottomLeftP == NbottomLeftP)) and (not(bottomRightP == NbottomRightP)):
            

            if(NtopLeftP == None or abs(np.array(NtopLeftP) - np.array(topLeftP)) < threshold) : 
                NtopLeftP = topLeftP
            if(NtopRightP == None or abs(np.array(NtopRightP) - np.array(topRightP)) < threshold) : 
                NtopRightP = topRightP
            if(NbottomLeftP == None or abs(np.array(NbottomLeftP) - np.array(bottomLeftP)) < threshold) : 
                NbottomLeftP = bottomLeftP
            if(NbottomRightP == None or abs(np.array(NbottomRightP) - np.array(bottomRightP)) < threshold) : 
                NbottomRightP = bottomRightP
            
            if frm is not None : 
                cv2.line(frm, NtopLeftP, NtopRightP, (0, 0, 255), 2)
                cv2.line(frm, NbottomLeftP, NbottomRightP, (0, 0, 255), 2)
                cv2.line(frm, NtopLeftP, NbottomLeftP, (0, 0, 255), 2)
                cv2.line(frm, NtopRightP, NbottomRightP, (0, 0, 255), 2)

                cv2.circle(frm, NtopLeftP, radius=0, color=(255, 0, 255), thickness=10)
                cv2.circle(frm, NtopRightP, radius=0, color=(255, 0, 255), thickness=10)
                cv2.circle(frm, NbottomLeftP, radius=0, color=(255, 0, 255), thickness=10)
                cv2.circle(frm, NbottomRightP, radius=0, color=(255, 0, 255), thickness=10)

            points_array = [NtopLeftP, NtopRightP, NbottomRightP, NbottomLeftP]
            # Calculate homography
            homography_matrix = calculate_homography(np.array(points_array), points_array, field_length, field_width)
            
            NtopRightP = np.array([[NtopRightP[0], NtopRightP[1]]], dtype=np.float32)
            NtopRightP = np.reshape(NtopRightP, (1,1,2))
            NtopLeftP = np.array([[NtopLeftP[0], NtopLeftP[1]]], dtype=np.float32)
            NtopLeftP = np.reshape(NtopLeftP, (1,1,2))
            NbottomLeftP = np.array([[NbottomLeftP[0], NbottomLeftP[1]]], dtype=np.float32)
            NbottomLeftP = np.reshape(NbottomLeftP, (1,1,2))
            NbottomRightP = np.array([[NbottomRightP[0], NbottomRightP[1]]], dtype=np.float32)
            NbottomRightP = np.reshape(NbottomRightP, (1,1,2))
                
            real_NtopRightP = cv2.perspectiveTransform(NtopRightP,homography_matrix)
            real_NtopLeftP = cv2.perspectiveTransform(NtopLeftP,homography_matrix)
            real_NbottomLeftP = cv2.perspectiveTransform(NbottomLeftP,homography_matrix)
            real_NbottomRightP= cv2.perspectiveTransform(NbottomRightP,homography_matrix)
            ratio =  np.linalg.norm(np.array(real_NtopRightP) - np.array(real_NtopLeftP)) /field_width
            ratio2 =  np.linalg.norm(np.array(real_NtopRightP) - np.array(real_NbottomRightP))/field_length
            print("RATIO px/meter: " + str(ratio)+ " . "  + str(ratio2))
        
            return homography_matrix, points_array, ratio

        else:
             if frm is not None : 
                cv2.line(frm, NtopLeftP, NtopRightP, (0, 0, 255), 2)
                cv2.line(frm, NbottomLeftP, NbottomRightP, (0, 0, 255), 2)
                cv2.line(frm, NtopLeftP, NbottomLeftP, (0, 0, 255), 2)
                cv2.line(frm, NtopRightP, NbottomRightP, (0, 0, 255), 2)
            
                cv2.circle(frm, NtopLeftP, radius=0, color=(255, 0, 255), thickness=10)
                cv2.circle(frm, NtopRightP, radius=0, color=(255, 0, 255), thickness=10)
                cv2.circle(frm, NbottomLeftP, radius=0, color=(255, 0, 255), thickness=10)
                cv2.circle(frm, NbottomRightP, radius=0, color=(255, 0, 255), thickness=10)

def select_points(image_path):
    # Display the image and allow the user to select points
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.title("Select points by clicking, press any key when done.")
    points = plt.ginput(n=-1, timeout=0, show_clicks=True)
    plt.close()
    return np.array(points)         

def calculate_homography(image_points, field_points, field_length, field_width):
    # Define the corresponding points in the field
    # The order is fundamental : if this is the final order, you have to choose the points in the given image in this way A->B->C->D
    # A-----------B
    # |           |
    # |           |
    # |           |
    # |           |
    # |           |
    # |           |
    # D-----------C
    #offset to move the origin 
    offset = 150
    field_corners = np.array([[offset + 0, offset + 0], [offset + field_width, offset +  0], [offset + field_width, offset +  field_length],  [offset + 0,offset +  field_length]], dtype=np.float32)

    # Calculate the homography
    homography, _ = cv2.findHomography(image_points, field_corners)

    return homography

def get_total_frames(video_path):
    clip = VideoFileClip(video_path)
    total_frames = clip.reader.nframes
    clip.close()
    return total_frames

def interpolate_missing_values(coords):
    jump = 5
    interval = jump*2
    numofcoords = len(coords)
    #print(f"\nTO INTERPOLATE ON: {numofcoords} coordinates")
    interpolated = []

    index = 0
    while coords[index] == (0,0):
        index += 1
        #print("\nJUMP\n")

    index += jump
    oversampling = 0
    k = 0
    while index+jump+oversampling-1 < numofcoords:
        #print(f"\nTO INTERPOLATE ON: {numofcoords} coordinates")
        #print(f"\nINTERPOLATION LOOP {k}\n")
        #print(f"Index: {index}\n")
        #print(f"Oversampling: {oversampling}\n")
        
        k += 1
        subpositions = []
        for x in range(index-jump, index+jump+oversampling):
            subpositions.append(coords[x]) #interval array from coords -> ..........,[X,X,X,X,X,JUMP,X,X,X,X,OVERSAMPLING],............

        #print(f"Length of subpos: {len(subpositions)}\n")

        #for x in subpositions:
        #    print(x)

        if all((item != (0,0)) for item in subpositions): #Skip if all values in interval exist already (no interpolation necessary)
            oversampling = 0
            index += jump
            #print("\n--> ALL VALUES THERE\n")
            continue

        all_zeros = True
        #print("\nsubposition considered:")
        #for x in range(index, index+jump+oversampling):
        #    print(subpositions[x-index])

        for h in range(index, index+jump+oversampling): #If all 5 new samples analyzed are null (interpolation might be imprecise), restart interval interpolation considering one more sample appended on the right
            if subpositions[h-index+jump] != (0,0):
                #print("\nTrue value detected")
                all_zeros = False
                break
        if all_zeros:
            oversampling += 1
            #print("\n--> TOO MANY ZEROS => OVERSAMPLING\n")
            continue
        

        # Estraction of non-zero values to create interpolation function
        non_zero_coords = [(x, y) for x, y in subpositions if (x, y) != (0, 0)]
        #zero_indices = [i for i, point in subpositions if point == (0, 0)]
        
        x_values = [x for x, _ in non_zero_coords]
        y_values = [y for _, y in non_zero_coords] 

        #print("\nMARK 1")
        #for c in non_zero_coords:
        #    print(c)

        # Creation of time axis for the considered interval
        t_axis = []
        #print("\nMARK 2")
        t_inst = 0
        lenghtsubpos = len(subpositions)
        while t_inst < lenghtsubpos:
            if subpositions[t_inst] != (0,0):
                t_axis.append(t_inst)    
            t_inst += 1

        zero_indices = []
        t_inst = 0
        while t_inst < lenghtsubpos:
            if subpositions[t_inst] == (0,0):
                zero_indices.append(t_inst)    
            t_inst += 1

        #print(f"Lent: {len(t_axis)}")
        #print(f"Lenx: {len(x_values)}")
        #print(f"Leny: {len(y_values)}")

        #Creation of function over x values
        x_interp_func = interp1d(t_axis, x_values, kind='slinear', fill_value='extrapolate') #spline interpolation function

        #Creation of function over y values
        y_interp_func = interp1d(t_axis, y_values, kind='slinear', fill_value='extrapolate') #spline interpolation function

        # Interpolation of missing (zero) values
        for i in zero_indices:
            x_interp_value = int(x_interp_func(i))
            y_interp_value = int(y_interp_func(i))
            coords[i+index-jump] = (x_interp_value, y_interp_value) 
            #print(f"Result of interpolation: {x_interp_value}, {y_interp_value}\n")
        
        for i in zero_indices: #Ottimizzabile
            if i+index-jump not in interpolated:
                interpolated.append(i+index-jump)

        oversampling = 0
        index += jump

    return interpolated

#def detect_racket_hits(ball_positions, tophead_positions, bottomhead_positions):
#def detect_racket_hits(ball_positions, rightwrist_positions_top, leftwrist_positions_top, rightwrist_positions_bot, leftwrist_positions_bot, height_values_top, height_values_bot):
    hits = []
    window = 49
    poly_order = 2

    ball_positions_array = np.array(ball_positions)
    y_positions = ball_positions_array[:, 1]
    smoothed_y_positions = savgol_filter(y_positions, window, poly_order)

    velocities = savgol_filter(y_positions, window, poly_order, deriv=1)  # Calcolo della derivata rispetto all'asse verticale
    accelerations = np.gradient(velocities)

    plotgraph(smoothed_y_positions, "Frame", "y Position", "pC.jpg")
    plotgraph(velocities, "Frame", "y Velocity", "vC.jpg")
    plotgraph(accelerations, "Frame", "y Accelerations", "aC.jpg")
    #for i in range(3, len(ball_positions)-3):
        #if (velocities[i] >= 0 and velocities[i-1] < 0) or (velocities[i] < 0 and velocities[i-1] >= 0):
        #   if velocities[i+1] >= 0 and velocities[i-2] < 0 or velocities[i+1] < 0 and velocities[i-2] >= 0:
        #        hits.append(i)

    sign_changes = np.where(np.diff(np.sign(velocities)))[0]

    for i in sign_changes:
        print("Detected racket hit: ")
        print(i)
        for j in range(i-15, i+15):
            if i+10 > len(ball_positions_array):
                break
            ball_pos = ball_positions_array[j]
            
            # Distanza dai polsi del giocatore in alto
            dist_right_top = np.linalg.norm(ball_pos - np.array(rightwrist_positions_top[j]))
            dist_left_top = np.linalg.norm(ball_pos - np.array(leftwrist_positions_top[j]))
            
            # Distanza dai polsi del giocatore in basso
            dist_right_bot = np.linalg.norm(ball_pos - np.array(rightwrist_positions_bot[j]))
            dist_left_bot = np.linalg.norm(ball_pos - np.array(leftwrist_positions_bot[j]))

            if (dist_right_top < height_values_top[j]*10 or dist_left_top < height_values_top[j]*10 or
                dist_right_bot < height_values_bot[j]*10 or dist_left_bot < height_values_bot[j]*10):
                hits.append(j)
                break

    #print("GRADIENT ON Y:")
    #print(velocities)
    #print("DETECTED HITS")
    #print(hits)
    return hits, velocities

def detect_racket_hits(ball_positions, rightwrist_positions_top, leftwrist_positions_top, rightwrist_positions_bot, leftwrist_positions_bot, height_values_top, height_values_bot):
    hits = []
    window = 49
    poly_order = 2
    global res_height
    global res_width

    ball_positions_array = np.array(ball_positions)
    y_positions = ball_positions_array[:, 1]
    smoothed_y_positions = savgol_filter(y_positions, window, poly_order)

    velocities = savgol_filter(y_positions, window, poly_order, deriv=1)  # Calcolo della derivata rispetto all'asse verticale
    accelerations = np.gradient(velocities)

    plotgraph(smoothed_y_positions, "Frame", "y Position", "pC.jpg")
    plotgraph(velocities, "Frame", "y Velocity", "vC.jpg")
    plotgraph(accelerations, "Frame", "y Accelerations", "aC.jpg")
    #for i in range(3, len(ball_positions)-3):
        #if (velocities[i] >= 0 and velocities[i-1] < 0) or (velocities[i] < 0 and velocities[i-1] >= 0):
        #   if velocities[i+1] >= 0 and velocities[i-2] < 0 or velocities[i+1] < 0 and velocities[i-2] >= 0:
        #        hits.append(i)

    sign_changes = np.where(np.diff(np.sign(velocities)))[0]

    for i in sign_changes:
        print("Detected racket hit: ")
        print(i)
        append_flag = False
        min_player_distance = float('inf')
        frame_ball_closest_to_player = i
        default_minimum_radius = 100 # pixel
        window = 20

        for j in range(max(0, i-window), min(len(ball_positions_array), i+window)):
            if j >= len(ball_positions_array):  # Assicurarsi di non superare i limiti dell'array
                break
            ball_pos = ball_positions_array[j]
            if np.array_equal(ball_pos, np.array([0, 0])):
                break
            
            # Distanza dai polsi del giocatore in alto
            dist_right_top = np.linalg.norm(ball_pos - np.array(rightwrist_positions_top[j]))
            dist_left_top = np.linalg.norm(ball_pos - np.array(leftwrist_positions_top[j]))
            
            # Distanza dai polsi del giocatore in basso
            dist_right_bot = np.linalg.norm(ball_pos - np.array(rightwrist_positions_bot[j]))
            dist_left_bot = np.linalg.norm(ball_pos - np.array(leftwrist_positions_bot[j]))

            if (dist_right_top < max(height_values_top[j]*10, default_minimum_radius) or dist_left_top < max(height_values_top[j]*10, default_minimum_radius) or
                dist_right_bot < max(height_values_bot[j]*10, default_minimum_radius) or dist_left_bot < max(height_values_bot[j]*10, default_minimum_radius)):
                append_flag = True
                distances_min = min(dist_right_top, dist_left_top, dist_right_bot, dist_left_bot)
                if distances_min < min_player_distance:
                    min_player_distance = distances_min
                    frame_ball_closest_to_player = j
            
        if append_flag:
                hits.append(frame_ball_closest_to_player)

    #print("GRADIENT ON Y:")
    #print(velocities)
    #print("DETECTED HITS")
    #print(hits)
    return hits, velocities

def plotgraph(values, xlabel, ylabel, filename):
    plt.plot(values, color='tab:blue', marker='.')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(filename, format='jpg', dpi=300)
    plt.close()

def detect_cropped_frames(video_cap):
    width = int(video_cap.get(3))
    height = int(video_cap.get(4))

    # Get the FPS of the video
    video_file_fps = video_cap.get(cv2.CAP_PROP_FPS)

    # Initialize frame index
    frame_index = 0

    playerDetection = PlayersDetections()
    # Initialize the detector
    detector = playerDetection.getDetector()  # Use your specific detection module

    det_bot = None
    det_top = None
    det_bot_prev = None
    det_top_prev = None
    threshold = 20

    min_y_top = None
    max_y_top = None
    min_x_top = None 
    max_x_top = None


    min_y_bot = None
    max_y_bot = None
    min_x_bot = None
    max_x_bot = None
    # Loop through the video frames
    while cv2.waitKey(1) < 0:
        # Read a frame from the video
        success, frame = cap.read()
        if not success:
            break  # Break the loop when no more frames are available
        
        # Calculate the timestamp of the current frame          
        frame_timestamp_ms = 1000 * frame_index / video_file_fps

        # Convert the frame received from OpenCV to a MediaPipe’s Image object.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # Perform object detection on the video frame.
        detection_result = detector.detect_for_video(mp_image, int(frame_timestamp_ms))
        detection_result = playerDetection.filterDetections(width/2, detection_result)

        # assigning to each player his detection
        for det in detection_result: 
            bbox = det.bounding_box
            y = bbox.origin_y
            if y < height/2 : 
                det_top = det 
            else :
                det_bot = det

        # Eliminating detection out of threshold - should be optimized by interpolation
        if det_top_prev == None or abs(det_top.bounding_box.origin_x - det_top_prev.bounding_box.origin_x) < threshold :
            det_top_prev = det_top
        if det_bot_prev == None or abs(det_bot.bounding_box.origin_x - det_bot_prev.bounding_box.origin_x) < threshold :
            det_bot_prev = det_bot

        if min_y_top is None or min_y_top > det_top_prev.bounding_box.origin_y : 
            min_y_top = det_top_prev.bounding_box.origin_y
        if max_y_top is None or max_y_top < det_top_prev.bounding_box.origin_y + det_top_prev.bounding_box.height : 
            max_y_top = det_top_prev.bounding_box.origin_y + det_top_prev.bounding_box.height
        if min_x_top is None or min_x_top > det_top_prev.bounding_box.origin_x : 
            min_x_top = det_top_prev.bounding_box.origin_x
        if max_x_top is None or max_x_top < det_top_prev.bounding_box.origin_x + det_top_prev.bounding_box.width: 
            max_x_top = det_top_prev.bounding_box.origin_x+ det_top_prev.bounding_box.width

        if min_y_bot is None or min_y_bot > det_bot_prev.bounding_box.origin_y : 
            min_y_bot = det_bot_prev.bounding_box.origin_y
        if max_y_bot is None or max_y_bot < det_bot_prev.bounding_box.origin_y + det_bot_prev.bounding_box.height :
            max_y_bot = det_bot_prev.bounding_box.origin_y + det_bot_prev.bounding_box.height
        if min_x_bot is None or min_x_bot > det_bot_prev.bounding_box.origin_x : 
            min_x_bot = det_bot_prev.bounding_box.origin_x
        if max_x_bot is None or max_x_bot < det_bot_prev.bounding_box.origin_x + det_bot_prev.bounding_box.width: 
            max_x_bot = det_bot_prev.bounding_box.origin_x+ det_bot_prev.bounding_box.width

        final_det = [det_top_prev, det_bot_prev]
        playerDetection.visualize(frame, final_det)
        cv2.imshow("Title", frame)

        # Increment frame index for the next iteration
        frame_index += 1

    # Release the video capture object
    cap.release()
    cv2.destroyAllWindows()
    #adding 5 to have tolerance
    return min_y_bot + 5, max_y_bot + 5, min_x_bot + 5 , max_x_bot + 5 , min_y_top + 5 , max_y_top + 5 , min_x_top + 5 , max_x_top + 5


############################################################################################################
####################################### TEST SECTION #######################################################

# Step 1: Camera Calibration using DLT
#def calibrate_camera(points_2d, points_3d):
#    assert len(points_2d) == len(points_3d) and len(points_2d) >= 6
#    A = []
#    for i in range(len(points_2d)):
#        X, Y, Z = points_3d[i]
#        u, v = points_2d[i]
#        A.append([X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z])
#        A.append([0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z])
#    A = np.array(A)
#    U, S, Vt = np.linalg.svd(A)
#    P = Vt[-1].reshape(3, 4)
#    return P

def calibrate_camera(points_2d, points_3d):
    assert len(points_2d) == len(points_3d) and len(points_2d) >= 6
    A = []
    for i in range(len(points_2d)):
        X, Y, Z = points_3d[i]
        u, v = points_2d[i]
        A.append([X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u])
        A.append([0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v])
    A = np.array(A)
    U, S, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape(3, 4)
    return P


# Step 2: Map 2D points to 3D using the inverse projection matrix
#def map_2d_to_3d(P, points_2d):
#    points_3d = []
#    P_inv = np.linalg.pinv(P)
#    for u, v in points_2d:
#        X, Y, Z, _ = P_inv @ np.array([u, v, 1])
#        points_3d.append((X/Z, Y/Z, 1))
#    return np.array(points_3d)

def map_2d_to_3d(P, points_2d):
    points_3d = []
    P_inv = np.linalg.pinv(P)
    for u, v in points_2d:
        X, Y, Z, W = P_inv @ np.array([u, v, 1])
        points_3d.append((X/W, Y/W, Z/W))
    return np.array(points_3d)

# Step 3: Initial trajectory estimation
def estimate_trajectory(initial_position, initial_velocity, time, g=9.8):
    X = initial_position[0] + time * initial_velocity[0]
    Y = initial_position[1] + time * initial_velocity[1]
    Z = initial_position[2] + time * initial_velocity[2] - 0.5 * g * time**2
    return X, Y, Z

# Step 4: Improved trajectory using geometric constraints
def compute_trajectory(P, points_2d, initial_params, touch_points, w=1.0):
    def cost_function(params):
        initial_position = params[:3]
        initial_velocity = params[3:]
        F_classic = 0
        F_touch = 0
        for t, (u, v) in enumerate(points_2d):
            X, Y, Z = estimate_trajectory(initial_position, initial_velocity, t)
            u_proj, v_proj, _ = P @ np.array([X, Y, Z, 1])
            u_proj /= u_proj[2]
            v_proj /= v_proj[2]
            F_classic += (u - u_proj[0])**2 + (v - v_proj[1])**2
            if t in touch_points:  # Add touch point constraint
                X_touch, Y_touch, Z_touch = compute_touch_point(P, u, v)
                F_touch += (X - X_touch)**2 + (Y - Y_touch)**2 + (Z - Z_touch)**2
        return F_classic + w * F_touch

    result = minimize(cost_function, initial_params, method='L-BFGS-B')
    return result.x

# Auxiliary function to compute touch points
def compute_touch_point(P, u, v):
    # Assuming Z=0 for the court plane
    A = np.array([
        [P[0,0] - u*P[2,0], P[0,1] - u*P[2,1], P[0,2] - u*P[2,2]],
        [P[1,0] - v*P[2,0], P[1,1] - v*P[2,1], P[1,2] - v*P[2,2]]
    ])
    b = np.array([u*P[2,3] - P[0,3], v*P[2,3] - P[1,3]])
    X, Y, Z = np.linalg.lstsq(A, b, rcond=None)[0]
    return X, Y, Z

##########################################################################################################
##########################################################################################################

############################### MAIN ####################################

# Field Dimensions (scaled)
field_length_real = 23.78 #meters
field_width_real = 10.97 #meters
net_post_real = 1.07 #meters, height of the sides of the net

scale_factor = 20 #pixel per meter (?)
field_length = field_length_real * scale_factor
field_width = field_width_real * scale_factor
net_post = net_post_real * scale_factor

# Field (x,y,z) relevant points
#    3____________________________4
#    |                            |
#    |                            |
#    |                            |
#    |                            |
#    |                            |
#    |                            |
#    |                            |
#  2________________________________5
#    |                            |
#    |                            |
#    |                            |
#    |                            |
#    |                            |
#    |                            |
#    |                            |
#    1____________________________6
# (X)
# 
# (X) is where the reference system is placed, so that point 1 is at x = 5 meters and y = 5 meters 

reference_offset = (5,5,0) # offset between (X) and PT1
field_pt1 = (reference_offset[1], reference_offset[2], 0)
field_pt2 = (reference_offset[1] - 0.91, reference_offset[2] + 11.83, 1.07)
field_pt3 = (reference_offset[1], reference_offset[2] + 23.78, 0)
field_pt4 = (reference_offset[1] + 10.97, reference_offset[2] + 23.78, 0)
field_pt5 = (reference_offset[1] + 10.97 + 0.91, reference_offset[2] + 11.83, 1.07)
field_pt6 = (reference_offset[1] + 10.97, reference_offset[2], 0)

# Load an image
image_path = "resources/frame.JPG"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #set the color from BGR to RGB

###### Ball trajectory util
# Buffers to use as queues in threads
positions_stack = []
rightwrist_stack_top = []
leftwrist_stack_top = []
rightwrist_stack_bot = []
leftwrist_stack_bot = []
realposition_buffer = []
height_top_buffer = []
height_bot_buffer = []

# Arrays for positions wrt frames
ball_positions = [] #array of the trajectory in the image
ball_positions_real = [] #array of the top-view trajectory in the image
rightwrist_positions_top = []
leftwrist_positions_top = []
rightwrist_positions_bot = []
leftwrist_positions_bot = []
height_values_top = []
height_values_bot = []

prevpos = (0,0)
lastvalidpos = (0,0)
pos_counter = 0
sequence = False
beginning = True

############## Task 2 ###################
# Font characteristics for the coordinates display 
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
color = (255,255,255)
thickness = 1
############## Task 2 ###################

############## Task 3 ###################
# Initialization of the variables that will retain the previous position of the feet + threshold for the detection of movement
prev_PleftA_image = [0,0]
prev_PrightA_image =[0,0]
prev_PleftB_image = [0,0]
prev_PrightB_image =[0,0]
############## Task 3 ###################

# Loading of the clip to analyze
video_path = "resources/tennis2full.mp4"
total_frames = get_total_frames(video_path)
cap = cv2.VideoCapture(video_path)


# Calculate homography taking the first frame of the video
#field_points_2D = []
homography_matrix, field_points_2D, ratiopxpermtr = autoComputeHomography(cap,None, None, None, None, None)

#since the field width has been scaled by scale_factor, the real ratio has to be multiplied by scale_factor
ratiopxpermtr = ratiopxpermtr * scale_factor
mpPose_A = mp.solutions.pose
pose_A = mpPose_A.Pose()
mpDraw_A = mp.solutions.drawing_utils

mpPose_B = mp.solutions.pose
pose_B = mpPose_B.Pose()
mpDraw_B = mp.solutions.drawing_utils

# PROCESSING LOOP
# each landmark has an id - https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
# ids 28 and 27 are for right and left ankle
# lists of the static points of the two players (A -> DOWN , B -> UP)
stationary_points_bot = list()
stationary_points_top = list()

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #set the color from BGR to RGB

#Rectifying the image using the homography matrix found previously
#changing the rectified image to "clean" it from the previous drawings of the center
rectified_image = cv2.warpPerspective(image, homography_matrix, (image.shape[1], image.shape[0]))

# Allocation to write the resulting evaluation in a video file at the end
# Maybe width has to be changed : TODO
result = cv2.VideoWriter('raw.mp4',
                         cv2.VideoWriter_fourcc(*'mp4v'),
                         60, (image.shape[1] + rectified_image.shape[1], 720))

ball_detector = BallDetector('TRACE/TrackNet/Weights.pth', out_channels=2)

#parameters for comparison in court detection
NtopLeftP = None
NtopRightP = None
NbottomLeftP = None
NbottomRightP = None
startofprocessing = True
res_height = 0
res_width = 0
rect_height = 0

min_y_bot_pl, max_y_bot_pl, min_x_bot_pl, max_x_bot_pl, min_y_top_pl, max_y_top_pl, min_x_top_pl, max_x_top_pl = detect_cropped_frames(cap)

cap = cv2.VideoCapture(video_path)

##################################################################################################################################
##################################### Calibration and 3D Estimation Test  ########################################################

#selected = select_points(image_path)
#print(selected)

#reference_offset = (5, 5, 0)  # offset between (X) and PT1
#field_pt1 = (reference_offset[1], reference_offset[2], 0)
#field_pt2 = (reference_offset[1] - 0.91, reference_offset[2] + 11.83, 1.07)
#field_pt3 = (reference_offset[1], reference_offset[2] + 23.78, 0)
#field_pt4 = (reference_offset[1] + 10.97, reference_offset[2] + 23.78, 0)
#field_pt5 = (reference_offset[1] + 10.97 + 0.91, reference_offset[2] + 11.83, 1.07)
#field_pt6 = (reference_offset[1] + 10.97, reference_offset[2], 0)

#points_2d = np.array([
#    [field_points_2D[3][0], field_points_2D[3][1]], 
#    selected[0], 
#    [field_points_2D[0][0], field_points_2D[0][1]], 
#    [field_points_2D[1][0], field_points_2D[1][1]], 
#    selected[1], 
#    [field_points_2D[2][0], field_points_2D[2][1]]
#])

#points_3d = np.array([field_pt1, field_pt2, field_pt3, field_pt4, field_pt5, field_pt6])

# Step 1: Calibrate the camera
#P = calibrate_camera(points_2d, points_3d)

# Step 2: Map 2D ball coordinates to 3D
#ball_2d_coords = [(490, 200), (495, 205)]  # 2D coordinates of the ball from video frames
#ball_3d_coords = map_2d_to_3d(P, ball_2d_coords)

# Step 3: Estimate the trajectory
#initial_params = [0, 0, 2, 30, 0, 10]  # Initial guess for position and velocity (2 meters height of the ball, 30m/s on x and 10 m/s on z)
#touch_points = {0: (0, 2.47, 0), 10: (1.525, 2.47, 0)}  # Example touch points
#touch_points = {}

#trajectory_params = compute_trajectory(P, ball_2d_coords, initial_params, touch_points)

# Extract and print top view coordinates (X, Y)
#for t in range(len(ball_2d_coords)):
#    X, Y, Z = estimate_trajectory(trajectory_params[:3], trajectory_params[3:], t)
#    print(f"Top view coordinates at t={t}: X={X}, Y={Y}")

######################################################################################################################################################

# First Main Loop
print("\nBall positions detected:")
i=0
computeMoving = False
threshold_moving = 5
counter = 0
dist_top = 0
dist_bot = 0
while cv2.waitKey(1) < 0:

    if counter%5 == 0 : 
        computeMoving = True
    else : 
        print(str(i))
        computeMoving = False

    counter +=1
    hasFrame, frame = cap.read()
    if not hasFrame:
        # cv2.waitKey()
        break
    if startofprocessing: #We extract the source resolution from the first available frame
        res_height, res_width, _ = frame.shape 
        startofprocessing = False
    
    cTime = time.time()

    #no item returned since it is just to show live court detection (too noisy to make live computation of homography)
    #autoComputeHomography(cap, frame, NtopLeftP, NtopRightP, NbottomLeftP, NbottomRightP)
    #changing the rectified image to "clean" it from the previous drawings of the center
    rectified_image = cv2.warpPerspective(image, homography_matrix, (image.shape[1], image.shape[0]))

    cropped_frame_top = frame[min_y_top_pl:max_y_top_pl, min_x_top_pl:max_x_top_pl].copy()
    cropped_frame_bot = frame[min_y_bot_pl:max_y_bot_pl, min_x_bot_pl:max_x_bot_pl].copy()
    
    #creating two threads to improve performances for the detection of the pose
    th_A = threading.Thread(target=computePoseAndAnkles, args=(cropped_frame_bot, stationary_points_bot, mpPose_A, pose_A, mpDraw_A, homography_matrix, prev_PrightA_image, prev_PleftA_image, threshold_moving, min_x_bot_pl, min_y_bot_pl, rectified_image, rightwrist_stack_bot, leftwrist_stack_bot, height_bot_buffer))
    th_B = threading.Thread(target=computePoseAndAnkles, args=(cropped_frame_top, stationary_points_top, mpPose_B, pose_B, mpDraw_B, homography_matrix, prev_PrightB_image, prev_PleftB_image, threshold_moving, min_x_top_pl,  min_y_top_pl, rectified_image, rightwrist_stack_top, leftwrist_stack_top, height_top_buffer))
    #th_C = threading.Thread(target=processBallTrajectory, args=(ball_detector, frame, positions_stack))
         
    #stationary_points_bot = stationary_points_bot[::60]
    #stationary_points_top = stationary_points_top[::60]

    th_A.start()
    th_B.start()
    #th_C.start()
    th_A.join()
    th_B.join()
    #th_C.join()

    #ballpos = positions_stack.pop()

    rightwrist_top = rightwrist_stack_top.pop()
    leftwrist_top = leftwrist_stack_top.pop()
    rightwrist_bot = rightwrist_stack_bot.pop()
    leftwrist_bot = leftwrist_stack_bot.pop()

    height_top = height_top_buffer.pop()
    height_bot = height_bot_buffer.pop()
    height_str_top = f"Top Height: {height_top}"
    height_str_bot = f"Bottom Height: {height_bot}"
    print(height_str_top)
    print(height_str_bot)

    rightwrist_positions_top.append(rightwrist_top)
    leftwrist_positions_top.append(leftwrist_top)
    rightwrist_positions_bot.append(rightwrist_bot)
    leftwrist_positions_bot.append(leftwrist_bot)
    height_values_top.append(height_top)
    height_values_bot.append(height_bot)

    pTime = time.time()

    #fps = 1/(cTime-pTime)
    fps = 1/(pTime-cTime)
    
    frame[min_y_bot_pl:max_y_bot_pl, min_x_bot_pl:max_x_bot_pl] = cropped_frame_bot
    frame[min_y_top_pl:max_y_top_pl, min_x_top_pl:max_x_top_pl] = cropped_frame_top

    #ballpos_real = (0,0)
    #if ballpos != (0,0):
    #    cv2.circle(frame, ballpos, 5, (0, 255, 0), cv2.FILLED)
    #    #ballpos_array = np.array([[ballpos[0], ballpos[1]]], dtype=np.float32)
    #    #ballpos_array = np.reshape(ballpos_array, (1,1,2))
    #    #transformedpos = map_2d_to_3d(P, np.array([ballpos]))
    #    #ballpos_real = (round(transformedpos[0][0]), round(transformedpos[0][1]))
    #    #cv2.circle(rectified_image, ballpos_real , 5, (255, 255, 0), cv2.FILLED)
    #ball_positions.append(ballpos)
    ##ball_positions_real.append(ballpos_real)

    percent = i/total_frames*100
    #print(f"FRAME {i}: {ballpos}; - {percent:.1f}%")

    # Putting ball position into perspective
    #real_ball_pos = cv2.perspectiveTransform(ballpos, homography_matrix)
    #ball_positions_real.append(real_ball_pos)

    frame_str = f"Frame: {i}"
    cv2.putText(frame, str(frame_str), (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255), 3)

    # Appending the perspective image on the side
    height = max(frame.shape[0], rectified_image.shape[0])
    rect_height = height
    frame = cv2.resize(frame, (int(frame.shape[1] * height / frame.shape[0]), height))
    rectified_image = cv2.resize(rectified_image, (int(rectified_image.shape[1] * height / rectified_image.shape[0]), height))
    rectified_image = cv2.cvtColor(rectified_image, cv2.COLOR_RGB2BGR)
    
    if len(stationary_points_bot)!=0 : 
        dist_bot = 0
        cv2.circle(rectified_image, stationary_points_bot[0], 2, (125, 125, 125), cv2.FILLED)
        for i in range(1,len(stationary_points_bot)):
            cv2.circle(rectified_image, stationary_points_bot[i], 2, (125, 125, 125), cv2.FILLED)
            cv2.line(rectified_image, stationary_points_bot[i-1],stationary_points_bot[i], (125,125,125), 3)
            dist_bot += np.linalg.norm(np.array(stationary_points_bot[i]) - np.array(stationary_points_bot[i-1]))/ratiopxpermtr
    print("LEN BOT " + str(len(stationary_points_bot)))
    dist_bot = np.trunc(dist_bot)
    cv2.putText(frame, "Bottom Player Distance : " + str(dist_bot)+" m", (50,80), cv2.FONT_HERSHEY_SIMPLEX,0.5,(70,150,255), 1)
    
    if len(stationary_points_top)!=0 : 
        dist_top = 0
        cv2.circle(rectified_image, stationary_points_top[0], 2, (125, 125, 125), cv2.FILLED)
        for i in range(1,len(stationary_points_top)):
            cv2.circle(rectified_image, stationary_points_top[i], 2, (125, 125, 125), cv2.FILLED)
            cv2.line(rectified_image, stationary_points_top[i-1],stationary_points_top[i], (125,125,125), 3)
            dist_top += np.linalg.norm(np.array(stationary_points_top[i]) - np.array(stationary_points_top[i-1]))/ratiopxpermtr
    
    dist_top= np.trunc(dist_top)
    cv2.putText(frame, "Top Player Distance : " + str(dist_top)+" m", (50,110), cv2.FONT_HERSHEY_SIMPLEX,0.5,(70,150,255), 1)

    combined_image = cv2.hconcat([frame, rectified_image])
    cv2.imshow('Combined Images', combined_image)
    result.write(combined_image)

    i += 1
    #if i > 300: break #FOR TESTING ON LIMITED INTERVAL

cap.release()
result.release()
cv2.destroyAllWindows()

#print("DETECTED POSITIONS")
#for l in ball_positions:
#    print(l)

#if i < total_frames:
#    print("Execution stopped by user")
#    sys.exit()

#interpolated_samples = interpolate_missing_values(ball_positions)
#print("\nInterpolation:\n")
#for r in interpolated_samples:
#    print(f"FRAME: {r}")
#    print(f"--> {ball_positions[r]}")
#
#cap = cv2.VideoCapture("raw.mp4")
#
#result = cv2.VideoWriter('processed.mp4',
#                         cv2.VideoWriter_fourcc(*'mp4v'),
#                         60, (image.shape[1] + rectified_image.shape[1], 720))
#
#print("\nInterpolation Completed. Drawing...\n")
#
#j = 0
#while cv2.waitKey(1) < 0:
#    
#    hasFrame, frame = cap.read()
#    if not hasFrame:
#        break
#    
#    percent = j/i*100
#    print(f"{percent:.1f}%")
#
#    #if j in interpolated_samples:
#
#        #Regular Pitch View: adding of interpolated ball position
#    #    original_frame_extr = frame[0:res_height,0:res_width]
#    #    cv2.circle(original_frame_extr, ball_positions[j], 7, (0,0,255), cv2.FILLED) 
#
#        #Top Pitch View: adding of interpolated ball position
#    #    rectified_image_extr = frame[0:res_height, res_width+1:frame.shape[1]]
#    #    interpolatedballpos = ball_positions[j]
#        #ballpos_array = np.array([[interpolatedballpos[0], interpolatedballpos[1]]], dtype=np.float32)
#        #ballpos_array = np.reshape(ballpos_array, (1,1,2))
#        #transformedpos = map_2d_to_3d(P, np.array([interpolatedballpos]))
#        #ballpos_real = (round(transformedpos[0][0]), round(transformedpos[0][1]))
#        #cv2.circle(rectified_image_extr, ballpos_real , 5, (255, 255, 0), cv2.FILLED)
#
#        #height = max(frame.shape[0], rectified_image_extr.shape[0])
#        #original_frame_extr = cv2.resize(original_frame_extr, (int(original_frame_extr.shape[1] * height / original_frame_extr.shape[0]), height))
#        #rectified_image = cv2.resize(rectified_image_extr, (int(rectified_image_extr.shape[1] * height / rectified_image_extr.shape[0]), height))
#    #    frame = cv2.hconcat([original_frame_extr, rectified_image_extr])
#        #cv2.imshow(f'Frame Interpolated: {j}', frame)
#    
#    if j in interpolated_samples:
#        cv2.circle(frame, ball_positions[j], 7, (0, 0, 255), cv2.FILLED)   
#        #cv2.imshow(f'Frame Interpolated: {j}', frame)
#
#    result.write(frame) 
#    j +=1
#cap.release()
#result.release()
#cv2.destroyAllWindows()
#print("The video was successfully processed")
#
#
#cap = cv2.VideoCapture("processed.mp4")
#
#
#
#result = cv2.VideoWriter('processed_winfo.mp4',
#                         cv2.VideoWriter_fourcc(*'mp4v'),
#                         60, (image.shape[1] + rectified_image.shape[1], 720))
#
#
#racket_hits, velocity = detect_racket_hits(ball_positions, rightwrist_positions_top, leftwrist_positions_top, rightwrist_positions_bot, leftwrist_positions_bot, height_values_top, height_values_bot)
#print("Detected racket hits:", racket_hits)
#j = 0
#hits = 0
#while cv2.waitKey(1) < 0:
#    
#    hasFrame, frame = cap.read()
#    if not hasFrame:
#        break
#    
#    percent = j/i*100
#    print(f"{percent:.1f}%")
#
#    if j > 5:
#        ypos = f"Y(curr): {ball_positions[j][1]}"
#        ypos1 = f"Y(-1): {ball_positions[j-1][1]}"
#        ypos2 = f"Y(-2): {ball_positions[j-2][1]}"
#        ypos3 = f"Y(-3): {ball_positions[j-3][1]}"
#        ypos4 = f"Y(-4): {ball_positions[j-4][1]}"
#        cv2.putText(frame, ypos, (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
#        cv2.putText(frame, ypos1, (50, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
#        cv2.putText(frame, ypos2, (50, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
#        cv2.putText(frame, ypos3, (50, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
#        cv2.putText(frame, ypos4, (50, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
#
#    vel =  f"Y speed: {velocity[j]:.2f}"
#    cv2.putText(frame, vel, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
#    
#    if j in racket_hits:
#        hits += 1
#    text_rackethits = f"Racket Hits: {hits}"
#    cv2.putText(frame, text_rackethits, (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)   
#    
#    result.write(frame)
#    j +=1
#
#cap.release()
#result.release()
#cv2.destroyAllWindows()
#print("The video was successfully processed")