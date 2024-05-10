###################################################################################################################################################################################################################
#  _____          _______      __  _____           _           _   
# |_   _|   /\   / ____\ \    / / |  __ \         (_)         | |  
#   | |    /  \ | |     \ \  / /  | |__) | __ ___  _  ___  ___| |_ 
#   | |   / /\ \| |      \ \/ /   |  ___/ '__/ _ \| |/ _ \/ __| __|
#  _| |_ / ____ \ |____   \  /    | |   | | | (_) | |  __/ (__| |_ 
# |_____/_/    \_\_____|   \/     |_|   |_|  \___/| |\___|\___|\__|
#                                                _/ |              
#                                               |__/               
# Project completed by Paolo Riva, Michelangelo Stasi, Mihai-Viorel Grecu
# This Computer Vision project aims at detecting two players involved in a tennis match through the widely-used Human Pose Detection method.
#
# The program focuses on detecting the movement performed by the players, specifically:
# 0. Identify the field lines and, knowing the field measures, find yhe homography H from field to image.
# 1. Use the well-known Human Pose Estimation  method (based on Deep Learning) to identify the articulated segments of the player.
# 2. Select the feet (end points of the leg segments) and their position Pleft and Pright in each image
# 3. Check whether the feet are static or they are moving (by checking if H^-1 Pleft and/or H^-1 Pright are constant along a short sequence).
#    If a foot is static, assume that it is placed on the ground.
# 4. Collect the time-sequence of the step points: i.e., the positions H^-1P of the feet in the instances when they were static.
# 5. In parallel, try to select the time instants when the player hits the ball eith yhe rackets, and compute statistics on the short runs between consecutive hits
# 
###################################################################################################################################################################################################################

## MODULES ########################################################
import sys
import cv2
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import threading
#from mediapipe import solutions
#from cv2 import cvtColor, COLOR_BGR2RGB, COLOR_RGB2BGR
import mediapipe as mp
import time
from TRACE.BallDetection import BallDetector
from TRACE.BallMapping import euclideanDistance, withinCircle
from moviepy.editor import VideoFileClip
##################################################################

#Global Variables
sequence = False
prevpos = (0,0)
pos_counter = 0
lastvalidpos = (0,0)
beginning = True

def computePoseAndAnkles(cropped_frame, static_centers_queue, mpPose, pose, mpDraw, hom_matrix, prev_right_ankle, prev_left_ankle, threshold, x_offset, y_offset, rect_img):

    imgRGB = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    #print(results.pose_landmarks)
    right_ankle, left_ankle = (0,0),(0,0)
    Pright_image, Pleft_image = (0,0),(0,0)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(cropped_frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w,c = cropped_frame.shape
            #print(id, lm)
            if id == 28 : 
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
            elif id == 27 :
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
            else :
                cx, cy = int(lm.x*w), int(lm.y*h)
                cv2.circle(cropped_frame, (cx, cy), 5, (255,0,0), cv2.FILLED)

    if prev_right_ankle is not None and prev_left_ankle is not None:
            
        left_foot_moved = np.linalg.norm(np.array(Pleft_image) - np.array(prev_left_ankle)) > threshold                              # Euclidean distance computation between the current Left foot position and its position in the previous frame, all compared to the chosen threshold 
        right_foot_moved = np.linalg.norm(np.array(Pright_image) - np.array(prev_right_ankle)) > threshold                           # Euclidean distance computation between the current Right foot position and its position in the previous frame, all compared to the chosen threshold                
        if left_ankle !=(0,0):                                                                                                                                # Check if the left ankle's point has been detected 
            if left_foot_moved:                                                                                                                         # Check if the left foot has moved 
                cv2.putText(cropped_frame, f"(LFoot) Moving", (left_ankle[0] +10, left_ankle[1] + 40), font, font_scale, color, thickness, cv2.LINE_AA)            # Display "(LFoot) Moving" under the player's left foot using the image coordinates of the left foot with an offset  
            else:
                #print(str(Pleft_image) + " " + str(prev_left_ankle))
                cv2.putText(cropped_frame, f"(LFoot) Static", (left_ankle[0] +10, left_ankle[1] + 40), font, font_scale, color, thickness, cv2.LINE_AA)            # Display "(LFoot) Static" under the player's left foot using the image coordinates of the left foot with an offset  
        if right_ankle!=(0,0):                                                                                                                                # Check if the right ankle's point has been detected
            if  right_foot_moved:                                                                                                                       # Check if the right foot has moved            
                cv2.putText(cropped_frame, f"(RFoot) Moving", (right_ankle[0] +10, right_ankle[1] -20 ), font, font_scale, color, thickness, cv2.LINE_AA)          # Display "(RFoot) Moving" under the player's right foot using the image coordinates of the left foot with an offset  
            else: 
                #print(str(Pright_image) + " " + str(prev_right_ankle))
                cv2.putText(cropped_frame, f"(RFoot) Static", (right_ankle[0] +10, right_ankle[1] -20 ), font, font_scale, color, thickness, cv2.LINE_AA)          # Display "(RFoot) Static" under the player's right foot using the image coordinates of the right foot with an offset
    
    prev_left_ankle[0] = Pleft_image[0]
    prev_left_ankle[1] = Pleft_image[1]                                                                                                                  # Update the values of the field coordinates of the feet from the previous frame  with the current ones
                                                                                                                      # Update the values of the field coordinates of the feet from the previous frame  with the current ones
    prev_right_ankle[0] = Pright_image[0]
    prev_right_ankle[1] = Pright_image[1]
    
    #computing the center position of the player in the real field
    center_real = tuple((int((Pright_image[0] + Pleft_image[0])/2), int((Pright_image[1] + Pleft_image[1])/2)))  
    
    #collecting static positions
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

        
    # Setting comparison points
    NtopLeftP = None
    NtopRightP = None
    NbottomLeftP = None
    NbottomRightP = None
    
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
            

            if(NtopLeftP == None or np.linalg.norm(np.array(NtopLeftP) - np.array(topLeftP)) < threshold) : 
                NtopLeftP = topLeftP
            if(NtopRightP == None or np.linalg.norm(np.array(NtopRightP) - np.array(topRightP)) < threshold) : 
                NtopRightP = topRightP
            if(NbottomLeftP == None or np.linalg.norm(np.array(NbottomLeftP) - np.array(bottomLeftP)) < threshold) : 
                NbottomLeftP = bottomLeftP
            if(NbottomRightP == None or np.linalg.norm(np.array(NbottomRightP) - np.array(bottomRightP)) < threshold) : 
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

            points = [NtopLeftP, NtopRightP, NbottomRightP, NbottomLeftP]
            # Calculate homography
            homography_matrix = calculate_homography(np.array(points), points, field_length, field_width)
            return homography_matrix

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

############################### MAIN ####################################
field_length = 23.78 #meters
field_width = 10.97 #meters

#we need a scale factor since the sizes are in meters and if scale_factor = 1 the returned image will be really small
scale_factor = 20
field_length *=scale_factor
field_width *=scale_factor

# Load an image
image_path = 'resources/frame.JPG'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #set the color from BGR to RGB

# Ball trajectory util
positions_stack = [] #stack to compute values in thread
ball_positions = [] #array of the trajectory in the image
ball_positions_real = [] #array of the top-view trajectory in the image
prevpos = (0,0)
lastvalidpos = (0,0)
pos_counter = 0
sequence = False
beginning = True

############## Task 2 ###################
# Font characteristics for the coordinates display 
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
color =(255,255,255)
thickness = 1
############## Task 2 ###################

############## Task 3 ###################
# Initialization of the variables that will retain the previous position of the feet + threshold for the detection of movement
prev_PleftA_image = [0,0]
prev_PrightA_image =[0,0]
prev_PleftB_image = [0,0]
prev_PrightB_image =[0,0]
threshold_moving = 5
############## Task 3 ###################

# Loading of the clip to analyze
video_path = "resources/tennis2.mp4"
total_frames = get_total_frames(video_path)
cap = cv2.VideoCapture(video_path)


# Calculate homography
homography_matrix = autoComputeHomography(cap,None, None, None, None, None)

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
stationary_points_A = list()
stationary_points_B = list()

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #set the color from BGR to RGB

#changing the rectified image to "clean" it from the previous drawings of the center
rectified_image = cv2.warpPerspective(image, homography_matrix, (image.shape[1], image.shape[0]))

# Allocation to write the resulting evaluation in a video file at the end
# Maybe width has to be changed : TODO
result = cv2.VideoWriter('raw.mp4',
                         cv2.VideoWriter_fourcc(*'mp4v'),
                         60, (image.shape[1] + rectified_image.shape[1], 720))

ball_detector = BallDetector('TRACE/TrackNet/Weights.pth', out_channels=2)

print("\nBall positions detected:")
i=0

#parameters for comparison in court detection
NtopLeftP = None
NtopRightP = None
NbottomLeftP = None
NbottomRightP = None
while cv2.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    if not hasFrame:
        # cv2.waitKey()
        break
    cTime = time.time()

    #no item returned since it is just to show live court detection (too noisy to make live computation of homography)
    autoComputeHomography(cap, frame, NtopLeftP, NtopRightP, NbottomLeftP, NbottomRightP)
    #changing the rectified image to "clean" it from the previous drawings of the center
    rectified_image = cv2.warpPerspective(image, homography_matrix, (image.shape[1], image.shape[0]))

    cropped_frame_B = frame[90:250, 390:855].copy()
    cropped_frame_A = frame[350:720, 100:1100].copy()
    
    #creating two threads to improve performances for the detection of the pose
    th_A = threading.Thread(target=computePoseAndAnkles, args=(cropped_frame_A, stationary_points_A, mpPose_A, pose_A, mpDraw_A, homography_matrix, prev_PrightA_image, prev_PleftA_image, threshold_moving, 100, 400, rectified_image))
    th_B = threading.Thread(target=computePoseAndAnkles, args=(cropped_frame_B, stationary_points_B, mpPose_B, pose_B, mpDraw_B, homography_matrix, prev_PrightB_image, prev_PleftB_image, threshold_moving, 390,  90, rectified_image))
    th_C = threading.Thread(target=processBallTrajectory, args=(ball_detector, frame, positions_stack))
    
    th_A.start()
    th_B.start()
    th_C.start()
    th_A.join()
    th_B.join()
    th_C.join()

    ballpos = positions_stack.pop()

    pTime = time.time()

    fps = 1/(cTime-pTime)
    
    frame[350:720, 100:1100] = cropped_frame_A
    frame[90:250, 390:855] = cropped_frame_B
    if ballpos != (0,0):
        cv2.circle(frame, ballpos, 5, (0, 255, 0), cv2.FILLED)
    ball_positions.append(ballpos)
    percent = i/total_frames*100
    print(f"FRAME {i}: {ballpos}; - {percent:.1f}%")
    i += 1

    # Putting ball position into perspective
    #real_ball_pos = cv2.perspectiveTransform(ballpos, homography_matrix)
    #ball_positions_real.append(real_ball_pos)

    cv2.putText(frame, str(int(fps)), (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0), 3)

    # Appending the perspective image on the side
    height = max(frame.shape[0], rectified_image.shape[0])

    frame = cv2.resize(frame, (int(frame.shape[1] * height / frame.shape[0]), height))
    rectified_image = cv2.resize(rectified_image, (int(rectified_image.shape[1] * height / rectified_image.shape[0]), height))
    rectified_image = cv2.cvtColor(rectified_image, cv2.COLOR_RGB2BGR)

    combined_image = cv2.hconcat([frame, rectified_image])
    cv2.imshow('Combined Images', combined_image)
    result.write(combined_image)

cap.release()
result.release()
cv2.destroyAllWindows()

#print("DETECTED POSITIONS")
#for l in ball_positions:
#    print(l)

if i < total_frames:
    print("Execution stopped by user")
    sys.exit()

interpolated_samples = interpolate_missing_values(ball_positions)
print("\nInterpolation:\n")
for r in interpolated_samples:
    print(f"FRAME: {r}")
    print(f"--> {ball_positions[r]}")

cap = cv2.VideoCapture("raw.mp4")

result = cv2.VideoWriter('processed.mp4',
                         cv2.VideoWriter_fourcc(*'mp4v'),
                         60, (image.shape[1] + rectified_image.shape[1], 720))

print("\nInterpolation Completed. Drawing...\n")

j = 0
while cv2.waitKey(1) < 0:
    percent = j/i*100
    print(f"{percent:.1f}%")
    hasFrame, frame = cap.read()
    if not hasFrame:
        break
    
    if j in interpolated_samples:
        cv2.circle(frame, ball_positions[j], 7, (0, 0, 255), cv2.FILLED)   
        cv2.imshow(f'Frame Interpolated: {j}', frame)
    
    result.write(frame) 
    j +=1

cap.release()
result.release()
cv2.destroyAllWindows()
print("The video was successfully processed")