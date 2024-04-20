import cv2
import numpy as np
import matplotlib.pyplot as plt
import threading
#from mediapipe import solutions
#from cv2 import cvtColor, COLOR_BGR2RGB, COLOR_RGB2BGR
import mediapipe as mp
import time

def computePoseAndAnkles(cropped_frame, static_centers_queue,  mpPose, pose, mpDraw, hom_matrix, prev_right_ankle, prev_left_ankle, threshold, x_offset, y_offset, rect_img):

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

def select_points(image):
    # Display the image and allow the user to select points
    plt.imshow(image)
    plt.title("Select points by clicking, press any key when done.")
    points = plt.ginput(n=-1, timeout=0, show_clicks=True) #GINPUT Allows selecting the points through the mouse 
    plt.close()
    return np.array(points) #x and y points 

def draw_lines(image, points):
    # Create a copy of the image to draw lines on
    image_with_lines = np.copy(image)

    # Convert the points to integer coordinates
    points = points.astype(int)

    # Draw lines between consecutive points
    for i in range(len(points) - 1):
        cv2.line(image_with_lines, tuple(points[i]), tuple(points[i+1]), (0, 255, 0), 2)

    # Draw a line between the last and first points to close the loop
    cv2.line(image_with_lines, tuple(points[-1]), tuple(points[0]), (0, 255, 0), 2)

    return image_with_lines

def create_lines(points):
    # Convert the points to integer coordinates
    points = points.astype(int) #convert to integer

    lines = []

    # Create lines between consecutive points
    for i in range(len(points) - 1):
        line = (tuple(points[i]), tuple(points[i+1]))
        lines.append(line)

    # Create a line between the last and first points to close the loop
    line = (tuple(points[-1]), tuple(points[0])) #from 2 points get a line
    lines.append(line)

    return lines


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

# Select points
points = select_points(image)
# Create lines based on the selected points
lines = create_lines(points)
# Draw lines based on the selected points
image_with_lines = draw_lines(image, points)

# Calculate homography
homography_matrix = calculate_homography(points, points, field_length, field_width)
homography_matrix_inv = np.linalg.inv(homography_matrix)

""" testing purpose
# Rectify the original image using the calculated homography
rectified_image = cv2.warpPerspective(image, homography_matrix, (image.shape[1], image.shape[0]))
# Display the rectified image to check the correctness of the homography matrix 
plt.imshow(rectified_image)
# Define a point in the original image (x, y)
point = np.array([[390, 200]], dtype=np.float32)
# Reshape the point array to shape (1, 1, 2)
point = np.reshape(point, (1, 1, 2))
point =cv2.perspectiveTransform(point, homography_matrix)
plt.plot(point[0][0][0], point[0][0][1], 'ro')
plt.title("Rectified Image")
plt.show()
# Display the image with selected points and lines
plt.imshow(image_with_lines)
plt.title("Image with selected points and lines")
plt.show()
"""


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
cap = cv2.VideoCapture("resources/tennis2.mp4")

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
result = cv2.VideoWriter('result2.mp4',
                         cv2.VideoWriter_fourcc(*'mp4v'),
                         60, (image.shape[1] + rectified_image.shape[1], 720))

while cv2.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    if not hasFrame:
        # cv2.waitKey()
        break
    cTime = time.time()
    
    #changing the rectified image to "clean" it from the previous drawings of the center
    rectified_image = cv2.warpPerspective(image, homography_matrix, (image.shape[1], image.shape[0]))

    cropped_frame_B = frame[90:250, 390:855].copy()
    cropped_frame_A = frame[400:720, 100:1100].copy()
    #creating two threads to improve performances for the detection of the pose
    th_A = threading.Thread(target=computePoseAndAnkles, args=( cropped_frame_A, stationary_points_A, mpPose_A, pose_A, mpDraw_A, homography_matrix, prev_PrightA_image, prev_PleftA_image, threshold_moving, 100, 400, rectified_image))
    th_B = threading.Thread(target=computePoseAndAnkles, args=( cropped_frame_B, stationary_points_B, mpPose_B, pose_B, mpDraw_B, homography_matrix, prev_PrightB_image, prev_PleftB_image, threshold_moving, 390,  90, rectified_image))
    
    th_A.start()
    th_B.start()
    th_A.join()
    th_B.join()
    pTime = time.time()

    fps = 1/(cTime-pTime)
    frame[400:720, 100:1100] = cropped_frame_A
    frame[90:250, 390:855] = cropped_frame_B
    cv2.putText(frame, str(int(fps)), (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0), 3)
    #combining the two images 
    height = max(frame.shape[0], rectified_image.shape[0])

    frame = cv2.resize(frame, (int(frame.shape[1] * height / frame.shape[0]), height))
    rectified_image = cv2.resize(rectified_image, (int(rectified_image.shape[1] * height / rectified_image.shape[0]), height))
    rectified_image = cv2.cvtColor(rectified_image, cv2.COLOR_RGB2BGR)

    combined_image = cv2.hconcat([frame, rectified_image])
    cv2.imshow('Combined Images', combined_image)
    result.write(combined_image)


cap.release()
result.release()
# Closes all the frames
cv2.destroyAllWindows()

print("The video was successfully saved")