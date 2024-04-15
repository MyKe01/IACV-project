import cv2
import numpy as np
import matplotlib.pyplot as plt
import threading
#from mediapipe import solutions
#from cv2 import cvtColor, COLOR_BGR2RGB, COLOR_RGB2BGR
import mediapipe as mp
import time

def computePoseAndAnkles(cropped_frame, ankles_queue,  mpPose, pose, mpDraw, homography_matrix_inv, prev_right_ankle, prev_left_ankle, threshold):

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
                right_ankle= (int(lm.x*w), int(lm.y*h))
                cv2.circle(cropped_frame, right_ankle, 5, (0,0,255), cv2.FILLED)
                Pright_image = cv2.perspectiveTransform(np.array([[right_ankle]], dtype=np.float32), homography_matrix_inv)[0][0]                         # Computation of the field coordinates of the left ankle using H^(-1)   (I believe the values are not correct, so modifications are needed)
                Pright_image = (round(Pright_image[0]),round(Pright_image[1]))                                                                       # Approximation to avoid displaying all the decimals
                cv2.putText(cropped_frame, f"{Pright_image}", (right_ankle[0] + 10, right_ankle [1]), font, font_scale, color, thickness, cv2.LINE_AA)        # Display of the left foot field coordinates values at image coordinates, with slight offset on the X axis to avoid overlapping with the actual foot
            elif id == 27 :
                left_ankle = (int(lm.x*w), int(lm.y*h))
                cv2.circle(cropped_frame, left_ankle, 5, (0,255,0), cv2.FILLED)
                Pleft_image = cv2.perspectiveTransform(np.array([[left_ankle]], dtype=np.float32), homography_matrix_inv)[0][0]                         # Computation of the field coordinates of the left ankle using H^(-1)   (I believe the values are not correct, so modifications are needed)
                Pleft_image = (round(Pleft_image[0]),round(Pleft_image[1]))                                                                       # Approximation to avoid displaying all the decimals
                cv2.putText(cropped_frame, f"{Pleft_image}", (left_ankle[0] + 10, left_ankle [1] + 20), font, font_scale, color, thickness, cv2.LINE_AA)        # Display of the left foot field coordinates values at image coordinates, with slight offset on the X axis to avoid overlapping with the actual foot
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
    
    if right_ankle != (0,0) and left_ankle != (0,0) and left_foot_moved != True and right_foot_moved != True : 
        ankles_queue.append((right_ankle, left_ankle))


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

    field_corners = np.array([[0, field_length], [field_width, field_length], [field_width, 0], [0, 0]], dtype=np.float32)

    # Calculate the homography
    homography, _ = cv2.findHomography(image_points, field_corners)

    return homography


############################### MAIN ####################################
field_length = 23.78 #meters
field_width = 10.97 #meters

#we need a scale factor since the sizes are in meters and if scale_factor = 1 the returned image will be really small
scale_factor = 30
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

"""
# Rectify the original image using the calculated homography
rectified_image = cv2.warpPerspective(image, homography_matrix, (image.shape[1], image.shape[0]))
# Display the rectified image to check the correctness of the homography matrix 
plt.imshow(rectified_image)
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

# Allocation to write the resulting evaluation in a video file at the end
result = cv2.VideoWriter('result2.mp4',
                         cv2.VideoWriter_fourcc(*'mp4v'),
                         60, (1280, 720))

mpPose_A = mp.solutions.pose
pose_A = mpPose_A.Pose()
mpDraw_A = mp.solutions.drawing_utils

mpPose_B = mp.solutions.pose
pose_B = mpPose_B.Pose()
mpDraw_B = mp.solutions.drawing_utils

# PROCESSING LOOP
# each landmark has an id - https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
# ids 28 and 27 are for right and left ankle
ankles_queue_A = list()
ankles_queue_B = list()

while cv2.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    if not hasFrame:
        # cv2.waitKey()
        break
    cTime = time.time()

    cropped_frame_B = frame[90:250, 390:855].copy()
    cropped_frame_A = frame[400:720, 100:1100].copy()
    th_A = threading.Thread(target=computePoseAndAnkles, args=(cropped_frame_A, ankles_queue_A, mpPose_A, pose_A, mpDraw_A, homography_matrix_inv, prev_PrightA_image, prev_PleftA_image, threshold_moving))
    th_B = threading.Thread(target=computePoseAndAnkles, args=(cropped_frame_B, ankles_queue_B,  mpPose_B, pose_B, mpDraw_B, homography_matrix_inv, prev_PrightB_image, prev_PleftB_image, threshold_moving))
    
    th_A.start()
    th_B.start()
    th_A.join()
    th_B.join()
    pTime = time.time()

    fps = 1/(cTime-pTime)
    frame[400:720, 100:1100] = cropped_frame_A
    frame[90:250, 390:855] = cropped_frame_B

    cv2.putText(frame, str(int(fps)), (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0), 3)
    cv2.imshow("Image", frame) 
    result.write(frame)


cap.release()
result.release()

# Closes all the frames
cv2.destroyAllWindows()

print("The video was successfully saved")