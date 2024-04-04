import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

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


import argparse

# Section considered if the script main.py is run with arguments in the Terminal, EXAMPLE: $user: main.py --input videotitle.mp4

parser = argparse.ArgumentParser()
parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera')
parser.add_argument('--thr', default=0.1, type=float, help='Threshold value for pose parts heat map')
parser.add_argument('--width', default=1720, type=int, help='Resize input to specific width.')
parser.add_argument('--height', default=550, type=int, help='Resize input to specific height.')

args = parser.parse_args()

# Parts and pairs evaluated by the Deep Learning Human pose model
BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
              "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
              "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
              "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
              ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
              ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
              ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
              ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]

# Input dimensions (A is Sinner's evaluation frame (Bottom), B is Medvedev's one (Top))
# inWidth = args.width
# inHeight = args.height
inWidth_A = 1720
inHeight_A = 550
inWidth_B = 1200
inHeight_B = 365


############## Task 2 ###################
# Font characteristics for the coordinates display 
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
color =(255,255,255)
thickness = 1
############## Task 2 ###################


# Loading of the Deep Learning Model to estimate the pose
net = cv2.dnn.readNetFromTensorflow("resources/pose_model.pb")

# Loading of the clip to analyze
cap = cv2.VideoCapture("resources/tennisMatchShort.mp4")

# Allocation to write the resulting evaluation in a video file at the end
result = cv2.VideoWriter('result1.mp4',
                         cv2.VideoWriter_fourcc(*'mp4v'),
                         10, (1920, 1080))


############## Task 3 ###################
# Initialization of the variables that will retain the previous position of the feet + threshold for the detection of movement
prev_PleftA_image = (0,0)
prev_PrightA_image = (0,0)
prev_PleftB_image = (0,0)
prev_PrightB_image = (0,0)
threshold_moving = 20
############## Task 3 ###################
############## Task 4 ###################
# Initialization of the lists that will collect the time sequence of the step points: i.e., the positions H^-1P of the feet in the instances when they were static.
PleftA_image_static_list = []
PrightA_image_static_list =[]
PleftB_image_static_list = []
PrightB_image_static_list =[]
############## Task 4 ###################

##
# PROCESSING LOOP
while cv2.waitKey(1) < 0:

    hasFrame, frame = cap.read()
    if not hasFrame:
        # cv2.waitKey()
        break

    ################################### BOTTOM DETECTION ##################################################
    # Refer to "TOP DETECTION" under this section to see the comments on how the parts are detected
    cropped_frame_A = frame[530:1080, 20:1740]
    frameWidth = cropped_frame_A.shape[1]
    frameHeight = cropped_frame_A.shape[0]
    # frame[30:350, 250:1700] = (0, 255, 0)

    
    net.setInput(
        cv2.dnn.blobFromImage(cropped_frame_A, 1.0, (inWidth_A, inHeight_A), (127.5, 127.5, 127.5), swapRB=True,
                              crop=False))
    out = net.forward()
    out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

    assert (len(BODY_PARTS) == out.shape[1])


    #Points extraction in region A in array format if it matches the desired points corresponding to the heatmaps
    
    pointsA = []     # changed from points => pointsA to access the data for region A separately
    
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponding body's part.
        heatMap = out[0, i, :, :]

        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this way.
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        # Add a point if it's confidence is higher than threshold.
        pointsA.append((int(x), int(y)) if conf > args.thr else None)

   # Joints pair identification and display
    for pair in POSE_PAIRS:

        partFrom = pair[0]
        partTo = pair[1]
        assert (partFrom in BODY_PARTS)
        assert (partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if pointsA[idFrom] and pointsA[idTo]:
            cv2.line(cropped_frame_A, pointsA[idFrom], pointsA[idTo], (0, 255, 0), 3)
            cv2.ellipse(cropped_frame_A, pointsA[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(cropped_frame_A, pointsA[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
    
 ######################################################################################################
    

    ############## Task 2 + Task 3 + Task 4 ###################
    #Initialization of the left and right ankles coordinates of the player in region A 
    
    PleftA = None
    PrightA = None
    
    # Left ankle coordinates display in region A
    for idx in [13]:                                                                                                                             # I've left the for loop because I thought we might also use the knee position if the ankle is not available (but I've abandonded that, for now. Also the knees might not be detected, but I thought about it as a slight countermeasure)
        if pointsA[idx]:
            
            PleftA = pointsA[idx]                                                                                                                # Current image coordinates of the left ankle
            cv2.circle(cropped_frame_A, PleftA, 5, (255, 0, 0), -1)                                                                              # Blue circle to highlight the left ankle position
            PleftA_image = cv2.perspectiveTransform(np.array([[PleftA]], dtype=np.float32), homography_matrix_inv)[0][0]                         # Computation of the field coordinates of the left ankle using H^(-1)   (I believe the values are not correct, so modifications are needed)
            PleftA_image = (round(PleftA_image[0]),round(PleftA_image[1]))                                                                       # Approximation to avoid displaying all the decimals
            cv2.putText(cropped_frame_A, f"{PleftA_image}", (PleftA[0] + 10, PleftA[1]), font, font_scale, color, thickness, cv2.LINE_AA)        # Display of the left foot field coordinates values at image coordinates, with slight offset on the X axis to avoid overlapping with the actual foot
            #print("Original point (PleftA):", PleftA)
            #print("Transformed point (PleftA_image):", PleftA_image)
         
        if  PleftA is None:                                                                                                                       # Avoid the "TypeError: 'NoneType' object is not subscriptable" => successful 
            PleftA = (0,0)   
    
    # Right ankle coordinates display in region A
    for idx in [10]:                                                                                                                             # I've left the for loop because I thought we might also use the knee position if the ankle is not available (but I've abandonded that, for now. Also the knees might not be detected, but I thought about it as a slight countermeasure)
        if pointsA[idx]:
            
            PrightA = pointsA[idx]                                                                                                               # Current image coordinates of the right ankle
            cv2.circle(cropped_frame_A, PrightA, 5, (255, 0, 0), -1)                                                                             # Blue circle to highlight the right ankle position
            PrightA_image = cv2.perspectiveTransform(np.array([[PrightA]], dtype=np.float32), homography_matrix_inv)[0][0]                       # Computation of the field coordinates of the right ankle using H^(-1)   (I believe the values are not correct, so modifications are needed)
            PrightA_image = (round(PrightA_image[0]),round(PrightA_image[1]))                                                                    # Approximation to avoid displaying all the decimals
            cv2.putText(cropped_frame_A, f"{PrightA_image}", (PrightA[0] + 10, PrightA[1]), font, font_scale, color, thickness, cv2.LINE_AA)     # Display of the right foot field coordinates values at image coordinates, with slight offset on the X axis to avoid overlapping with the actual foot
            
        if  PrightA is None:                                                                                                                     # Avoid the "TypeError: 'NoneType' object is not subscriptable" => successful
            PrightA = (0,0)
 
        # Check if the player's position in region A has changed compared to the previous frame
    if prev_PleftA_image is not None and prev_PrightA_image is not None:
                
        left_foot_moved_A = np.linalg.norm(np.array(PleftA_image) - np.array(prev_PleftA_image)) > threshold_moving                              # Euclidean distance computation between the current Left foot position and its position in the previous frame, all compared to the chosen threshold 
        right_foot_moved_A = np.linalg.norm(np.array(PrightA_image) - np.array(prev_PrightA_image)) > threshold_moving                           # Euclidean distance computation between the current Right foot position and its position in the previous frame, all compared to the chosen threshold                

        if PleftA !=(0,0):                                                                                                                                # Check if the left ankle's point has been detected 
            if left_foot_moved_A:                                                                                                                         # Check if the left foot has moved 
                cv2.putText(cropped_frame_A, f"(LFoot) Moving", (PleftA[0]-60, PleftA[1]+70), font, font_scale, color, thickness, cv2.LINE_AA)            # Display "(LFoot) Moving" under the player's left foot using the image coordinates of the left foot with an offset  
        
            else:
                cv2.putText(cropped_frame_A, f"(LFoot) Static", (PleftA[0]-60, PleftA[1]+70), font, font_scale, color, thickness, cv2.LINE_AA)            # Display "(LFoot) Static" under the player's left foot using the image coordinates of the left foot with an offset  
                PleftA_image_static_list.append(PleftA_image)                                                                                             # Add the image coordinates of the left foot to the list if it is static (Task4)
        if PrightA!=(0,0):                                                                                                                                # Check if the right ankle's point has been detected
            if  right_foot_moved_A:                                                                                                                       # Check if the right foot has moved            
                cv2.putText(cropped_frame_A, f"(RFoot) Moving", (PrightA[0]+60, PrightA[1]+40), font, font_scale, color, thickness, cv2.LINE_AA)          # Display "(RFoot) Moving" under the player's right foot using the image coordinates of the left foot with an offset  
  
            else:                         
                cv2.putText(cropped_frame_A, f"(RFoot) Static", (PrightA[0]+60, PrightA[1]+40), font, font_scale, color, thickness, cv2.LINE_AA)          # Display "(RFoot) Static" under the player's right foot using the image coordinates of the right foot with an offset
                PrightA_image_static_list.append(PrightA_image)                                                                                           # Add the image coordinates of the right foot to the list if it is static (Task4)
        prev_PleftA_image = PleftA_image                                                                                                                  # Update the values of the field coordinates of the feet from the previous frame  with the current ones
        prev_PrightA_image = PrightA_image
   
    else:
   
        prev_PleftA_image = prev_PleftA_image                                                                                                             # Keep the values of the field coordinates of the feet from the previous frame constant in case they weren't detected
        prev_PrightA_image = prev_PrightA_image
        

          
    ############## Task 2 + Task 3 + Task 4 ###################
            
    ####################################################################################################
    ################################### TOP DETECTION ##################################################
    
    cropped_frame_B = frame[20:385, 400:1700]
    frameWidth = cropped_frame_B.shape[1]
    frameHeight = cropped_frame_B.shape[0]
    # frame[20:385, 400:1700] = (255, 0, 0)

    # Set frame to analyze with the model
    net.setInput(cv2.dnn.blobFromImage(cropped_frame_B, 1.0, (inWidth_B, inHeight_B), (127.5, 127.5, 200), swapRB=True,
                                       crop=False)) 
    out = net.forward()  # Evaluation by the net
    out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements (R,G,B)  OUT is a set of heatmaps

    assert (len(BODY_PARTS) == out.shape[1])

    #Points extraction in region B in array format if it matches the desired points corresponding to the heatmaps

    pointsB = []   # changed from points => pointsB to access the data for region B separately
    
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponding body's part.
        heatMap = out[0, i, :, :]

        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this way.
        _, conf, _, point = cv2.minMaxLoc(heatMap)  # The detection looks at the maximum probability distribution for the considered body part
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        # Add a point if it's confidence is higher than threshold.
        pointsB.append((int(x),int(y)) if conf > 0.25 else None) 
        # IF CONFIDENCE INTERVAL IS ABOVE 0.2, THE SPECIFIC POINT IS CONSIDERED AS DETECTED (0.2 was imposed by me as the best-looking so far)
    
    # Joints pair identification and display
    for pair in POSE_PAIRS:
        
        partFrom = pair[0]
        partTo = pair[1]
        assert (partFrom in BODY_PARTS)
        assert (partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if pointsB[idFrom] and pointsB[idTo]:

            cv2.line(cropped_frame_B, pointsB[idFrom], pointsB[idTo], (0, 255, 0), 3)
            cv2.ellipse(cropped_frame_B, pointsB[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(cropped_frame_B, pointsB[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
######################################################################################################


    ############## Task 2 + Task 3 + Task 4 ###################
    #Initialization of the left and right ankles coordinates of the player in region B 
    
    PleftB = None
    PrightB = None

    #Left ankle coordinates display in region B
    for idx in [13]:                                                                                                                               # I've left the for loop because I thought we might also use the knee position if the ankle is not available (but I've abandonded that, for now. Also the knees might not be detected, but I thought about it as a slight countermeasure)
        if pointsB[idx]:

            PleftB = pointsB[idx]                                                                                                                  # Current image coordinates of the left ankle
            cv2.circle(cropped_frame_B, PleftB, 5, (255, 0, 0), -1)                                                                                # Blue circle to highlight the left ankle position
            PleftB_image = cv2.perspectiveTransform(np.array([[PleftB]], dtype=np.float32), homography_matrix_inv)[0][0]                           # Computation of the field coordinates of the left ankle using H^(-1)   (I believe the values are not correct, so modifications are needed)
            PleftB_image = (round(PleftB_image[0]),round(PleftB_image[1]))                                                                         # Approximation to avoid displaying all the decimals
            cv2.putText(cropped_frame_B, f"{PleftB_image}", (PleftB[0] + 10, PleftB[1]), font, font_scale, color, thickness, cv2.LINE_AA)          # Display of the left foot field coordinates values at image coordinates, with slight offset on the X axis to avoid overlapping with the actual foot
        
        if PleftB is None:
            PleftB = (0,0)                                                                                                                         # Avoid the "TypeError: 'NoneType' object is not subscriptable" => successful
    
    #Right ankle coordinates display in region B
    for idx in [10]:
        if pointsB[idx]:

            PrightB = pointsB[idx]                                                                                                               # Current image coordinates of the right ankle
            cv2.circle(cropped_frame_B, PrightB, 5, (255, 0, 0), -1)                                                                             # Blue circle to highlight the right ankle position
            PrightB_image = cv2.perspectiveTransform(np.array([[PrightB]], dtype=np.float32), homography_matrix_inv)[0][0]                       # Computation of the field coordinates of the right ankle using H^(-1)   (I believe the values are not correct, so modifications are needed)
            PrightB_image = (round(PrightB_image[0]),round(PrightB_image[1]))                                                                    # Approximation to avoid displaying all the decimals
            cv2.putText(cropped_frame_B, f"{PrightB_image}", (PrightB[0] + 10, PrightB[1]), font, font_scale, color, thickness, cv2.LINE_AA)     # Display of the right foot field coordinates values at image coordinates, with slight offset on the X axis to avoid overlapping with the actual foot
            
        if PrightB is None:
            PrightB = (0,0)                                                                                                                      # Avoid the "TypeError: 'NoneType' object is not subscriptable" => successful
        
        # Check if the player's position in region A has changed compared to the previous frame
    if prev_PleftB_image is not None and prev_PrightB_image is not None:
                
        left_foot_moved_B = np.linalg.norm(np.array(PleftB_image) - np.array(prev_PleftB_image)) > threshold_moving                              # Euclidean distance computation between the current Left foot position and its position in the previous frame, all compared to the chosen threshold 
        right_foot_moved_B = np.linalg.norm(np.array(PrightB_image) - np.array(prev_PrightB_image)) > threshold_moving                           # Euclidean distance computation between the current Right foot position and its position in the previous frame, all compared to the chosen threshold

        if PleftB !=(0,0):                                                                                                                       # Check if the left ankle's point has been detected
            if left_foot_moved_B:                                                                                                                # Check if the left foot has moved
                cv2.putText(cropped_frame_B, f"(Lfoot) Moving", (PleftB[0]-60, PleftB[1]+70), font, font_scale, color, thickness, cv2.LINE_AA)   # Display "(LFoot) Moving" under the player's left foot using the image coordinates of the left foot with an offset  
                            
            else:
                cv2.putText(cropped_frame_B, f"(Lfoot) Static", (PleftB[0]-60, PleftB[1]+70), font, font_scale, color, thickness, cv2.LINE_AA)   # Display "(LFoot) Static" under the player's left foot using the image coordinates of the left foot with an offset  
                PleftB_image_static_list.append(PleftB_image)                                                                                    # Add the image coordinates of the left foot to the list if it is static (Task4)
        if PrightB !=(0,0):                                                                                                                      # Check if the right ankle's point has been detected
            if right_foot_moved_B:                                                                                                               # Check if the right foot has moved 
                cv2.putText(cropped_frame_B, f"(Rfoot) Moving", (PrightB[0]-60, PrightB[1]+40), font, font_scale, color, thickness, cv2.LINE_AA) # Display "(RFoot) Moving" under the player's right foot using the image coordinates of the left foot with an offset  
                            
            else:
                cv2.putText(cropped_frame_B, f"(Rfoot) Static", (PrightB[0]-60, PrightB[1]+40), font, font_scale, color, thickness, cv2.LINE_AA) # Display "(RFoot) Static" under the player's right foot using the image coordinates of the right foot with an offset
                PrightB_image_static_list.append(PrightB_image)                                                                                  # Add the image coordinates of the right foot to the list if it is static (Task4)
                
        prev_PleftB_image = PleftB_image                                                                                                         # Update the values of the field coordinates of the feet from the previous frame  with the current ones
        prev_PrightB_image = PrightB_image
        
    else:
            
        prev_PleftB_image = prev_PleftB_image                                                                                                    # Keep the values of the field coordinates of the feet from the previous frame constant in case they weren't detected
        prev_PrightB_image = prev_PrightB_image
              
    ############## Task 2 + Task 3 + Task 4 ###################


    t, _ = net.getPerfProfile()
    freq = cv2.getTickFrequency() / 1000

    # Overlapping of processed sections to the main video frame
    frame[530:1080, 20:1740] = cropped_frame_A
    frame[20:385, 400:1700] = cropped_frame_B

    cv2.putText(frame, '%.2fms' % (t / freq), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    cv2.imshow('Tennis Human Pose estimation through OpenCV', frame)
    result.write(frame)
cap.release()
result.release()

# Closes all the frames
cv2.destroyAllWindows()

print("The video was successfully saved")
