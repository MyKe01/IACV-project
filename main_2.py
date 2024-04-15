import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
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
#from mediapipe import solutions
#from cv2 import cvtColor, COLOR_BGR2RGB, COLOR_RGB2BGR
import mediapipe as mp
import time

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

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
color =(255,255,255)
thickness = 1

# Loading of the Deep Learning Model to estimate the pose
net = cv2.dnn.readNetFromTensorflow("resources/pose_model.pb")

# Loading of the clip to analyze
cap = cv2.VideoCapture("resources/tennis2.mp4")

# Allocation to write the resulting evaluation in a video file at the end
result = cv2.VideoWriter('result2.mp4',
                         cv2.VideoWriter_fourcc(*'mp4v'),
                         10, (1280, 720))



"""
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

    pointsA = []
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
    PleftA = None
    PrightA = None
    
    for idx in [13]:
        if pointsA[idx]:
            PleftA = pointsA[idx]
            cv2.circle(cropped_frame_A, PleftA, 5, (255, 0, 0), -1)
            cv2.putText(cropped_frame_A, f"{PleftA}", (PleftA[0] + 10, PleftA[1]), font, font_scale, color, thickness, cv2.LINE_AA)
    
    for idx in [10]:
        if pointsA[idx]:
            PrightA = pointsA[idx]
            cv2.circle(cropped_frame_A, PrightA, 5, (255, 0, 0), -1)
            cv2.putText(cropped_frame_A, f"{PrightA}", (PrightA[0] + 10, PrightA[1]), font, font_scale, color, thickness, cv2.LINE_AA)


    ####################################################################################################
    ################################### TOP DETECTION ##################################################
    
    cropped_frame_B = frame[20:385, 400:1700]
    frameWidth = cropped_frame_B.shape[1]
    frameHeight = cropped_frame_B.shape[0]
    # frame[450:1030, 200:1800] = (255, 0, 0)

    # Set frame to analyze with the model
    net.setInput(cv2.dnn.blobFromImage(cropped_frame_B, 1.0, (inWidth_B, inHeight_B), (127.5, 127.5, 200), swapRB=True,
                                       crop=False)) 
    out = net.forward()  # Evaluation by the net
    out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements (R,G,B)  OUT is a set of heatmaps

    assert (len(BODY_PARTS) == out.shape[1])

    pointsB = []   #get the array of final detected points only if
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponding body's part.
        heatMap = out[0, i, :, :]

        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this way.
        _, conf, _, point = cv2.minMaxLoc(
            heatMap)  # The detection looks at the maximum probability distribution for the considered body part
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        # Add a point if it's confidence is higher than threshold.
        pointsB.append((int(x),int(y)) if conf > 0.25 else None) 
        # IF CONFIDENCE INTERVAL IS ABOVE 0.2, THE SPECIFIC POINT IS CONSIDERED AS DETECTED (0.2 was imposed by me as the best-looking so far)

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

    PleftB = None
    PrightB = None
    
    for idx in [13]:
        if pointsB[idx]:
            PleftB = pointsB[idx]
            cv2.circle(cropped_frame_B, PleftB, 5, (255, 0, 0), -1)
            cv2.putText(cropped_frame_B, f"{PleftB}", (PleftB[0] + 10, PleftB[1]), font, font_scale, color, thickness, cv2.LINE_AA)
    
    for idx in [10]:
        if pointsB[idx]:
            PrightB = pointsB[idx]
            cv2.circle(cropped_frame_B, PrightB, 5, (255, 0, 0), -1)
            cv2.putText(cropped_frame_B, f"{PrightB}", (PrightB[0] + 10, PrightB[1]), font, font_scale, color, thickness, cv2.LINE_AA)


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
"""

"""

class crop1:
    x: float = 50/100
    xoffset: float = 0/100
    xcenter: int = 1 
    
    y: float = 33/100
    yoffset: float = 0/100
    ycenter: int = 0
    
class crop2:
    x: float = 83/100
    xoffset: float = 0/100
    xcenter: int = 1 
    
    y: float = 60/100
    yoffset: float = 40/100
    ycenter: int = 0

mp_pose = solutions.pose
pose1 = mp_pose.Pose(model_complexity=2, min_detection_confidence=0.25, min_tracking_confidence=0.25)
pose2 = mp_pose.Pose(model_complexity=2, min_detection_confidence=0.25, min_tracking_confidence=0.25)
"""

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

# PROCESSING LOOP
while cv2.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    if not hasFrame:
        # cv2.waitKey()
        break
    cTime = time.time()
    #cropped_frame_A = frame[530:1080, 20:1740]
    cropped_frame_A = frame[400:720, 100:1100]
    frameWidth = cropped_frame_A.shape[1]
    frameHeight = cropped_frame_A.shape[0]
    imgRGB = cv2.cvtColor(cropped_frame_A, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(cropped_frame_A, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w,c = cropped_frame_A.shape
            print(id, lm)
            cx, cy = int(lm.x*w), int(lm.y*h)
            cv2.circle(cropped_frame_A, (cx, cy), 5, (255,0,0), cv2.FILLED)

    
    
    pTime = time.time()

    fps = 1/(cTime-pTime)
    frame[400:720, 100:1100] = cropped_frame_A
    cv2.putText(frame, str(int(fps)), (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0), 3)
    cv2.imshow("Image", frame) 
    result.write(frame)



cap.release()
result.release()

# Closes all the frames
cv2.destroyAllWindows()

print("The video was successfully saved")

"""
#DETECTION FUNCTION
def bodyMap(frame, pose1, pose2, crop1, crop2):
        
    # Mapping of Player 1
    frame1 = frame[crop1.yoffset:crop1.y+crop1.yoffset,crop1.xoffset:crop1.x+crop1.xoffset]
    frame1 = cvtColor(frame1, COLOR_BGR2RGB)
    results1 = pose1.process(frame1)
    frame1 = cvtColor(frame1, COLOR_RGB2BGR)
    # mp_drawing.draw_landmarks(frame1, results1.pose_landmarks,solutions.pose.POSE_CONNECTIONS)
    if results1.pose_landmarks is not None:
        l1_foot_x = int(results1.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x * crop1.x) + crop1.xoffset
        l1_foot_y = int(results1.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y * crop1.y) + crop1.yoffset

        r1_foot_x = int(results1.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x * crop1.x) + crop1.xoffset
        r1_foot_y = int(results1.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y * crop1.y) + crop1.yoffset

        l1_hand_x = int(results1.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].x * crop1.x) + crop1.xoffset
        l1_hand_y = int(results1.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].y * crop1.y) + crop1.yoffset

        r1_hand_x = int(results1.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].x * crop1.x) + crop1.xoffset
        r1_hand_y = int(results1.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].y * crop1.y) + crop1.yoffset

        nose1_x = int(results1.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * crop1.x) + crop1.xoffset
        nose1_y = int(results1.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * crop1.y) + crop1.yoffset
    else:
        l1_foot_x = None
        l1_foot_y = None

        r1_foot_x = None
        r1_foot_y = None

        l1_hand_x = None
        l1_hand_y = None

        r1_hand_x = None
        r1_hand_y = None

        nose1_x = None
        nose1_y = None

    # Mapping of Player 2
    frame2 = frame[crop2.yoffset:crop2.y+crop2.yoffset,crop2.xoffset:crop2.x+crop2.xoffset]
    frame2 = cvtColor(frame2, COLOR_BGR2RGB)
    results2 = pose2.process(frame2)
    frame2 = cvtColor(frame2, COLOR_RGB2BGR)
    # mp_drawing.draw_landmarks(frame2, results2.pose_landmarks,solutions.pose.POSE_CONNECTIONS)

    if results2.pose_landmarks is not None:
        l2_foot_x = int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x * crop2.x) + crop2.xoffset
        l2_foot_y = int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y * crop2.y) + crop2.yoffset

        r2_foot_x = int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x * crop2.x) + crop2.xoffset
        r2_foot_y = int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y * crop2.y) + crop2.yoffset

        l2_hand_x = int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].x * crop2.x) + crop2.xoffset
        l2_hand_y = int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].y * crop2.y) + crop2.yoffset

        r2_hand_x = int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].x * crop2.x) + crop2.xoffset
        r2_hand_y = int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].y * crop2.y) + crop2.yoffset

        nose2_x = int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * crop2.x) + crop2.xoffset
        nose2_y = int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * crop2.y) + crop2.yoffset
    else:
        l2_foot_x = None
        l2_foot_y = None

        r2_foot_x = None
        r2_foot_y = None

        l2_hand_x = None
        l2_hand_y = None

        r2_hand_x = None
        r2_hand_y = None

        nose2_x = None
        nose2_y = None

    return ([[[l1_foot_x,l1_foot_y],[r1_foot_x,r1_foot_y],[l2_foot_x,l2_foot_y],[r2_foot_x,r2_foot_y]], [[l1_hand_x,l1_hand_y],[r1_hand_x,r1_hand_y],[l2_hand_x,l2_hand_y],[r2_hand_x,r2_hand_y]], [[nose1_x, nose1_y], [nose2_x, nose2_y]]])
"""