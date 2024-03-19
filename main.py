import cv2
import numpy as np
import matplotlib.pyplot as plt
"""
def select_points(image):
    # Display the image and allow the user to select points
    plt.imshow(image)
    plt.title("Select points by clicking, press any key when done.")
    points = plt.ginput(n=-1, timeout=0, show_clicks=True)
    plt.close()
    return np.array(points)

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
    points = points.astype(int)
    
    lines = []
    
    # Create lines between consecutive points
    for i in range(len(points) - 1):
        line = (tuple(points[i]), tuple(points[i+1]))
        lines.append(line)
    
    # Create a line between the last and first points to close the loop
    line = (tuple(points[-1]), tuple(points[0]))
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
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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
# Display the rectified image to check the correcteness of the homography matrix 
plt.imshow(rectified_image)
plt.title("Rectified Image")
plt.show()
# Display the image with selected points and lines
plt.imshow(image_with_lines)
plt.title("Image with selected points and lines")
plt.show()
"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera')
parser.add_argument('--thr', default=0.1, type=float, help='Threshold value for pose parts heat map')
parser.add_argument('--width', default=600, type=int, help='Resize input to specific width.')
parser.add_argument('--height', default=600, type=int, help='Resize input to specific height.')

args = parser.parse_args()


BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

inWidth = args.width
inHeight = args.height

net = cv2.dnn.readNetFromTensorflow("resources/pose_model.pb")

cap = cv2.VideoCapture("resources/tennisMatchShort.mp4")

result = cv2.VideoWriter('result.mp4',  
                         cv2.VideoWriter_fourcc(*'mp4v'), 
                         10, (1280, 720))

while cv2.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv2.waitKey()
        break

    cropped_frame = frame[530:1080, 20:1740]
    frameWidth = cropped_frame.shape[1]
    frameHeight = cropped_frame.shape[0]
    
    net.setInput(cv2.dnn.blobFromImage(cropped_frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

    assert(len(BODY_PARTS) == out.shape[1])

    points = []
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponging body's part.
        heatMap = out[0, i, :, :]

        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this way.
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        # Add a point if it's confidence is higher than threshold.
        points.append((int(x), int(y)) if conf > args.thr else None)

    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv2.line(cropped_frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv2.ellipse(cropped_frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(cropped_frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

    t, _ = net.getPerfProfile()
    freq = cv2.getTickFrequency() / 1000
    frame[530:1080, 20:1740] = cropped_frame
    cv2.putText(frame, '%.2fms' % (t / freq), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    cv2.imshow('Tennis Human Pose estimation through OpenCV', frame)
    result.write(frame)
cap.release() 
result.release() 
    
# Closes all the frames 
cv2.destroyAllWindows() 
   
print("The video was successfully saved") 
""""""
20-530
1740-1080
""""""