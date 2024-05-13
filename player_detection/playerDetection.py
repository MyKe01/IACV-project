import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red

class PlayersDetections : 

    def __init__(self) -> None:
        base_options = python.BaseOptions(model_asset_buffer=open('models\efficientdet_lite0.tflite', "rb").read())
        ObjectDetector = mp.tasks.vision.ObjectDetector
        ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        options = ObjectDetectorOptions(
            base_options=base_options,
            max_results=5,
            running_mode=VisionRunningMode.VIDEO,
            category_allowlist = ["person"])
        
        self.detector = ObjectDetector.create_from_options(options) 

    def getDetector(self) :
        return self.detector
    
    def filterDetections(self, mid_frame, detection_results) -> np.ndarray :
        detections = []
        for detection in detection_results.detections:
            # Draw bounding_box
            bbox = detection.bounding_box
            start_point = bbox.origin_x, bbox.origin_y
            end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height

            mid_point = (start_point[0] + end_point[0] ) /2
            distance = np.linalg.norm(mid_point - mid_frame) 
            detections.append((detection,distance))

        # Sort detections based on distance
        detections.sort(key=lambda x: x[1])

        # Get the first two detections
        top_detections = [detection[0] for detection in detections[:2]]

        return top_detections

    def visualize(self, image, detection_result) -> np.ndarray:
      
        """Draws bounding boxes on the input image and return it.
        Args:
        image: The input RGB image.
        detection_result: The list of all "Detection" entities to be visualized.
        Returns:
           Image with bounding boxes.
        """
        for detection in detection_result:
            # Draw bounding_box
            bbox = detection.bounding_box
            start_point = bbox.origin_x, bbox.origin_y
            end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)
    
            # Draw label and score
            category = detection.categories[0]
            category_name = category.category_name
            probability = round(category.score, 2)
            result_text = category_name + ' (' + str(probability) + ')'
            text_location = (MARGIN + bbox.origin_x,
                             MARGIN + ROW_SIZE + bbox.origin_y)
            cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)
        return image