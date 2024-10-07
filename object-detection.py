import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2  # OpenCV for drawing bounding boxes

# Define the path to the model
model_path = "efficientdet_lite0.tflite"

# Set up the options for the Object Detector
BaseOptions = mp.tasks.BaseOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Initialize object detector with options
options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    max_results=2,
    running_mode=VisionRunningMode.IMAGE
)

with ObjectDetector.create_from_options(options) as detector:
    # Load the image using OpenCV
    image_path = 'test.png'
    cv_image = cv2.imread(image_path)
    
    # Convert the OpenCV image (BGR) to the MediaPipe image (RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

    # Perform object detection on the image
    result = detector.detect(mp_image)

    # Draw bounding boxes and print coordinates on the image
    for detection in result.detections:
        bbox = detection.bounding_box

        # Bounding box coordinates
        start_point = (int(bbox.origin_x), int(bbox.origin_y))  # Top-left corner
        end_point = (int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height))  # Bottom-right corner
        
        # Draw bounding box on the image
        cv2.rectangle(cv_image, start_point, end_point, (0, 255, 0), 2)

        # Write the coordinates on the image (top-left and bottom-right)
        coord_text = f"TL: {start_point}, BR: {end_point}"
        cv2.putText(cv_image, coord_text, (start_point[0], start_point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Print the coordinates
        print(f"Coordinates: Top-left {start_point}, Bottom-right {end_point}")

    # Save and display the output image with bounding boxes
    output_path = 'output_with_coordinates.png'
    cv2.imwrite(output_path, cv_image)

    # Show the result using OpenCV
    cv2.imshow('Detected Objects with Coordinates', cv_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
