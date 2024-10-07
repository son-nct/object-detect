import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2  # OpenCV for drawing bounding boxes
import numpy as np

# Define the path to the model
model_path = "efficientdet_lite0.tflite"

# Set up the options for the Object Detector
BaseOptions = mp.tasks.BaseOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Runway coordinates from Google Earth (latitude, longitude)
runway_corners_geo = [
    (10.815000, 106.636944),  # 07L end
    (10.814722, 106.636944),  # Adjacent to 07L end
    (10.824444, 106.663056),  # 25R end
    (10.824722, 106.663056)   # Adjacent to 25R end
]

# Corresponding pixel coordinates of the runway segment in the image
runway_corners_pixel = [
    (50, 50),    # Pixel coordinate for Corner 1
    (950, 50),   # Pixel coordinate for Corner 2
    (950, 550),  # Pixel coordinate for Corner 3
    (50, 550)    # Pixel coordinate for Corner 4
]

# Function to map geographical coordinates to pixel coordinates (forward transformation)
def geo_to_pixel(geo_point, geo_corners, pixel_corners):
    lat, lon = geo_point
    lat_min, lon_min = geo_corners[0]
    lat_max, lon_max = geo_corners[2]

    x_min, y_min = pixel_corners[0]
    x_max, y_max = pixel_corners[2]

    # Linear interpolation
    x = x_min + (lon - lon_min) / (lon_max - lon_min) * (x_max - x_min)
    y = y_min + (lat - lat_min) / (lat_max - lat_min) * (y_max - y_min)

    return int(x), int(y)

# Reverse mapping function to convert pixel coordinates back to geographical coordinates
def pixel_to_geo(pixel_point, geo_corners, pixel_corners):
    x, y = pixel_point
    lat_min, lon_min = geo_corners[0]
    lat_max, lon_max = geo_corners[2]

    x_min, y_min = pixel_corners[0]
    x_max, y_max = pixel_corners[2]

    # Reverse interpolation for latitude and longitude
    lon = lon_min + (x - x_min) / (x_max - x_min) * (lon_max - lon_min)
    lat = lat_min + (y - y_min) / (y_max - y_min) * (lat_max - lat_min)

    return lat, lon

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
        
        # Calculate the center of the bounding box to draw a circle
        center_x = (start_point[0] + end_point[0]) // 2
        center_y = (start_point[1] + end_point[1]) // 2
        radius = max((end_point[0] - start_point[0]) // 2, (end_point[1] - start_point[1]) // 2)

        # Draw a circle around the detected object
        cv2.circle(cv_image, (center_x, center_y), radius, (0, 0, 255), 2)

        # Write the coordinates on the image (center of the detected object)
        coord_text = f"Center: ({center_x}, {center_y})"
        cv2.putText(cv_image, coord_text, (start_point[0], start_point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Print the coordinates
        print(f"Coordinates: Center ({center_x}, {center_y})")

    # Now we map the FOD coordinates to pixel space
    fod_geo_coords = [
        (10.819167, 106.650833),  # FOD point 1
        (10.819167, 106.650833),  # FOD point 2
        (10.819167, 106.651944),  # FOD point 3
        (10.819444, 106.651667)   # FOD point 4
    ]

    # Map each FOD geographical coordinate to pixel space and circle it
    for fod_geo in fod_geo_coords:
        fod_pixel = geo_to_pixel(fod_geo, runway_corners_geo, runway_corners_pixel)
        cv2.circle(cv_image, fod_pixel, 10, (0, 255, 0), -1)  # Circle the FOD in green
        print(f"FOD detected at pixel coordinates: {fod_pixel}")

        # Reverse the pixel coordinates to geographical coordinates
        fod_geo_mapped = pixel_to_geo(fod_pixel, runway_corners_geo, runway_corners_pixel)
        print(f"FOD geographical coordinates on CHC road: {fod_geo_mapped}")

    # Save and display the output image with circles and coordinates
    output_path = 'output_with_fod.png'
    cv2.imwrite(output_path, cv_image)

    # Show the result using OpenCV
    cv2.imshow('Detected Objects with FOD Coordinates', cv_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
