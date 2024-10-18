import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np

# Define the path to the model
model_path = "efficientdet_lite0.tflite"

# Set up the options for the Object Detector
BaseOptions = mp.tasks.BaseOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Function to convert DMS to decimal degrees
def dms_to_dd(degrees, minutes, seconds):
    return degrees + (minutes / 60.0) + (seconds / 3600.0)

runway_corners_geo = [
    (dms_to_dd(10, 48, 50), dms_to_dd(106, 38, 35)),
    (dms_to_dd(10, 48, 49), dms_to_dd(106, 38, 36)),
    (dms_to_dd(10, 49, 24), dms_to_dd(106, 40, 10)),
    (dms_to_dd(10, 49, 26), dms_to_dd(106, 40, 10))
]

image_corners_geo = [
    (dms_to_dd(10, 49, 3), dms_to_dd(106, 39, 11)),
    (dms_to_dd(10, 49, 3), dms_to_dd(106, 39, 11)),
    (dms_to_dd(10, 49, 2), dms_to_dd(106, 39, 11)),
    (dms_to_dd(10, 49, 4), dms_to_dd(106, 39, 17))
]

def geo_to_pixel(geo_point, geo_corners, image_shape):
    lat, lon = geo_point
    lat_min, lon_min = min(gc[0] for gc in geo_corners), min(gc[1] for gc in geo_corners)
    lat_max, lon_max = max(gc[0] for gc in geo_corners), max(gc[1] for gc in geo_corners)

    height, width = image_shape[:2]

    x = int((lon - lon_min) / (lon_max - lon_min) * width)
    y = int((lat_max - lat) / (lat_max - lat_min) * height)

    return x, y

def pixel_to_geo(pixel_point, geo_corners, image_shape):
    x, y = pixel_point
    height, width = image_shape[:2]
    lat_min, lon_min = min(gc[0] for gc in geo_corners), min(gc[1] for gc in geo_corners)
    lat_max, lon_max = max(gc[0] for gc in geo_corners), max(gc[1] for gc in geo_corners)

    lon = lon_min + (x / width) * (lon_max - lon_min)
    lat = lat_max - (y / height) * (lat_max - lat_min)

    return lat, lon

def image_geo_to_runway_geo(image_geo, image_corners, runway_corners):
    lat, lon = image_geo
    
    # Calculate relative position within the image
    lat_min_img, lon_min_img = min(ic[0] for ic in image_corners), min(ic[1] for ic in image_corners)
    lat_max_img, lon_max_img = max(ic[0] for ic in image_corners), max(ic[1] for ic in image_corners)
    
    lat_rel = (lat - lat_min_img) / (lat_max_img - lat_min_img)
    lon_rel = (lon - lon_min_img) / (lon_max_img - lon_min_img)
    
    # Map to runway coordinates
    lat_min_rwy, lon_min_rwy = min(rc[0] for rc in runway_corners), min(rc[1] for rc in runway_corners)
    lat_max_rwy, lon_max_rwy = max(rc[0] for rc in runway_corners), max(rc[1] for rc in runway_corners)
    
    lat_rwy = lat_min_rwy + lat_rel * (lat_max_rwy - lat_min_rwy)
    lon_rwy = lon_min_rwy + lon_rel * (lon_max_rwy - lon_min_rwy)
    
    return lat_rwy, lon_rwy

# Initialize object detector with options
options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    max_results=4,
    running_mode=VisionRunningMode.IMAGE
)

with ObjectDetector.create_from_options(options) as detector:
    # Load the image using OpenCV
    image_path = 'ex5.jpg'
    cv_image = cv2.imread(image_path)
    
    # Convert the OpenCV image (BGR) to the MediaPipe image (RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

    # Perform object detection on the image
    result = detector.detect(mp_image)

    # Print the number of detected objects
    print(f"Number of detected objects: {len(result.detections)}")

    # Sort detections from left to right
    sorted_detections = sorted(result.detections, key=lambda d: d.bounding_box.origin_x)

    # Draw circles for each detected object and print coordinates on the image
    for i, detection in enumerate(sorted_detections, 1):
        bbox = detection.bounding_box

        # Calculate center and radius of the circle
        center_x = int(bbox.origin_x + bbox.width / 2)
        center_y = int(bbox.origin_y + bbox.height / 2)
        radius = int(max(bbox.width, bbox.height) / 2)

        # Convert pixel coordinates to geographical coordinates within the image
        geo_lat, geo_lon = pixel_to_geo((center_x, center_y), image_corners_geo, cv_image.shape)

        # Draw a circle covering the detected object
        cv2.circle(cv_image, (center_x, center_y), radius, (0, 255, 0), 2)

        # Convert decimal degrees to DMS for display
        def dd_to_dms(dd):
            d = int(dd)
            m = int((dd - d) * 60)
            s = (dd - d - m/60) * 3600
            return f"{d}°{m}'{s:.2f}\""

        lat_dms = dd_to_dms(geo_lat)
        lon_dms = dd_to_dms(geo_lon)

        # Write the coordinates on the image
        coord_text = f"Object {i}: {lat_dms}N, {lon_dms}E"
        cv2.putText(cv_image, coord_text, (center_x - radius, center_y - radius - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Print the coordinates
        print(f"Object {i} Location: Latitude {lat_dms}N, Longitude {lon_dms}E")

    # Save and display the output image with circles for detected objects
    output_path = 'output_with_object_location.png'
    cv2.imwrite(output_path, cv_image)

    # Show the result using OpenCV
    cv2.imshow('Detected Objects with Location', cv_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
