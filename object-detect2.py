import cv2

# Load the image
image_path = 'test.png'
image = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply GaussianBlur to reduce noise and improve edge detection
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Use Canny edge detection to detect edges in the image
edges = cv2.Canny(blurred, 50, 150)

# Find contours in the image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort contours by area and pick the largest contour assuming it's the runway
contours = sorted(contours, key=cv2.contourArea, reverse=True)

# Assuming the largest contour is the runway, we approximate the polygon for it
approx = cv2.approxPolyDP(contours[0], 0.02 * cv2.arcLength(contours[0], True), True)

# Create a copy of the original image to draw on
image_copy = image.copy()

# Ensure the approximation has 4 corners
if len(approx) == 4:
    # Draw the corners on the image
    for point in approx:
        cv2.circle(image_copy, tuple(point[0]), 10, (0, 0, 255), -1)

# Save the result image with detected runway corners
cv2.imwrite('/mnt/data/runway_detected.png', image_copy)

# Extract pixel coordinates of the corners
runway_corners_pixel = [tuple(point[0]) for point in approx]

# Print the result to the console
print("Runway corners detected:", runway_corners_pixel)
