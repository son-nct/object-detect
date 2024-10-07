import cv2

# Load the image
image_path = 'test.png'
image = cv2.imread(image_path)

# Function to capture the mouse click events
def select_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Selected point: ({x}, {y})")
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Image", image)

# Display the image and set the callback function for mouse clicks
cv2.imshow("Image", image)
cv2.setMouseCallback("Image", select_points)

# Wait until a key is pressed, then close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
