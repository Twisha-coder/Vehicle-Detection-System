import cv2
import numpy as np
import os

# Directory containing images
image_directory = 'Images'

# Ensure the directory path exists
if not os.path.exists(image_directory):
    print("Directory not found.")
    exit()

# List images in the directory
image_files = os.listdir(image_directory)
print("Number of images:", len(image_files))

# Loop through each image in the directory
for image_file in image_files:
    print("Processing image:", image_file)
    # Load image
    frame1 = cv2.imread(os.path.join(image_directory, image_file))
    if frame1 is None:
        print("Failed to load image:", image_file)
        continue
    print("Image shape:", frame1.shape)

    # Apply Gaussian blur to denoise the image
    denoised_image = cv2.GaussianBlur(frame1, (3, 3), 0)

    # Convert denoised image to grayscale
    gray_image = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to segment the image
    _, segmented_image = cv2.threshold(gray_image, 102, 255, cv2.THRESH_BINARY)

    # Apply Canny edge detection
    edges = cv2.Canny(segmented_image, 100, 200)  # Adjust the thresholds as needed

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through contours and classify shapes
    for contour in contours:
        # Calculate perimeter of contour
        perimeter = cv2.arcLength(contour, True)

        # Approximate contour to a polygon
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)

        # Get the number of vertices
        num_vertices = len(approx)

        # If the shape has 4 vertices (rectangle), classify as a vehicle
        if num_vertices == 10:
            # Get bounding box coordinates
            x, y, w, h = cv2.boundingRect(contour)
            
            # Draw contour and classify as vehicle
            cv2.drawContours(frame1, [contour], -1, (0, 255, 0), 2)
            cv2.putText(frame1, 'Vehicle', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the processed image
    cv2.imshow("Processed Image", frame1)

    # Wait for user input (press any key) to show the next image
    if cv2.waitKey(0) == 27:
        break

cv2.destroyAllWindows()
