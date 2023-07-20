import cv2
import numpy as np
import matplotlib.pyplot as plt
from scan.functions import sharpen

# Load the image
image = cv2.imread('m5.jpg')
height, width, channels = image.shape
# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
gray = sharpen(gray)
image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
black_pixel_counts = []
thresholded_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51,
                                             15)
_, thresholded_imageo = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# thresholded_image = cv2.bitwise_or(thresholded_image, thresholded_imageo)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))  # Adjust the kernel size as needed
# Perform the opening operation
thresholded_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_OPEN, kernel)

imageo = cv2.cvtColor(thresholded_image, cv2.COLOR_GRAY2RGB)
# Perform morphological dilation
kernel_size = (2, 1)  # (Width, Height)
# Create the custom kernel
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
# Perform one-sided dilation with the recalculated weighted center
dilated = 255 - imageo.copy()  # Create a copy of the original image
for _ in range(20):  # Number of iterations for dilation
    # Calculate the weighted center of the current kernel
    kernel_sum = np.sum(kernel)
    weighted_center = (int(np.sum(np.multiply(kernel, np.arange(kernel_size[0]))) / kernel_sum),
                           int(np.sum(np.multiply(kernel, np.arange(kernel_size[1]))) / kernel_sum))

    # Perform dilation with the recalculated anchor point
    dilated = cv2.dilate(dilated, kernel, anchor=weighted_center, iterations=1)
    # Invert the image
inverted = cv2.bitwise_not(dilated)
inverted_gray = cv2.cvtColor(inverted, cv2.COLOR_BGR2GRAY)

# Apply thresholding to create a binary image
# _, binary = cv2.threshold(inverted_gray, 127, 255, cv2.THRESH_BINARY)
_, binary = cv2.threshold(inverted_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# Find contours of the text regions
contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# contours, _ = cv2.findContours(inverted, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = list(reversed(contours))

# Calculate the average of rotated angles of all contours
total_rotated_angle = 0.0
num_contours = 0

# Filter contours based on width, height, and aspect ratio
filtered_contours = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = h / float(w)
    if w > image.shape[1] / 1150 and w < width and h >= 30 and aspect_ratio <= 5 and h < height * 0.9:
       filtered_contours.append(cnt)
       # Calculate the rotated angle of the current contour
       rect = cv2.minAreaRect(cnt)
       rotated_angle = rect[2]  # Extract the angle from the rectangle
       total_rotated_angle += rotated_angle
       num_contours += 1

# Draw the filtered contours on the image
contour_image = imageo.copy()
cv2.drawContours(contour_image, filtered_contours, -1, (0, 0, 255), 2)

# Create a blank image to hold the filled contours
filled_contour_image = np.zeros_like(contour_image)
# Fill the contours with yellow color
for contour in filtered_contours:
    cv2.drawContours(filled_contour_image, [contour], 0, (0, 255, 255), thickness=cv2.FILLED)


import cv2
import numpy as np
import matplotlib.pyplot as plt

import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_mean_pixel_vertical_position_in_contour(contour, image):
    x, y, w, h = cv2.boundingRect(contour)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], 0, 255, thickness=cv2.FILLED)

    # Apply the mask to the image and extract the vertical positions of the pixels within the contour
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    column_positions = []
    for col in range(x, x + w):
        column_pixels = np.argwhere(masked_image[:, col, 0] > 0)[:, 0]
        if column_pixels.any():
            mean_vertical_pos = np.mean(column_pixels)
        else:
            # Set a default value or handle empty column_pixels list as needed
            mean_vertical_pos = y + h // 2  # Center of the contour
        column_positions.append(mean_vertical_pos)

    return column_positions

# Rest of your code...

# Draw the filtered contours on the image
contour_image = imageo.copy()
cv2.drawContours(contour_image, filtered_contours, -1, (0, 0, 255), 2)  # Displaying all filtered contours

# Create a blank image to hold the filled contours
filled_contour_image = np.zeros_like(contour_image)

# Lists to store all column positions from all contours
all_column_positions = []

# Fill the contours with yellow color and plot curves for all filtered contours
for contour in filtered_contours:
    cv2.drawContours(filled_contour_image, [contour], 0, (0, 255, 255), thickness=cv2.FILLED)

    # Get the mean vertical position for each column within the contour
    column_positions = calculate_mean_pixel_vertical_position_in_contour(contour, image)

    # Find the minimum x-coordinate of the contour
    min_x = np.min(contour[:, 0, 0])

    # Create x-axis values relative to the contour's position
    x_values = np.arange(len(column_positions)) + min_x

    # Plot the curve for the current contour on top of the filled contour image
    plt.plot(x_values, column_positions, color='red', linewidth=1)

    # Append the column_positions to the all_column_positions list
    all_column_positions.extend(column_positions)

# Show the figure with both the filled contour image and the overlaid curves for all contours
plt.imshow(filled_contour_image)
plt.xlabel('Column Position (Relative to Contour)')
plt.ylabel('Mean Vertical Position')
plt.title('Curves Overlay on Contours')

# Set high resolution for the saved image (300 dpi, for example)
plt.savefig("out.jpg", dpi=300)

# Show the plot
plt.show()

# Plot a horizontal histogram of all column positions
plt.figure()
plt.hist(all_column_positions, bins=300, orientation='horizontal', color='blue')
plt.gca().invert_yaxis()
plt.xlabel('Frequency')
plt.ylabel('Mean Vertical Position')
plt.title('Horizontal Histogram of Mean Vertical Positions')
plt.savefig("histogram.jpg", dpi=300)
plt.show()

import cv2
import numpy as np


def find_contours_in_same_row(contours):
    row_contours = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        centroid_y = y + h // 2

        # Check if there are any existing rows to group the contour
        row_found = False
        for row_contour in row_contours:
            # Find the y-coordinate of the row's centroid
            row_x, row_y, row_w, row_h = cv2.boundingRect(row_contour[0])
            row_centroid_y = row_y + row_h // 2

            # Define a tolerance for grouping contours in the same row
            tolerance = 100

            # If the contour's centroid is close to the row's centroid, add it to the row
            if abs(centroid_y - row_centroid_y) < tolerance:
                row_contour.append(contour)
                row_found = True
                break

        # If no row found, create a new row with the current contour
        if not row_found:
            row_contours.append([contour])

    return row_contours


# Example usage:
row_contours = find_contours_in_same_row(filtered_contours)

# Draw the contours of each row with different colors
output_image = imageo.copy()
for i, row_contour in enumerate(row_contours):
    color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
    for contour in row_contour:
        cv2.drawContours(output_image, [contour], -1, color, 2)

cv2.imshow("Contours Grouped by Rows", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("Contours Grouped by Rows.jpg", output_image)
