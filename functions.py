import cv2
from imutils.perspective import four_point_transform
import numpy as np
import pytesseract
import matplotlib as plt
import imageio
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import norm
from statsmodels.formula.api import ols

def fdisplay(im_path):
    dpi = 80
    im_data = imageio.imread(im_path)

    height, width = im_data.shape[:2]

    # What size does the figure need to be in inches to fit the image?
    figsize = width/2.5 / float(dpi), height / float(dpi)/1.2

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(im_data, cmap='gray')

    plt.show()

def display(im_data):
    dpi = 80
    height, width = im_data.shape[:2]

    # What size does the figure need to be in inches to fit the image?
    figsize = width/2.5 / float(dpi), height / float(dpi)/1.2

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(im_data, cmap='gray')

    plt.show()

def contour_image(image_name, backg=10):
    approxa = []
    border_thickness = 1
    image0 = cv2.imread(image_name)  # Read the image

    height, width, channels = image0.shape
    # Calculate the number of pixels
    num_pixels = height * width

    if num_pixels < 1000000:
        factor = 3
        new_size = (image0.shape[1] * factor, image0.shape[0] * factor)
        image = cv2.resize(image0, new_size, interpolation=cv2.INTER_CUBIC)
    else:
        image = image0
    image = cv2.copyMakeBorder(image, border_thickness, border_thickness, border_thickness, border_thickness,
                                cv2.BORDER_CONSTANT, value=(0, 0, 0))

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width, channels = image.shape
    gnum_pixels = height * width

    adaptive_thresholded_image = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                       cv2.THRESH_BINARY, 21, 0)
    adaptive_thresholded_image0 = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                       cv2.THRESH_BINARY, 21, backg)

    contours, _ = cv2.findContours(adaptive_thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt1 = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    image_with_contours = cv2.cvtColor(adaptive_thresholded_image, cv2.COLOR_GRAY2BGR)
    perimeter = cv2.arcLength(cnt1, True)
    epsilon = 0.02 * perimeter
    approxa = cv2.approxPolyDP(cnt1, epsilon, True)

    if cv2.contourArea(cnt1) >= gnum_pixels * 0.96:
        adaptive_thresholded_image = adaptive_thresholded_image[:adaptive_thresholded_image.shape[0]-5, :]
        contours, _ = cv2.findContours(adaptive_thresholded_image, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cnt1 = sorted(contours, key=cv2.contourArea, reverse=True)[0]
        image_with_contours = cv2.cvtColor(adaptive_thresholded_image, cv2.COLOR_GRAY2BGR)

    cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)
    cv2.drawContours(image_with_contours, [cnt1], -1, (0, 0, 255), 2)

    if cv2.contourArea(cnt1) >= gnum_pixels / 5:
        perimeter = cv2.arcLength(cnt1, True)
        epsilon = 0.02 * perimeter
        approxx = cv2.approxPolyDP(cnt1, epsilon, True)
        contoured = four_point_transform(adaptive_thresholded_image0, approxx.reshape(4, 2) * 1)
    else:
        contoured = adaptive_thresholded_image0
        approx = approxa

    f_contourred = "f_contourred.jpg"
    cv2.imwrite(f_contourred, contoured)  # Save the new image

    return contoured, f_contourred, approxx

def sharpen(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def fcontour(image_path, subtraction=15):
    # Load the image
    image = cv2.imread(image_path)
    height, width, channels = image.shape

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = sharpen(gray)
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    black_pixel_counts = []

    thresholded_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51,
                                              subtraction)
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

    return contour_image, filled_contour_image, imageo, filtered_contours





def cluster_regression(data_file_path, kluster=12, pvari=1000):
    # Read data from the CSV file
    data = pd.read_csv(data_file_path)
    kluster = kluster
    y = data["y"]
    x = data[["intercept", "x", "x2"]]

    sdd = np.full(kluster, 1000)
    pii = np.full(kluster, 1)
    pii0 = pii * 9
    n_n = data.shape[0]
    inc = np.empty(n_n)

    pvar = np.diag([pvari, pvari, pvari])

    cbeta = np.percentile(y, np.linspace(0, 100, kluster))
    pBeta = np.vstack((cbeta, np.linspace(0, 1e-6, kluster), np.linspace(0, 1e-10, kluster)))
    Beta = pBeta
    T = np.zeros((n_n, kluster))

    for i in range(30):
        for j in range(kluster):
            T[:, j] = pii[j] * norm.pdf(y - np.dot(x[['intercept', 'x', 'x2']], Beta[:, j]), 0, sdd[j])

        row_sums = T.sum(axis=1)
        T = T / row_sums[:, np.newaxis]
        pii = np.sum(T, axis=0) / n_n
        if len(pii) == len(pii0):
            if np.sum(pii - pii0) == 0 and i > 3:
                break

        pii0 = pii
        kluster0 = kluster
        ww = np.empty(n_n)
        max_column_indices = np.argmax(T, axis=1)
        ww[:] = max_column_indices
        tbww = pd.Series(ww).value_counts()
        print(tbww)
        grps = tbww.index.astype(int)
        kluster = len(grps)
        pii = pii[grps]
        Beta = Beta[:, grps]
        T = T[:, grps]
        print(i)

        for j in range(kluster - 1, -1, -1):
            for w in range(n_n):
                inc[w] = T[w, j] == np.max(T[w, :])

            inc_array = np.array(inc)  # Convert inc to a NumPy array
            x_inc = x[inc == 1]
            x_inc = x_inc[['intercept', 'x', 'x2']]
            y_inc = y[inc == 1]
            x_inc_T = x_inc.T

            Beta[:, j] = np.dot(np.linalg.inv(np.linalg.inv(pvar) + np.dot(x_inc_T, x_inc)), (np.dot(np.linalg.inv(pvar),
                                                                                                   pBeta[:, j]) + np.dot(x_inc_T, y_inc)))
            pBeta = Beta
            sdd[j] = np.apply_along_axis(np.std, 0, y_inc - np.dot(x_inc, Beta[:, j]))

            if i > 5:
                xx = pd.DataFrame(x)
                xx['y'] = y
                xx['ww'] = pd.Categorical(ww)
                # Create the linear regression model
                lm = ols('y ~ x + x2 + C(ww)', data=xx).fit()
                sm = lm.summary()
                coef = lm.params

                Beta[0, 0] = coef['Intercept']
                Beta[1, :] = coef['x']
                Beta[2, :] = coef['x2']
                Beta[0, 1:] = [coef[0] + m for m in coef[1:(len(coef)-2)]]

        mean_sdd = np.mean(sdd)
        # Repeat the mean value 'kluster' times to create the new 'sdd' array
        sdd = np.full(kluster, mean_sdd)

    return Beta, ww

