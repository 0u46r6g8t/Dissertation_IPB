import cv2
import numpy as np

# Read an image using OpenCV
image = cv2.imread('/home/quieregatog/Documents/Documentos/Master-d_gree/Master_Dissertation/Master-s_Thesis/database/train/oli-pot-def/oli-pot-def_00001.png', cv2.IMREAD_GRAYSCALE)

# Define a 3D filter/kernel
kernel =    # Example: 3x3x3 averaging filter

# Perform 3D convolution using scipy's convolve function
result = cv2.filter2D(image, -1, kernel)

# Display the original and filtered images
cv2.imshow('Original', image)
cv2.imshow('Filtered', result.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
