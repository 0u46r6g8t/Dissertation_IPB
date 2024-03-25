import cv2, numpy as np
import matplotlib.pyplot as plt

image = cv2.imread(
    "/home/quieregatog/Documents/Documentos/Master-d_gree/Master_Dissertation/study/dataset/oli-bor-def/oli-bor-def_00002.png",
    cv2.IMREAD_COLOR,
)

channels = [0]
histSize = [256]
ranges = [0, 255]

hist = cv2.calcHist([image], channels, None, histSize, ranges)


# Ravel transform image to vector (one dimensional)
def __plot_hist_grayscale(_image=None):
    plt.hist(_image.ravel(), 256, [0, 256])
    plt.show()


def __plot_hist_image(_image=None):
    color = ("b", "g", "r")

    image = cv2.cvtColor(_image, cv2.IMREAD_COLOR)

    for i, col in enumerate(color):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=col)

        plt.xlim([0, 256])
    plt.show()


def __equalize_image(_image=None):
    image = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)

    eq = cv2.equalizeHist(image)

    res = np.hstack((image, eq))

    cv2.imshow("Image normal", res)

    histOriginal = cv2.calcHist([image], [0], None, [256], [0, 256])
    histEqualize = cv2.calcHist([eq], [0], None, [256], [0, 256])

    plt.subplot(121), plt.plot(histOriginal)
    plt.subplot(122), plt.plot(histEqualize)

    plt.show()


def __equalize_image_bgr(_image=None):

    image = cv2.cvtColor(_image, cv2.COLOR_BGR2YUV)

    image[:, :, 0] = cv2.equalizeHist(image[:, :, -1])

    image_out = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)

    cv2.imshow("Color input image", _image)
    cv2.imshow("Histogram", image_out)


def __calculate_hist_image(_image=None):
    # Create a mask
    mask = np.zeros(_image.shape[:2], np.uint8)
    mask[10:250, 80:190] = 255

    # Calculate histogram
    hist_full = cv2.calcHist([_image], [0], None, [256], [0, 256])
    hist_mask = cv2.calcHist([_image], [0], mask, [256], [0, 256])

    # Plot histogram
    plt.subplot(131), plt.imshow(image, "gray")
    plt.subplot(132), plt.imshow(mask, "gray")
    plt.subplot(133), plt.plot(hist_full), plt.plot(hist_mask)

    plt.xlim([0, 256])
    plt.show()


for i, val in zip(range(0, 256), hist):
    if (val) > 0:
        print(f"hist[{i}]={val}")


# __plot_hist_grayscale(image)
# __plot_hist_image(image)
__calculate_hist_image(image)
# __equalize_image(image)
# __equalize_image_bgr(image)

cv2.waitKey(0)
cv2.destroyAllWindows()
     