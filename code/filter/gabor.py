import numpy as np
import cv2
import matplotlib.pyplot as plt

# Get image
img_bor = cv2.imread(
    "/home/quieregatog/Documents/Documentos/Master-d_gree/Master_Dissertation/Master-s_Thesis/database/train/oli-pot-def/oli-pot-def_00001.png"
)

ksize = 3
sigma = 3
theta = 1 * np.pi / 4
lambda_ = 1 * np.pi / 4
gamma = 0.4
phi = 0


def _getImageFiltering(_image=None):
    _image = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)

    # Configure the kernel
    kernel = cv2.getGaborKernel(
        (ksize, ksize),
        sigma,
        theta,
        gamma,
        phi,
        ktype=cv2.CV_32F,
    )

    filtering_img = cv2.filter2D(_image, cv2.CV_8UC3, kernel)

    kernel_resize = cv2.resize(kernel, (400, 400))

    return filtering_img, kernel_resize, 


filtering_image, kernel_ = _getImageFiltering(img_bor)

kernel = kernel_

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

h, w = kernel.shape

X, Y = np.meshgrid(np.arange(0, w), np.arange(0, h))

X_f = X.flatten()
Y_f = Y.flatten()
Z_f = kernel.flatten()


# Apply normalization color
norm = plt.Normalize(Z_f.min(), Z_f.max())
colors = plt.cm.viridis(norm(Z_f))

ax.scatter(X_f, Y_f, Z_f, c=colors, cmap='viridis')

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Kernel value")

mappable = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
mappable.set_array(Z_f)
cbar = plt.colorbar(mappable, ax=ax, orientation='vertical', shrink=0.5, aspect=10)
cbar.set_label('Kernel value')


plt.show()
cv2.imshow('Image original', img_bor)
cv2.imshow('Image Filtering', filtering_image)
cv2.imshow('Image kernel', kernel_)

cv2.waitKey(0)
cv2.destroyAllWindows()