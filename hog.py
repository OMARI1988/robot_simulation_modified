import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, color, exposure
from skimage.io import imread,imsave
from skimage.color import rgb2gray
from skimage.transform import rescale
from skimage.util import random_noise
#image = color.rgb2gray(data.astronaut())
image_m = imread('/home/omari/Dropbox/Reports/AAAI16/Muhannad_with_modifications/pics/to_be_hog_4.png')
image = rgb2gray(image_m)
# image = []
# for i in range(10):
#     image_m = random_noise(image_o, mode='s&p',amount=.002)
#     if image == []:
#         image = rgb2gray(image_m)/10
#     else:
#         image += rgb2gray(image_m)/10

fd, hog_image = hog(image, orientations=8, pixels_per_cell=(8, 8),
                    cells_per_block=(1, 1), visualise=True)
fig = plt.figure()

# ax1.axis('off')
# ax1.imshow(image, cmap=plt.cm.gray)
# ax1.set_title('Input image')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
# print hog_image_rescaled

# imsave('/home/omari/Dropbox/Reports/AAAI16/Muhannad_with_modifications/pics/jet_cube.png',hog_image_rescaled)
plt.axis('off')
plt.imshow(hog_image_rescaled, cmap=plt.cm.jet)
# ax1.set_title('Histogram of Oriented Gradients')
plt.tight_layout()
plt.savefig('/home/omari/Dropbox/Reports/AAAI16/Muhannad_with_modifications/pics/jet_sphere.png', dpi=200, facecolor='w', edgecolor='w',
        orientation='portrait', bbox_inches=None)
plt.show()
