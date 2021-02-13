# import necessary packages

from numpy.lib.function_base import gradient
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

import numpy as np
from PIL import Image, ImageEnhance

# Packages for K Means Clustering


# im = Image.open(image_path)

# data = np.array(im)

# image_data = im.load()
# print(image_data)
# height, width = im.size

# print(height, width)
# for loop1 in range(height):
#     for loop2 in range(width):
#         r, g, b = image_data[loop1, loop2]
#         print(r, g, b)

image_path = './music_score.png'
image_path_two = './music_score_two.png'
image_path_three = './tradition.jpg'
image_path_four = './testing.png'

image = cv2.imread(image_path)


# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
# new_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)


pixel_colors = image.reshape(np.shape(image)[0] * np.shape(image)[1], 3)


def get_2d_pixels(pixel_colors):
    for i in range(pixel_colors.shape[0]):
        if any(pixel_colors[i] == [0, 0, 0]):
            print(pixel_colors[i])


def generate_color_hist(image):
    r, g, b = cv2.split(image)
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")

    pixel_colors = image.reshape((np.shape(image)[0]*np.shape(image)[1], 3))
    norm = colors.Normalize(vmin=-1., vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()

    axis.scatter(r.flatten(), g.flatten(), b.flatten(),
                 facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Red")
    axis.set_ylabel("Green")
    axis.set_zlabel("Blue")
    plt.show()


def preprocess_image(image_path) -> Image:

    # PIL -> Enhance -> Contrast Stretch ->  5%
    img1 = Image.open(image_path)
    img2 = ImageEnhance.Contrast(img1).enhance(2.8)
    pil_image = img2.convert('RGBA')
    return pil_image


def mask_with_pil(image):

    print(image.split())

    data = np.array(image)
    red, green, blue, alpha = data.T

    print(data)
    colored_region = (red > 100) & (blue > 100) & (green > 100)

    dark_region = (red < 100) & (blue < 100) & (
        green < 100)

    print(colored_region.shape)
    # or (red <= 100) & (blue <= 100) & (green <= 100)

    data[..., :-1][colored_region.T] = (0, 0, 255)
    # data[..., :-1][colored_region.T] = (0, 0, 255)

    new_image = Image.fromarray(data)
    new_image.show()


# pil_image = preprocess_image(image_path=image_path)
# mask_with_pil(pil_image)

# print(pil_image.T)

hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

black = (0, 0, 0)
white = (255, 255, 255)

# Create a mask of the dominant color (black)
# Do Bitwise operation to replace colors not white or black.

# mask = cv2.inRange(hsv_image, black, white)

# res = cv2.bitwise_and(image, image, mask)

# cv2.imshow('Mask', hsv_image)

# cv2.waitKey(0)


# conversion
def do_conversion(filename):
    # im = Image.open(filename)
    im = preprocess_image(filename)
    R, G, B = im.convert('RGB').split()
    r = R.load()
    g = G.load()
    b = B.load()
    w, h = im.size

    # Convert non-black pixels to white
    for i in range(w):
        for j in range(h):
            if(r[i, j] > 100 or g[i, j] > 100 or b[i, j] > 100):
                r[i, j] = 255  # Just change R channel

    # Merge just the R channel as all channels
    im = Image.merge('RGB', (R, R, R))
    im.save("black_and_white.png")
    im.show()


# conversion
def do_conversion_two(filename):
    im = preprocess_image(filename)
    R, G, B = im.convert('RGB').split()
    r = R.load()
    g = G.load()
    b = B.load()
    w, h = im.size

    # Convert non-black pixels to white
    for i in range(w):
        for j in range(h):
            if(r[i, j] < 10 or g[i, j] < 10 or b[i, j] < 100):
                r[i, j] = 255  # Just change R channel

    # Merge just the R channel as all channels
    im = Image.merge('RGB', (R, R, R))
    im.save("black_and_white.png")
    im.show()


def do_thresholding(image):
    im = cv2.imread(image)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    t, thresh = cv2.threshold(im, 100, 255, cv2.THRESH_BINARY)

    cv2.imshow('Gray', thresh)
    cv2.waitKey(0)


def do_dilation(image):
    img = cv2.imread(image, 0)

    dilated_img = cv2.dilate(img, np.ones((7, 7), np.uint64))
    bg_img = cv2.medianBlur(dilated_img, 21)
    diff_img = 255 - cv2.absdiff(img, bg_img)

    norm_img = diff_img.copy()  # Needed for 3.x compatibility
    cv2.normalize(diff_img, norm_img, alpha=0, beta=255,
                  norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    _, thr_img = cv2.threshold(norm_img, 230, 0, cv2.THRESH_TRUNC)

    cv2.normalize(thr_img, thr_img, alpha=0, beta=255,
                  norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    cv2.imshow('Dilated', thr_img)
    cv2.waitKey(0)


def hsv_masking(image):
    img = cv2.imread(image)
    # img = img*3
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 0, 0])
    upper = np.array([180, 255, 255])
    mask = cv2.inRange(img, lower, upper)

    # dilate mask

    w, h = mask.shape[0], mask.shape[1]
    white_mask = np.zeros((w, h), dtype="uint8")
    cv2.rectangle(white_mask, (0, 0), (h, w), 255, -1)

    # dilating to improve text readability
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    dilate = cv2.dilate(mask, kernel, iterations=1)
    # result = cv2.bitwise_and(image, image, mask=dilate)
    # result[dilate == 0] = [255, 255, 255]

    # new_shape = mask.reshape(mask, (mask.shape(0), mask.shape(1), 3))
    # print(new_shape)
    res = cv2.bitwise_xor(mask, white_mask)

    cv2.imwrite('hsv_masking.jpg', res)
    cv2.imshow('HSV Image', res)

    cv2.waitKey(0)


def pil_process(image_path) -> Image:

    # PIL -> Enhance -> Contrast Stretch ->  5%
    img1 = Image.open(image_path)
    # img2 = ImageEnhance.Contrast(img1).enhance(2.8)
    pil_image = img1.convert('LAB')
    pil_image.show()


# do_conversion(image_path)
hsv_masking(image_path)

# hsv_masking(image_path_two)

"""
def hsv_masking(image):
    img = cv2.imread(image)
    pil_image = pil_to_opencv(image)
    hsv_image = cv2.cvtColor(pil_image, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 0, 0])
    upper = np.array([180, 255, 255])
    masking = cv2.inRange(pil_image, lower, upper)

    cv2.imshow('HSV Image', pil_image)

    cv2.waitKey(0)


def pil_to_opencv(image):
    pil_image = preprocess_image(image)
    open_cv_image = np.array(pil_image)
    # Convert RGB to BGR
    # open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image


hsv_masking(image_path)
"""
