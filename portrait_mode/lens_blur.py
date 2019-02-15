"""
Modified file of the original from Mike Pound of Computerphile
Source code: https://github.com/mikepound/convolve
Source video: https://www.youtube.com/watch?v=vNG3ZAd8wCc
"""
from functools import reduce
import cv2
import numpy as np
from .complex_kernels import get_parameters, complex_kernel_1d, normalise_kernels, weighted_sum
from scipy import signal


def gamma_exposure(image, gamma):
    np.power(image, gamma, out=image)


def gamma_exposure_inverse(image, gamma):
    np.clip(image, 0, None, out=image)
    np.power(image, 1.0 / gamma, out=image)


def pre_process_image(image):
    arr = np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32)
    arr /= 255
    return arr


def post_process_image(image):
    image = image.transpose(1, 2, 0) * 255
    return image.astype('uint8')


def lens_blur(image, radius=32, components=2, exposure_gamma=3.0):
    im_arr = pre_process_image(image)
    img_channels, img_h, img_w = im_arr.shape

    # Obtain component parameters / scale values
    component_params, scale = get_parameters(component_count=components)

    # Create each component for size radius, using scale and other component parameters
    components = []
    for params in component_params:
        c = complex_kernel_1d(radius, scale, params)
        components.append(c)

    # Normalise all kernels together (the combination of all applied kernels in 2D must sum to 1)
    normalise_kernels(components, component_params)

    # Increase exposure to highlight bright spots
    gamma_exposure(im_arr, exposure_gamma)

    # Process RGB channels for all components
    component_output = []
    for component, params in zip(components, component_params):
        channels = []
        for channel in range(img_channels):
            inter = signal.convolve2d(im_arr[channel], component, boundary='symm', mode='same')
            channels.append(signal.convolve2d(inter, component.transpose(), boundary='symm', mode='same'))

        # The final component output is a stack of RGB, with weighted sums of real and imaginary parts
        component_image = np.stack([weighted_sum(channel, params) for channel in channels])
        component_output.append(component_image)

    # Add all components together
    output_image = reduce(np.add, component_output)

    # Reverse exposure
    gamma_exposure_inverse(output_image, exposure_gamma)

    # Avoid out of range values - generally this only occurs with small negatives
    # due to imperfect complex kernels
    np.clip(output_image, 0, 1, out=output_image)

    output = post_process_image(output_image)

    return output


if __name__ == "__main__":
    import imutils

    TEST_IMAGE_PATH = 'jasper.jpg'
    img = cv2.imread(TEST_IMAGE_PATH)
    img = imutils.resize(img, width=750)

    blurred_img = lens_blur(img, radius=4)

    cv2.imshow('Computerphile Blur', blurred_img)
    cv2.waitKey(0)
