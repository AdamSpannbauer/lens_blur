import numpy as np
from .mask_rcnn import read_and_detect
from .mask_rcnn_utils import largest_mask
from .lens_blur import lens_blur


def portrait_mode(image_path, blur_radius=5):
    detection_results, original_img = read_and_detect(image_path, threshold=0.5)
    main_mask = largest_mask(detection_results['masks'])
    object_segment = np.where(main_mask != 0)

    lens_blurred_image = lens_blur(original_img, radius=blur_radius)
    lens_blurred_image[object_segment] = original_img[object_segment]

    return lens_blurred_image, original_img
