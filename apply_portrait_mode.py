"""Slow as h*ck homegrown 'portrait mode' implementation

* Runs pre-trained COCO Mask RCNN on image (reference: [gluoncv](https://gluon-cv.mxnet.io/api/model_zoo.html#gluoncv.model_zoo.MaskRCNN)).
* Stores largest detected object's mask.
* Apply lens blur everywhere but on detected object's mask (reference: [Computerphile](https://github.com/mikepound/convolve))
"""
import cv2
from portrait_mode import portrait_mode

IMAGE_PATH = 'images/jasper_deck.jpg'

# Run detection and blurring effects
portrait_mode_img, original_img = portrait_mode(IMAGE_PATH)

# Display results
cv2.imshow("Poor Man's Portrait Mode", portrait_mode_img)
cv2.imshow('Original', original_img)
cv2.waitKey(0)
