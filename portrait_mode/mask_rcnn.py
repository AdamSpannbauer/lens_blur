import cv2
from gluoncv import model_zoo, data, utils
from .mask_rcnn_utils import filter_results


def read_and_detect(image_path, threshold=0.5):
    net = model_zoo.get_model('mask_rcnn_resnet50_v1b_coco', pretrained=True)
    x, orig_img = data.transforms.presets.rcnn.load_test(image_path)
    bgr_orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)

    results = [xx[0].asnumpy() for xx in net(x)]

    ids, scores, bounding_boxes, masks = filter_results(results, threshold)

    h, w = bgr_orig_img.shape[:2]
    expanded_masks = utils.viz.expand_mask(masks, bounding_boxes, (w, h), scores)

    results_dict = {
        'ids': ids,
        'scores': scores,
        'bounding_boxes': bounding_boxes,
        'masks': expanded_masks
    }

    return results_dict, bgr_orig_img
