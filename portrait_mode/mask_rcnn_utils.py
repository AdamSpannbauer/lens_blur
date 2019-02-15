def filter_results(results, threshold=0.5):
    ids, scores, bounding_boxes, masks = results
    n = (scores > threshold).sum()

    return ids[:n], scores[:n], bounding_boxes[:n], masks[:n]


def mask_area(mask):
    return (mask != 0).sum()


def largest_mask(masks):
    mask_areas = [mask_area(m) for m in masks]
    largest_mask_index = mask_areas.index(max(mask_areas))
    return masks[largest_mask_index]
