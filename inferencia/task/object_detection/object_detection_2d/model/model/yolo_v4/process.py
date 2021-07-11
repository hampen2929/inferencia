from typing import Tuple, List, Union

import numpy as np
import cv2


def pre_process(images: list,
                input_width: int,
                input_height: int) -> Tuple[np.ndarray, list]:
    img_in = []
    image_sizes = []
    for image in images:
        image_sizes.append(image.shape)

        resized = cv2.resize(image,
                             (input_width, input_height),
                             interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        img /= 255.0
        img_in.append(img)
    img_in = np.array(img_in)
    return img_in, image_sizes


def validate_bbox(xmin,
                  ymin,
                  xmax,
                  ymax,
                  frame_height,
                  frame_width,
                  square=False
                  ):
    if square:
        raise NotImplementedError
    else:
        xmin = max(int(xmin * frame_width), 0)
        ymin = max(int(ymin * frame_height), 0)
        xmax = min(int(xmax * frame_width), frame_width)
        ymax = min(int(ymax * frame_height), frame_height)
    return xmin, ymin, xmax, ymax


def nms_cpu(boxes, confs, nms_thresh=0.5, min_mode=False):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = confs.argsort()[::-1]

    keep = []
    while order.size > 0:
        idx_self = order[0]
        idx_other = order[1:]

        keep.append(idx_self)

        xx1 = np.maximum(x1[idx_self], x1[idx_other])
        yy1 = np.maximum(y1[idx_self], y1[idx_other])
        xx2 = np.minimum(x2[idx_self], x2[idx_other])
        yy2 = np.minimum(y2[idx_self], y2[idx_other])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        if min_mode:
            over = inter / np.minimum(areas[order[0]], areas[order[1:]])
        else:
            over = inter / (areas[order[0]] + areas[order[1:]] - inter)

        inds = np.where(over <= nms_thresh)[0]
        order = order[inds + 1]

    return np.array(keep)


def post_processing(output, conf_thresh, nms_thresh):
    box_array = output[0]
    confs = output[1]

    # t1 = time.time()

    if type(box_array).__name__ != 'ndarray':
        box_array = box_array.cpu().detach().numpy()
        confs = confs.cpu().detach().numpy()

    num_classes = confs.shape[2]

    # [batch, num, 4]
    box_array = box_array[:, :, 0]

    # [batch, num, num_classes] --> [batch, num]
    max_conf = np.max(confs, axis=2)
    max_id = np.argmax(confs, axis=2)

    # t2 = time.time()

    bboxes_batch = []
    for i in range(box_array.shape[0]):

        argwhere = max_conf[i] > conf_thresh
        l_box_array = box_array[i, argwhere, :]
        l_max_conf = max_conf[i, argwhere]
        l_max_id = max_id[i, argwhere]

        bboxes = []
        # nms for each class
        for j in range(num_classes):

            cls_argwhere = l_max_id == j
            ll_box_array = l_box_array[cls_argwhere, :]
            ll_max_conf = l_max_conf[cls_argwhere]
            ll_max_id = l_max_id[cls_argwhere]

            keep = nms_cpu(ll_box_array, ll_max_conf, nms_thresh)

            if (keep.size > 0):
                ll_box_array = ll_box_array[keep, :]
                ll_max_conf = ll_max_conf[keep]
                ll_max_id = ll_max_id[keep]

                for k in range(ll_box_array.shape[0]):
                    bboxes.append([ll_box_array[k, 0], ll_box_array[k, 1], ll_box_array[k, 2],
                                   ll_box_array[k, 3], ll_max_conf[k], ll_max_conf[k], ll_max_id[k]])

        bboxes_batch.append(bboxes)

    # t3 = time.time()

    # print('-----------------------------------')
    # print('       max and argmax : %f' % (t2 - t1))
    # print('                  nms : %f' % (t3 - t2))
    # print('Post processing total : %f' % (t3 - t1))
    # print('-----------------------------------')

    return bboxes_batch
