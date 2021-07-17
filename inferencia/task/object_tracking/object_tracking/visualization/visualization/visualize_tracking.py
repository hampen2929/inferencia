import cv2


def visualize_tracking(image,
                       class_name,
                       xmin,
                       ymin,
                       xmax,
                       ymax,
                       confidence,
                       tracking_id,
                       bbox_color=(0, 75, 255),
                       text_color=(0, 0, 0)):
    txt = '{}-{}-{:.2f}'.format(class_name,
                                tracking_id,
                                confidence)
    cat_size = cv2.getTextSize(
        txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

    cv2.rectangle(image,
                  (xmin, ymin - cat_size[1] - 2),
                  (xmin + cat_size[0], ymin - 2), bbox_color, -1)

    cv2.rectangle(image,
                  (xmin, ymin),
                  (xmax, ymax), bbox_color, 2)

    cv2.putText(image,
                txt,
                (xmin, ymin - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, text_color, thickness=1, lineType=cv2.LINE_AA)
    return image
