import cv2
import numpy as np
from bopflow.models.utils import DOutput


def draw_outputs(img, detections: [DOutput]):
    box_color = (255, 0, 0)
    box_width = 2
    wh = np.flip(img.shape[0:2])
    for result in detections:
        #x1y1 = tuple((result.box.x1y1 * wh).astype(np.int32))
        #x2y2 = tuple((result.box.x2y2 * wh).astype(np.int32))
        # TODO - after fixing tfrecord generation change this to
        # casting back to coordinates instead of the normailized floats
        # atm tfrecord testing is done with does not have normalized coordinates
        x1y1 = (result.box.x1y1).astype(np.int32)
        x2y2 = (result.box.x2y2).astype(np.int32)

        img = cv2.rectangle(
            img=img,
            pt1=tuple(x1y1),
            pt2=tuple(x2y2),
            color=box_color,
            thickness=box_width)
        text = "label: {}| score: {}".format(
            result.label.number,
            result.confidence_score,
        )
        img = cv2.putText(
            img=img,
            text=text,
            org=tuple(x2y2 + 5),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=box_color,
            thickness=1,
        )
    return img