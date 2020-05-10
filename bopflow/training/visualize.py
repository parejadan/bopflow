import cv2
import numpy as np
from bopflow.models.utils import DOutput


def draw_outputs(img, detections: [DOutput]):
    box_color = (255, 0, 0)
    box_width = 2
    wh = np.flip(img.shape[0:2])
    for result in detections:
        x1y1 = (result.box.x1y1 * wh).astype(np.int32)
        x2y2 = (result.box.x2y2 * wh).astype(np.int32)

        img = cv2.rectangle(
            img=img,
            pt1=tuple(x1y1),
            pt2=tuple(x2y2),
            color=box_color,
            thickness=box_width,
        )
        text = "label: {}| score: {}".format(
            result.label.number, result.confidence_score
        )
        text_coordinates = (x1y1[0] + 5, x2y2[1] - 5)
        img = cv2.putText(
            img=img,
            text=text,
            org=text_coordinates,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=box_color,
            thickness=1,
        )
    return img
