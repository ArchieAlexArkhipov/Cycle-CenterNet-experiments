import cv2
import numpy as np


def print_LC(img_path, pred, out_path="../images/example_with_bounding_boxes.jpg"):
    img = cv2.imread(img_path)
    pred[np.isnan(pred)] = 0
    for box in pred:
        x0 = box[0]
        x1 = box[2]
        y0 = box[1]
        y1 = box[3]

        start_point = (int(x0), int(y0))
        end_point = (int(x1), int(y1))
        cv2.rectangle(img, start_point, end_point, color=(255, 50, 255), thickness=2)

        cv2.putText(
            img,
            "|".join(map(str, map(int, box[4:]))),
            (int(x0) + 15, int(y0) + 15),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.4,
            color=(255, 50, 255),
            thickness=1,
        )

    cv2.imwrite(out_path, img)
