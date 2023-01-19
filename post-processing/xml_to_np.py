import xml.etree.ElementTree as ET

import numpy as np


def xml_to_np(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    if root.tag != "annotation":
        raise Exception(
            "pascal voc xml root element should be annotation, rather than {}".format(
                root.tag
            )
        )
    gt_boxes = []
    for elem in root.findall("object/bndbox"):
        print(elem.text)
        startrow = int(elem.find("startrow").text)
        endrow = int(elem.find("endrow").text)
        startcol = int(elem.find("startcol").text)
        endcol = int(elem.find("endcol").text)
        xmin = float(elem.find("xmin").text)
        ymin = float(elem.find("ymin").text)
        xmax = float(elem.find("xmax").text)
        ymax = float(elem.find("ymax").text)

        gt_boxes.append(
            [xmin, ymin, xmax, ymax, startrow, endrow, startcol, endcol]
        )
    np_gt_boxes = np.array(gt_boxes)
    np.set_printoptions(precision=2, suppress=True)
    return np_gt_boxes
