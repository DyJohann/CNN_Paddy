from os import walk
import os
from os.path import join
import numpy as np
import cv2
import random

path = "Paddy\\"

for root, dirs, files in walk(path):
    for f in files:
        fullpath = join(root, f)
        img = cv2.imread(fullpath)

        print(root, random.uniform(0, 1))
        if img is None:
            # os.remove(fullpath)
            print("None !")
        else:
            height, width = img.shape[:2]
            BorderSize = 0
            if (height > width):
                BorderSize = height
            else:
                BorderSize = width
            nimg = cv2.copyMakeBorder(img, int((BorderSize - height) / 2), int((BorderSize - height) / 2)
                                      , int((BorderSize - width) / 2), int((BorderSize - width) / 2),
                                      cv2.BORDER_CONSTANT)
            if random.uniform(0, 1) > 0.3:
                sPath = "train\\" + root
            else:
                sPath = "test\\" + root
            if not os.path.isdir(sPath):
                os.mkdir(sPath)
        cv2.imwrite(sPath + "\\" + f, nimg)
