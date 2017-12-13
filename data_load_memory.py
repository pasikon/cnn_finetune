from subprocess import check_output

import cv2
import numpy as np
import pandas as pd
from PIL import Image


def load_train_data_dir(data_root_dir, train_dir):
    data_root = data_root_dir

    train = train_dir

    classes = check_output(["ls", ("%s/%s" % (data_root, train))]).decode("utf8").strip().split("\n")
    dir_list = []

    for c in classes:
        files = check_output(["ls", "%s/%s/%s" % (data_root, train, c)]).decode("utf8").strip().split("\n")
        dir_list.append(files)
        files = check_output(["ls", "-l", "%s/%s/%s" % (data_root, train, c)]).decode("utf8").strip().split("\n")

    images = []
    pathes = []
    im_class = []
    im_height = []
    im_width = []
    for c, files in zip(classes, dir_list):
        for img in files:
            img_path = "%s/%s/%s/%s" % (data_root, train, c, img)
            pathes.append(img_path)
            im = Image.open(img_path)
            images.append(img)
            im_class.append(c)
            im_height.append(im.height)
            im_width.append(im.width)

    df_all = pd.DataFrame({"class": im_class, "height": im_height, "width": im_width}, index=images)

    df_by_class = pd.DataFrame(
        {"file": images, "class": im_class, "height": im_height, "width": im_width, "path": pathes},
        index=pathes)

    df_by_class

    one_hot_by_class = pd.get_dummies(df_by_class['class'])

    print('y_loaded: ' + str(one_hot_by_class.shape))

    # (4750,12)
    y_loaded = one_hot_by_class.as_matrix()

    imagez = []

    for im_path in one_hot_by_class.index:
        imageCV = cv2.imread(im_path)
        imagez.append(cv2.resize(imageCV, (300, 300), interpolation=cv2.INTER_AREA))

    # (4750, 300, 300, 3)
    imagez = np.array(imagez)
    print('x_loaded: ' + str(imagez.shape))

    return imagez, y_loaded
