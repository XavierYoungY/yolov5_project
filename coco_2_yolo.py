# -*- coding:utf-8 -*-

from __future__ import print_function
import os, sys, zipfile
import numpy as np
import json
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab


def extreme_points(points, x, y, size):

    left_index = np.argmin(x)
    left = points[left_index]
    right_index = np.argmax(x)
    right = points[right_index]
    up_index = np.argmin(y)
    up = points[up_index]
    bottom_index = np.argmax(y)
    bottom = points[bottom_index]

    extreme = np.array([up, bottom, left, right])
    convert_extreme(extreme, size)
    extreme = extreme.reshape(-1).tolist()

    return extreme


def convert_extreme(extreme, size):
    dw = 1. / (size[0])
    dh = 1. / (size[1])

    extreme[:, 0] *= dw
    extreme[:, 1] *= dh
    return extreme





def create_extreme_list(data, ids):
    dataDir = '/media/yy/DATA/datasets/COCO'
    dataType = 'val2017'
    annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
    coco = COCO(annFile)
    count = 0
    for img_id, img in coco.imgs.items():

        filename = img["file_name"]
        img_width = img["width"]
        img_height = img["height"]
        size = [img_width, img_height]

        ana_txt_name = 'extreme_' + filename.split(".")[0] + ".txt"
        f_txt = open(os.path.join(ana_txt_save_path, ana_txt_name), 'w')

        annIds = coco.getAnnIds(imgIds=img_id)
        if len(annIds) == 0:
            count += 1

        anns = coco.loadAnns(annIds)
        extremes = []
        for ann in anns:
            cat = ann['category_id']
            cat = ids[cat]
            segment = ann['segmentation']

            if not ann['iscrowd']:
                if len(segment) > 1:
                    # 如果遇到截断,拼起来
                    tmp = []
                    for s in segment:
                        tmp.extend(s)
                    segment = tmp
                segment = np.array(segment).reshape(-1, 2)
                x = segment[:, 0]
                y = segment[:, 1]
                extreme = extreme_points(segment, x, y, size)

            if ann['iscrowd']:
                # 如果是密集标注 bbox 四个边的中电作为极值点 定位边框 左上角 [x,y,w,h]
                x, y, w, h = ann['bbox']
                up = [x + w / 2, y]
                bottom = [x + w / 2, y + h]
                left = [x, y + h / 2]
                right = [x + w, y + h / 2]
                extreme = np.array([up, bottom, left, right])
                extreme = convert_extreme(extreme, size)
                extreme = extreme.reshape(-1).tolist()

            extremes.append(extreme)
            f_txt.write("%s %s %s %s %s %s %s %s %s\n" %
                        (cat, extreme[0], extreme[1], extreme[2], extreme[3],
                         extreme[4], extreme[5], extreme[6], extreme[7]))
        f_txt.close()
        # I = io.imread(dataDir + '/val2017/' + filename)

        # plt.imshow(I)
        # coco.showAnns(anns)
        # extremes=np.array(extremes).reshape(-1,2)
        # x=extremes[:,0];y=extremes[:,1]
        # color=[(238, 99, 99),(238, 173, 14),(0, 255, 127),(0, 229, 238)]
        # for i in range(4):
        #     x_=x[i::4]
        #     y_=y[i::4]
        #     c=color[i]
        #     plt.scatter(x_, y_,s=20)
        # plt.show()

def create_img_name_list(data,img_folder,name_list):
    # get all the img paths
    img_paths = []
    imgs = data.imgs
    for img in imgs.values():
        img_name = img['file_name']
        img_path = os.path.join(img_folder, img_name)
        img_paths.append(img_path)

    with open(name_list, 'w') as f:
        f.write('\n'.join(img_paths))


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])

    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0
    w = box[2]
    h = box[3]

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def create_xywh_list(data, ana_txt_save_path, ids):

    for img_id, img in data.imgs.items():

        filename = img["file_name"]
        img_width = img["width"]
        img_height = img["height"]
        size = [img_width, img_height]

        ana_txt_name = filename.split(".")[0] + ".txt"
        f_txt = open(os.path.join(ana_txt_save_path, ana_txt_name), 'w')

        annIds = data.getAnnIds(imgIds=img_id)

        anns = data.loadAnns(annIds)
        for ann in anns:
            box = convert((img_width, img_height), ann["bbox"])
            cat = ann["category_id"]
            cat = ids[cat]
            f_txt.write("%s %s %s %s %s\n" %
                            (cat, box[0], box[1], box[2], box[3]))
        f_txt.close()



if __name__ == "__main__":

    json_file = 'data/datasets/jd/val.json'  # # Object Instance 类型的标注

    img_folder='data/datasets/jd/images'

    name_list = 'data/datasets/jd/labels/val.txt'

    ana_txt_save_path = 'data/datasets/jd/labels/val'


    data = COCO(json_file)
    ids = {}
    for i, key in data.cats.items():
        ids[key['id']] = i

    # create_img_name_list(data, img_folder, name_list)

    create_xywh_list(data, ana_txt_save_path, ids)
    pass
