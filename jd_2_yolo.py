import json
import os
import cv2
import shutil


def create_nameList(anns,img_folder,name_list):
    img_names = []
    for ann in anns:
        ann=ann.strip('\n').split(' ')
        img_name=os.path.join(img_folder, ann[0])
        img_names.append(img_name)
    with open(name_list,'w') as f:
        f.write('\n'.join(img_names))

def convert(size, box):
    #box:xmin ymin w h

    box=list(map(float,box))
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

def create_xywh_list(anns, ana_txt_save_path, img_folder, category_id):
    for ann in anns:
        ann = ann.strip(' \n').split(' ')
        filename = ann[0]
        # if filename == '10.110.0.12_01_20200116151636129_150.jpg':  
        #     print('')

        # img_path = os.path.join(img_folder, ann[0])
        # img=cv2.imread(img_path)
        # h,w,_=img.shape
        h=1080
        w=1920
        ana_txt_name = filename.rstrip('.jpg') + ".txt"
        with open(os.path.join(ana_txt_save_path, ana_txt_name), 'w') as f_txt:
            bboxes=ann[1:]
            for i in range(len(bboxes)//5):
                box = bboxes[i*5:(i+1)*5]
                cat=category_id[box[-1]]
                box = convert((w, h), box[:4])

                f_txt.write("%s %s %s %s %s\n" %
                            (cat, box[0], box[1], box[2], box[3]))







if __name__ == '__main__':

    category_id={}
    with open('data/datasets/jd/labels/val.json') as f:
        anns=json.load(f)
        categories = anns['categories']
        for cat in categories:
            category_id[cat['name']] = cat['id']

    with open('data/datasets/jd/labels/jd_xminyminwhc.txt','r') as f:
        anns=f.readlines()
        length=len(anns)


        train_length=int(length*0.9)
        train=anns[:train_length]
        val=anns[train_length:]





    img_folder = 'data/datasets/jd/images/'

    #for val
    name_list = 'data/datasets/jd/labels/val.txt'

    ana_txt_save_path = 'data/datasets/jd/labels/val'
    if os.path.exists(ana_txt_save_path):
        shutil.rmtree(ana_txt_save_path)
    os.makedirs(ana_txt_save_path)

    create_nameList(val, img_folder+'val/', name_list)

    create_xywh_list(val, ana_txt_save_path, img_folder, category_id)

    print('VAL_finish')

    #for train
    name_list = 'data/datasets/jd/labels/train.txt'

    ana_txt_save_path = 'data/datasets/jd/labels/train'
    if os.path.exists(ana_txt_save_path):
        shutil.rmtree(ana_txt_save_path)
    os.makedirs(ana_txt_save_path)

    create_nameList(train, img_folder+'train/', name_list)

    create_xywh_list(train, ana_txt_save_path, img_folder, category_id)

    print('train_finish')




    pass
