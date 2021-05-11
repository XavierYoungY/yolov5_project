import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (check_img_size, non_max_suppression,
                           apply_classifier, scale_coords, xyxy2xywh,
                           plot_one_box, strip_optimizer)
from utils.torch_utils import select_device, load_classifier, time_synchronized

import colorsys
import random
import datetime
from socket import *
import json

coco_class_name = [
    'person', 'cellphone', 'identification', 'notebook', 'face', 'chest'
]

coco_chinese_name = {
    'person': '人',
    'cellphone': '',
    'identification': '证件',
    'notebook': '',
    'face': '脸',
    'chest': '躯干'
}



colors = [(187, 127, 77), (212, 82, 94), (118, 194, 101), (59, 154, 176),
          (255, 215, 0), (255, 250, 250)]

class_colors = {}
for i in range(len(coco_class_name)):
    class_name = coco_class_name[i]
    color = colors[i]
    class_colors[class_name] = color


def doConnect(host, port):
    sock = socket(AF_INET, SOCK_STREAM)
    try:
        sock.connect((host, port))
    except:
        pass
    return sock


def a_in_b(a, b):
    #a:xywh b:xyxy
    if a[0] > b[0] and a[0] < b[2] and a[1] > b[1] and a[1] < b[3]:
        return True
    else:
        return False


def filter_items(det, names, gn, keep_height=0.0):
    people = {}
    i = 0
    face_list = []
    chest_list = []
    identification_list = []

    results = []

    if det is not None and len(det):
        for *xyxy, conf, cls in det:

            class_name = names[int(cls)]
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) /
                    gn).view(-1).tolist()  #xy center

            center = (xyxy2xywh(torch.tensor(xyxy).view(
                1, 4))).view(-1).tolist()[:2]  #xy center
            if xywh[1] > keep_height:  # y在keep_height 下面
                if class_name == 'person':
                    people[i] = {
                        'xyxy': torch.tensor(xyxy).tolist(),
                        'face': False,
                        'chest': False,
                        'identification': False
                    }
                    i += 1
                elif class_name == 'face':
                    face_list.append(center)
                elif class_name == 'chest':
                    chest_list.append(center)
                elif class_name == 'identification':
                    identification_list.append(center)

    for person_id, person in people.items():
        person_location = person['xyxy']
        #if identification is in person
        if len(identification_list):
            for identification in identification_list:
                if a_in_b(identification, person_location):
                    person['identification'] = True
        # tell if this person has the front face
        if len(face_list):
            for face in face_list:
                if a_in_b(face, person_location):
                    person['face'] = True
        #tell if one can see the chest
            if len(chest_list):
                for chest in chest_list:
                    if a_in_b(chest, person_location):
                        person['chest'] = True

    people_not_wear_id = []
    for person_id, person in people.items():
        if person['face'] and person['chest'] and not person['identification']:
            person['not_wear_identification'] = True
            people_not_wear_id.append(person['xyxy'])

    if det is not None and len(det):
        for *xyxy, conf, cls in det:
            _ = torch.tensor(xyxy).tolist()
            class_name = names[int(cls)]
            if class_name == 'person' and _ in people_not_wear_id:
                CN_class_name = '未佩戴证件'
            else:
                CN_class_name = coco_chinese_name[class_name]
            color = class_colors[class_name]
            _.extend([CN_class_name, conf.item(), color])
            results.append(_)

    return results


def mess_send(des_socket, bboxes, timestamp, messageId, ip, length, stringData,
              des_ip, des_port):
    # print(ip)
    response_data = {
        "messageName": "FuncControlResponse",
        "messageId": messageId,
        "cameraIP": ip,
        "function": "NoIDDetection",
        "responseStatus": 1,
        "resultData": {
            "timestamp": timestamp,
            "position": bboxes,
            "image": stringData,
            "Messages": length
        }
    }
    # print(bboxes)
    response_data = json.dumps(response_data) + '<reqJson>'

    #print(sys.getsizeof(response_data))
    try:
        des_socket.send(response_data.encode("utf-8"))
    except error:
        print("\r\nsocket error,do reconnect" + ip)

        des_socket = doConnect(des_ip, des_port)
        #des_socket.send(response_data.encode("utf-8"))

    return des_socket


def detect(save_img=False):

    # 对方socket
    des_socket = socket(AF_INET, SOCK_STREAM)
    # 链接服务器
    des_ip, des_port = '127.0.0.1', 10002

    des_socket.connect((des_ip, des_port))
    messageId = 0
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]

    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith(
        'http') or source.endswith('.txt')

    # Initialize
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    x = torch.cuda.FloatTensor(256, 1024, 1000)
    del x

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(
            torch.load('weights/resnet101.pt',
                       map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None

    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = False
        view_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)]
              for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img
              ) if device.type != 'cpu' else None  # run once


    for path, img, im0s, vid_cap in dataset:

        # imgs, im0s_list = img_split(img, im0s)
        # for index_img in range(len(imgs)):
        #     img = imgs[index_img]
        #     im0s = im0s_list[index_img]

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred,
                                   opt.conf_thres,
                                   opt.iou_thres,
                                   classes=opt.classes,
                                   agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + (
                '_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1,
                                          0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4],
                                          im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                # for *xyxy, conf, cls in det:
                #     if save_img or view_img:  # Add bbox to image
                #         label = '%s %.2f' % (names[int(cls)], conf)
                #         plot_one_box(xyxy,
                #                      im0,
                #                      label=label,
                #                      color=colors[int(cls)],
                #                      line_thickness=3)

            filtered_results = filter_items(det, names, gn)
            # Stream results
            # if view_img:
            #     # print(0)
            #     cv2.imshow(p, im0)
            #     if cv2.waitKey(1) == ord('q'):  # q to quit
            #         raise StopIteration
            for i in filtered_results:
                if i[4] == '未佩戴证件':
                    label = 'no ID'
                else:
                    label = ''
                plot_one_box(i[:4],
                             im0,
                             label=label,
                             color=i[-1],
                             line_thickness=3)
            cv2.imshow('', im0)
            if cv2.waitKey(1) == ord('q'):  # q to quit
                raise StopIteration

            stringData = []
            length = {}

            curr_time = datetime.datetime.now()
            timestamp = '%s-%s-%s %s:%s:%s' % (
                curr_time.year, curr_time.month, curr_time.day, curr_time.hour,
                curr_time.minute, curr_time.second)

            bboxes = filtered_results

            messageId += 1

            _, cam_ip = opt.source.split('@')

            # print(cam_ip)
            cam_ip = cam_ip.replace(':554/h264/ch1/main/av_stream', '')

            des_socket = mess_send(des_socket, bboxes, timestamp, messageId,
                                   cam_ip, length, stringData, des_ip,
                                   des_port)


if __name__ == '__main__':

    ip = "172.23.0.37"
    video_path = 'rtsp://admin:123456789bit@' + ip + '/main/Channels/1'
    # video_path = '/media/yy/DATA/data/vid/monitor/cellphone.mp4'

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',
                        nargs='+',
                        type=str,
                        default='chest.pt',
                        help='model.pt path(s)')
    parser.add_argument('--source',
                        type=str,
                        default=video_path,
                        help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output',
                        type=str,
                        default='inference/output',
                        help='output folder')  # output folder
    parser.add_argument('--img-size',
                        type=int,
                        default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres',
                        type=float,
                        default=0.4,
                        help='object confidence threshold')
    parser.add_argument('--iou-thres',
                        type=float,
                        default=0.5,
                        help='IOU threshold for NMS')
    parser.add_argument('--device',
                        default='0',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img',
                        action='store_true',
                        help='display results')
    parser.add_argument('--save-txt',
                        action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--classes',
                        nargs='+',
                        type=int,
                        help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms',
                        action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment',
                        action='store_true',
                        help='augmented inference')
    parser.add_argument('--update',
                        action='store_true',
                        help='update all models')
    opt = parser.parse_args()

    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in [
                    'yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt'
            ]:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
