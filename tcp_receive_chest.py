import cv2
import time
import numpy as np
import os
from timeit import time
import sys
import queue
from threading import Thread
import subprocess as sp
from socket import *
import json
 
num_of_gpus = 2
ips = list(range(255))
IP_list = {}
for i, ip in enumerate(ips):
    IP_list[str(ip)] = i % num_of_gpus

gpu_in_use = {}
for i in range(num_of_gpus):
    gpu_in_use[i] = 0


def sub_process(cameraIP, threshold, gpu_id, user, pw):
    print(str(gpu_id) + '   ******************')
    video_path = 'rtsp://' + user + ':' + pw + '@' + cameraIP + ':554/h264/ch1/main/av_stream'
    
    python_path = 'video_tcp_chest.py'

    command = [
        '/home/ubuntu/anaconda3/envs/yolov5/bin/python',
        os.getcwd() + '/' + python_path, '--source', video_path,
        "--conf-thres",
        str(threshold), '--device',
        str(gpu_id)
    ]
    framechild = sp.Popen(command)
    # time.sleep(5)
    # framechild.kill()
    return framechild


def doConnect(host, port):
    sock = socket(AF_INET, SOCK_STREAM)
    try:
        sock.connect((host, port))
    except:
        pass
    return sock


def get_GPU_ID():
    key = min(gpu_in_use, key=gpu_in_use.get)
    return key


def tcp_receive():
    # 本机socket
    local_socket = socket(AF_INET, SOCK_STREAM)
    local_socket.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
    address = ('127.0.0.1', 10007)
    local_socket.bind(address)
    #listen里的数字表征同一时刻能连接客户端的程度.
    local_socket.listen(128)
  
    # 对方socket
    des_socket = socket(AF_INET, SOCK_STREAM)
    # 链接服务器
    des_ip, des_port = '127.0.0.1', 10002
    des_socket.connect((des_ip, des_port))

    all_cams = {}
    # for ip in IP_list.keys():
    #     all_cams[ip] = {'sp': '', 'funcControStatus': 0, 'gpu_id': 0}

    while True:
        client_socket, clientAddr = local_socket.accept()
        print('accept-------------------')
        while True:
            data = client_socket.recv(1024).decode("utf-8")
            if len(data):
                print(data)
                print('------------------------------------------------')
                if data == 'ping':
                    continue

                data = json.loads(data)
                messageId = data['messageId']
                #接收到IP
                cameraIP = data['cameraIP']
                if cameraIP not in all_cams:
                    all_cams[cameraIP] = {
                        'sp': '',
                        'funcControStatus': 0,
                        'gpu_id': 0
                    }

                function = data['function']
                threshold = data['params']['threshold']
                messageName = data['messageName']

                if messageName == 'FuncControlRequest':
                    funcControStatus = data['funcControStatus']
                    if funcControStatus == 0 and all_cams[cameraIP][
                            'funcControStatus'] == 1:
                        all_cams[cameraIP]['sp'].kill()
                        all_cams[cameraIP]['funcControStatus'] = 0
                        gpu_id = all_cams[cameraIP]['gpu_id']
                        gpu_in_use[gpu_id] -= 1
                        if gpu_in_use[gpu_id] < 0: gpu_in_use[gpu_id] = 0
                        print(gpu_in_use)
                        print('kill-------------------')
                        des_socket = response(des_socket, data, 1, des_ip,
                                              des_port)

                    elif funcControStatus == 1 and all_cams[cameraIP][
                            'funcControStatus'] == 0:
                        gpu_id = get_GPU_ID()
                        print('gpuid------' + str(gpu_id))
                        gpu_in_use[gpu_id] += 1
                        print(gpu_in_use)

                        user = data['userName']
                        pw = data['passWord']
                        cameraIP_sp = sub_process(cameraIP, threshold, gpu_id,
                                                  user, pw)
                        all_cams[cameraIP]['sp'] = cameraIP_sp
                        all_cams[cameraIP]['funcControStatus'] = 1
                        all_cams[cameraIP]['gpu_id'] = gpu_id

                        des_socket = response(des_socket, data, 1, des_ip,
                                              des_port)
                    else:
                        # 已经开启或者关闭
                        des_socket = response(des_socket, data, 1, des_ip,
                                              des_port)

                else:
                    responseStatus = data['responseStatus']
            else:
                break


def response(des_socket, data, responseStatus, des_ip, des_port):
    response_data = {
        "messageName": "FuncControlRequest",
        "messageId": data['messageId'],
        "cameraIP": data['cameraIP'],
        "function": data['function'],
        'responseStatus': responseStatus
    }
    response_data = json.dumps(response_data)
    try:
        des_socket.send(response_data.encode("utf-8"))
    except error:
        print(
            "\r\nsocket error,do reconnect--ObjDetection--------------------- "
        )

        des_socket = doConnect(des_ip, des_port)
        #des_socket.send(response_data.encode("utf-8"))

    return des_socket


if __name__ == "__main__":
    tcp_receive()
