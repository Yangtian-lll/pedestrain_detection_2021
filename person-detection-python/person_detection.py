# -*- coding: utf-8 -*-
import sys

import argparse

import cv2
import torch
import numpy as np

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import colors, plot_one_box


class Person_Detection():
    def __init__(self, opt):

        ######################################################################################## 初始化
        print('Initializing...')
        self.source, self.weights, self.view_img, self.imgsz = opt.source, opt.weights, opt.view_img, opt.img_size

        # Initialize
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        stride = int(self.model.stride.max())  # model stride
        imgsz = check_img_size(self.imgsz, s=stride)  # check img_size
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
        if self.half:
            self.model.half()  # to FP16

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, imgsz, imgsz).to(self.device).type_as(next(self.model.parameters())))

    def detection(self, im0s): # 检测函数
        img = letterbox(im0s, 640, 32)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = self.model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms,
                                   max_det=opt.max_det)

        im0 = im0s.copy()
        # Process detections
        for i, det in enumerate(pred):  # 每张图的检测结果画出来
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if self.view_img:  # 添加矩形框
                        c = int(cls)  # integer class
                        if c != 0:
                            continue
                        label = None if opt.hide_labels else (self.names[c] if opt.hide_conf else f'{self.names[c]} {conf:.2f}')
                        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=opt.line_thickness)

        return im0


    def openimage(self, path):   # 打开图片识别
        try:
            Im = cv2.imread(path)  # 通过Opencv读入一张图片
            Im = self.detection(Im)
            cv2.imshow('result', Im)
            cv2.waitKey(0)
        except:
            print("打不开图片！！！")


    def openvideo(self, path):  # 打开视频识别
        # 获得视频的格式
        try:
            videoCapture = cv2.VideoCapture(path)
        except:
            print("打不开视频！！！")

        # 读帧
        success, frame = videoCapture.read()
        while success:
            frame = self.detection(frame)
            cv2.imshow('result', frame)
            cv2.waitKey(10)  # 延迟
            success, frame = videoCapture.read()  # 获取下一帧

        videoCapture.release()

    def opencamera(self):   # 打开相机识别
        # 获取摄像头视频
        try:
            cap = cv2.VideoCapture(0)
        except:
            print("打不开相机！！！")

        while cap.isOpened():
            ret, frame = cap.read()
            frame = self.detection(frame)
            cv2.imshow('result', frame)
            cv2.waitKey(10)  # 延迟
        cap.release()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=r'weights/models.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default=r'data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.35, help='IOU threshold for NMS')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum number of detections per image')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', default='true', help='display results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--hide-labels', default=True, action='store_true', help='hide labels')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    opt = parser.parse_args()

    my = Person_Detection(opt = opt)
    #my.openimage('data/test.jpg') # 图片检测
    my.openvideo('data/test.avi') # 视频检测
    #my.opencamera() #相机检测
