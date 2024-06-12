import cv2
from ultralytics import YOLO
import numpy as np
import math


model = YOLO("yolov8x-pose.pt")
paths = ["https://cs.kwansei.ac.jp/~kitamura/lecture/RyoikiJisshu/images/ex1.jpg","https://cs.kwansei.ac.jp/~kitamura/lecture/RyoikiJisshu/images/ex2_307.jpg","https://cs.kwansei.ac.jp/~kitamura/lecture/RyoikiJisshu/images/ex2_336.jpg","https://cs.kwansei.ac.jp/~kitamura/lecture/RyoikiJisshu/images/ex2_2015.jpg","https://cs.kwansei.ac.jp/~kitamura/lecture/RyoikiJisshu/images/ex2_3077.jpg","https://cs.kwansei.ac.jp/~kitamura/lecture/RyoikiJisshu/images/ex2_5175.jpg"]
img = []
results = []

keypoints_list=[]
for i in paths:
    result = model(i)
    keypoints_list.append(result[0].keypoints)

keypoints_base=keypoints_list[0]
distance_list = []
for id , keypoints in enumerate(keypoints_list[1:],1):
    total_dis=0
    for i in range (0,17):
        distance = math.sqrt(int(keypoints_base.data[0][i][0]-keypoints.data[0][i][0])**2+int(keypoints_base.data[0][i][1]-keypoints.data[0][i][1])**2)
        total_dis+=distance
    
    distance_list.append([id,total_dis])

distance_list.sort(key=lambda x: x[1])
for id,total_dis in distance_list:
    print(paths[id])
    print(total_dis)
