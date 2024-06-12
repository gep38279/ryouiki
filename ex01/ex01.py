import cv2
from ultralytics import YOLO
import numpy as np

path = "https://cs.kwansei.ac.jp/~kitamura/lecture/RyoikiJisshu/images/ex1.jpg"
img = cv2.imread(path)

model = YOLO("yolov8x-pose.pt")
results = model(path, save=True,
save_txt=True, save_conf=True)
keypoints = results[0].keypoints
print(keypoints.data)

#print(len(keypoints.data))
#print(keypoints.data[0][0][0])

skelton = [[5,6],[6,8],[8,10],[5,7],[7,9],[6,12],[5,11],[11,12],[12,14],[14,16],[11,13],[13,15]]

for i in range(5,17):
    cv2.circle(img,center=(int(keypoints.data[0][i][0]),int(keypoints.data[0][i][1])),radius = 5,color = (0,250,0),thickness = -1)

for (a,b) in skelton:
    s = [int(keypoints.data[0][a][0]),int(keypoints.data[0][a][1])]
    e = [int(keypoints.data[0][b][0]),int(keypoints.data[0][b][1])]
    cv2.line(img,s,e,color = (0,0,250),thickness = 2)

cv2.imwrite('save.jpg',img)