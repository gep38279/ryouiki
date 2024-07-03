import cv2
from ultralytics import YOLO
import mediapipe as mp
import math

model = YOLO("yolov8x-pose.pt")

img_path = "ex03/ex1.jpg"
img = cv2.imread(img_path)

img_results = model(img_path, save=True,
save_txt=True, save_conf=True)
img_keypoints = img_results[0].keypoints
img_img = cv2.imread("ex03/ex1.jpg")

center_point = [5,6,11,12]

# 動画ファイルの読み込み
video_path = "ex03/ex3a.mp4"
cap = cv2.VideoCapture(video_path)
# 顔以外の骨格の関節インデックス
skelton = [[5, 6], [6, 8], [8, 10], [5, 7], [7, 9], [6, 12], [5, 11], [11, 12], [12, 14], [14, 16], [11, 13], [13, 15]]

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

output_path = "ex03/ex3b.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path,fourcc,fps,(frame_width,frame_height))

max_frame = 5487
frame_idx = 0
x_img_center = 0
y_img_center = 0

for i in center_point:
    
    x_img_center += img_keypoints.data[0][i][0]
    y_img_center += img_keypoints.data[0][i][1]

x_img_ave =int(x_img_center /len(center_point))
y_img_ave =int(y_img_center /len(center_point))


while cap.isOpened() and frame_idx < max_frame:
    ret, frame = cap.read()
    if not ret:
        break

    # 骨格情報を取得
    results = model(frame)
    video_keypoints = results[0].keypoints

    x_img_center = 0
    y_img_center = 0
    x_vid = 0
    y_vid = 0

    for i in center_point:

        x_vid += video_keypoints.data[0][i][0]
        y_vid += video_keypoints.data[0][i][1]
    
    x_ave = int(x_vid /len(center_point))
    y_ave = int(y_vid /len(center_point))

    cv2.circle(frame,(x_ave,y_ave),5,(0,0,0))

    diff_x = x_ave-x_img_ave
    diff_y = y_ave-y_img_ave
    print(diff_x)
    total_dis = 0

    for (a,b) in skelton:
        total_dis += int(math.sqrt(((img_keypoints.data[0][a][0])-(video_keypoints.data[0][a][0]))**2 + ((img_keypoints.data[0][a][1])-(video_keypoints.data[0][a][1]))**2))
    for (a,b) in skelton:
        
        s = [int(img_keypoints.data[0][a][0]+ diff_x),int(img_keypoints.data[0][a][1]+ diff_y)]
        e = [int(img_keypoints.data[0][b][0]+ diff_x),int(img_keypoints.data[0][b][1]+ diff_y)]

        
        # s = [s[0] + diff_x,s[1] + diff_y]
        # e = [e[0] + diff_x,s[1] + diff_y]
        cv2.line(frame,s,e,color = (0,0,250),thickness = 2)

        s_vid = [int(video_keypoints.data[0][a][0]),int(video_keypoints.data[0][a][1])]
        e_vid = [int(video_keypoints.data[0][b][0]),int(video_keypoints.data[0][b][1])]
    
    
        if total_dis <100:
            cv2.line(frame,s_vid,e_vid,color = (250,0,0),thickness = 2)
        else:
            cv2.line(frame,s_vid,e_vid,color = (0,250,0),thickness = 2)

    # 描画したフレームを表示
    out.write(frame)
    frame_idx += 1
    print(f"フレーム{frame_idx}/{frame_count}")

cap.release()
out.release()
cv2.destroyAllWindows()
