from ultralytics import YOLO
import cv2
import numpy

# YOLOモデルのロード
model = YOLO("yolov8x.pt")

# 画像のパス
path = "https://cs.kwansei.ac.jp/~kitamura/lecture/RyoikiJisshu/images/ex4.jpg"

# モデルを使用して画像から物体を検出
results = model(path, save=True)

# 検出結果からボックスを抽出
boxes = results[0].boxes
class_names = results[0].names

# 画像の読み込み
img = cv2.imread("ex04/ex4.jpg")  # YOLOによって保存された画像を読み込みます

# ボックスごとにループして人物を検出
for box in boxes:
    data = box.data.cpu().numpy()[0]
    x1, y1, x2, y2, conf, cls = data
    class_id = int(cls)
    if class_names[class_id] == 'person':
        # バウンディングボックスを描画
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)  # 赤色の枠

# 結果の表示
cv2.imwrite("ex04/person.jpg", img)
