from ultralytics import YOLO
import cv2

# YOLOモデルのロード
model = YOLO("yolov8x.pt")

# 動画のパス
path = "ex05/ex5.mp4"
cap = cv2.VideoCapture(path)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

output_path = "ex05/ex5b.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))


frame_idx = 0

while cap.isOpened() :
    ret, frame = cap.read()
    if not ret:
        break
    
    # モデルを使用して画像から物体を検出
    results = model(frame)

    # 検出結果からボックスを抽出
    boxes = results[0].boxes
    class_names = results[0].names

    for box in boxes:
        data = box.data.cpu().numpy()[0]  
        x1, y1, x2, y2, conf, cls = data
        class_id = int(cls)
        if class_names[class_id] == 'person':
            # バウンディングボックスを描画
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)  # 赤色の枠

    # 出力フレームを書き込み
    out.write(frame)
    frame_idx += 1
    print(f"フレーム{frame_idx}/{frame_count}")

# リソースの解放
cap.release()
out.release()
cv2.destroyAllWindows()
