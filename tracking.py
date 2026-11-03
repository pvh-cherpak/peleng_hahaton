import csv
import cv2
import numpy as np
import torch
from ultralytics import YOLO

def process_truck_segmentation_gpu(
    input_video_path: str,
    output_video_path: str,
    output_csv_path: str,
    model_weights: str = "yolov8n-seg.pt",
    conf_threshold: float = 0.45
):
    # Проверка доступности RTX 3050 Ti
    if torch.cuda.is_available():
        device = 0  # Индекс дискретной видеокарты NVIDIA
        use_half = True  # FP16 для тензорных ядер RTX (экономит VRAM и повышает FPS)
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Используется GPU: {gpu_name} (режим FP16 включен)")
    else:
        device = "cpu"
        use_half = False
        print("ВНИМАНИЕ: CUDA не найдена, скрипт работает на процессоре (CPU)!")

    # Загрузка модели и перенос на видеокарту
    model = YOLO(model_weights)
    model.to(device)

    TRUCK_CLASS_ID = 7  # ID класса 'truck' в COCO

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Не удалось открыть видео: {input_video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    with open(output_csv_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "frame_number",
            "timestamp_sec",
            "truck_id",
            "confidence",
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
            "silhouette_area_px",
            "centroid_x",
            "centroid_y"
        ])

        frame_number = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1
            timestamp_sec = round(frame_number / fps, 3)

            # Инференс на GPU с включенным FP16 (half=True)
            results = model.track(
                source=frame,
                classes=[TRUCK_CLASS_ID],
                conf=conf_threshold,
                device=device,
                half=use_half,
                imgsz=640,
                persist=True,
                verbose=False
            )

            result = results[0]
            overlay = frame.copy()

            if result.masks is not None and result.boxes is not None:
                # retina_masks=False по умолчанию, полигоны уже в координатах кадра
                polygons = result.masks.xy
                boxes = result.boxes

                for i, poly in enumerate(polygons):
                    if len(poly) < 3:
                        continue

                    pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))

                    area_px = round(cv2.contourArea(pts), 1)
                    moments = cv2.moments(pts)
                    if moments["m00"] != 0:
                        cx = int(moments["m10"] / moments["m00"])
                        cy = int(moments["m01"] / moments["m00"])
                    else:
                        cx, cy = 0, 0

                    box = boxes[i]
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = round(float(box.conf[0]), 4)
                    track_id = int(box.id[0]) if box.id is not None else i + 1

                    cv2.fillPoly(overlay, [pts], color=(0, 200, 80))
                    cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 100), thickness=2)
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

                    label = f"Truck #{track_id} ({conf:.2f})"
                    cv2.putText(
                        frame, label, (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                    )

                    writer.writerow([
                        frame_number,
                        timestamp_sec,
                        track_id,
                        conf,
                        x1, y1, x2, y2,
                        area_px,
                        cx, cy
                    ])

            alpha = 0.4
            annotated_frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            out_video.write(annotated_frame)

    cap.release()
    out_video.release()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Обработка завершена! Видео: {output_video_path}, CSV: {output_csv_path}")

if __name__ == "__main__":
    process_truck_segmentation_gpu(
        input_video_path="C:\\Андрей\\SpaceHack\\Way1.FinalImage.CineCameraActor1.mp4",
        output_video_path="output_segmented.mp4",
        output_csv_path="truck_frames_data.csv",
        model_weights="yolov8s-seg.pt",  # На RTX 3050 Ti можно использовать более точную yolov8s-seg.pt
        conf_threshold=0.45
    )