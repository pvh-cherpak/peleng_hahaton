import csv
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

def process_truck_max_gpu(
    input_video_path: str,
    output_video_path: str,
    output_csv_path: str,
    model_weights: str = "yolo11l-seg.pt",  # l или x для максимальной точности
    img_size: int = 960,                    # 960 или 1280 (кратно 32) для четких границ
    conf_threshold: float = 0.45,
    iou_threshold: float = 0.5
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA не найдена! Проверьте установку PyTorch с поддержкой GPU.")

    device = torch.device("cuda:0")
    # Включаем автотюнер cuDNN и ускорение матричных операций Ampere (RTX 30-й серии)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f"GPU активирован: {torch.cuda.get_device_name(0)}")
    print(f"Загрузка высокоточной модели: {model_weights} (imgsz={img_size}, retina_masks=True)")

    model = YOLO(model_weights)
    model.to(device)

    TRUCK_CLASS_ID = 7  # Класс 'truck' в датасете COCO

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Не удалось открыть видео: {input_video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Цвет заливки силуэта (BGR: ярко-зеленый) и коэффициент прозрачности на GPU
    mask_color_gpu = torch.tensor([80.0, 220.0, 0.0], device=device, dtype=torch.float16)
    alpha = 0.45

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

        # Отключаем расчет градиентов для экономии VRAM и ускорения
        with torch.inference_mode():
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_number += 1
                timestamp_sec = round(frame_number / fps, 3)

                # Инференс с retina_masks=True (маски сразу в высоком разрешении на GPU)
                results = model.track(
                    source=frame,
                    classes=[TRUCK_CLASS_ID],
                    conf=conf_threshold,
                    iou=iou_threshold,
                    imgsz=img_size,
                    device=0,
                    half=True,
                    retina_masks=True,
                    persist=True,
                    verbose=False
                )

                result = results[0]

                if result.masks is not None and result.boxes is not None:
                    # Тензор масок уже находится в видеопамяти RTX 3050 Ti: (N, H, W)
                    masks_gpu = result.masks.data
                    if masks_gpu.shape[1:] != (height, width):
                        masks_gpu = F.interpolate(
                            masks_gpu.unsqueeze(1).float(),
                            size=(height, width),
                            mode="bilinear",
                            align_corners=False
                        ).squeeze(1)

                    bool_masks_gpu = masks_gpu > 0.5  # Бинарные маски на GPU
                    boxes = result.boxes

                    # Переносим кадр в видеопамять для быстрого смешивания слоев
                    frame_gpu = torch.from_numpy(frame).to(device, non_blocking=True).to(torch.float16)

                    # Объединяем все маски грузовиков в кадре и накладываем полупрозрачный цвет на GPU
                    combined_mask = bool_masks_gpu.any(dim=0)
                    frame_gpu[combined_mask] = (
                        frame_gpu[combined_mask] * (1.0 - alpha) + mask_color_gpu * alpha
                    )

                    # Возвращаем обработанный кадр в оперативную память
                    annotated_frame = frame_gpu.to(torch.uint8).cpu().numpy()

                    # Обработка каждого найденного грузовика на GPU
                    for i in range(len(boxes)):
                        single_mask = bool_masks_gpu[i]

                        # Точный попиксельный подсчет площади силуэта на GPU
                        area_px = int(single_mask.sum().item())
                        if area_px == 0:
                            continue

                        # Вычисление центра масс (центроида) прямо на GPU
                        y_indices, x_indices = torch.where(single_mask)
                        cx = int(x_indices.float().mean().item())
                        cy = int(y_indices.float().mean().item())

                        box = boxes[i]
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = round(float(box.conf[0].item()), 4)
                        track_id = int(box.id[0].item()) if box.id is not None else i + 1

                        # Отрисовка четкого внешнего контура, центра масс и текста
                        if i < len(result.masks.xy) and len(result.masks.xy[i]) >= 3:
                            pts = np.array(result.masks.xy[i], dtype=np.int32).reshape((-1, 1, 2))
                            cv2.polylines(annotated_frame, [pts], isClosed=True, color=(0, 255, 100), thickness=2)

                        cv2.circle(annotated_frame, (cx, cy), 5, (0, 0, 255), -1)
                        label = f"Truck #{track_id} ({conf:.2f})"
                        cv2.putText(
                            annotated_frame, label, (x1, max(y1 - 10, 20)),
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
                else:
                    annotated_frame = frame

                out_video.write(annotated_frame)

    cap.release()
    out_video.release()
    torch.cuda.empty_cache()
    print(f"Готово! Видео: {output_video_path}, CSV: {output_csv_path}")

if __name__ == "__main__":
    process_truck_max_gpu(
        input_video_path="C:\\Андрей\\SpaceHack\\Way1.FinalImage.CineCameraActor1.mp4",
        output_video_path="output_segmented_hq.mp4",
        output_csv_path="truck_frames_hq.csv",
        model_weights="yolo11l-seg.pt",  # Можно поставить "yolo11x-seg.pt" для абсолютного максимума
        img_size=960,                    # Если памяти хватает, можно поднять до 1280
        conf_threshold=0.45
    )



    