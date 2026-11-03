import csv
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

def create_occlusion_tracker_config(yaml_path: str = "truck_botsort.yaml") -> str:
    """Создает конфиг трекера BoT-SORT, настроенный на сложные перекрытия и пропадания."""
    config_content = """tracker_type: botsort
track_high_thresh: 0.35  
track_low_thresh: 0.08   # Подхват частично скрытого объекта или вида сзади
new_track_thresh: 0.40   
track_buffer: 60         # Хранить пропавший объект в памяти до 60 кадров (2 сек при 30 FPS)
match_thresh: 0.85       
fuse_score: True
gmc_method: sparseOptFlow
proximity_thresh: 0.5
appearance_thresh: 0.25
with_reid: False
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(config_content)
    return yaml_path


class TruckKalman2D:
    """Фильтр Калмана для стабилизации центра масс силуэта при перекрытиях и пропаданиях."""
    def __init__(self, init_x: float, init_y: float):
        self.kf = cv2.KalmanFilter(4, 2, 0, cv2.CV_32F)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        self.kf.measurementMatrix = np.eye(2, 4, dtype=np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.kf.statePost = np.array([[init_x], [init_y], [0.0], [0.0]], dtype=np.float32)
        self.missed_frames = 0

    def update(self, cx: float, cy: float, is_occluded: bool) -> tuple[int, int]:
        self.missed_frames = 0
        self.kf.predict()
        # При сильном перекрытии снижаем доверие к деформированному центру маски
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * (20.0 if is_occluded else 1.2)
        est = self.kf.correct(np.array([[np.float32(cx)], [np.float32(cy)]]))
        return int(est[0, 0]), int(est[1, 0])

    def predict_blind(self) -> tuple[int, int]:
        self.missed_frames += 1
        pred = self.kf.predict()
        return int(pred[0, 0]), int(pred[1, 0])


def is_near_frame_edge(x1: int, y1: int, x2: int, y2: int, w: int, h: int, margin: int = 80) -> bool:
    """Проверяет, появляется ли объект на границе кадра."""
    return x1 < margin or y1 < margin or x2 > (w - margin) or y2 > (h - margin)


def process_final_truck_pipeline(
    input_video_path: str,
    output_video_path: str,
    output_csv_path: str,
    model_weights: str = "yolo11l-seg.pt",
    img_size: int = 960,
    max_blind_frames: int = 25
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA не найдена! Проверьте, что PyTorch установлен с поддержкой видеокарты.")

    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    print(f"Инициализация GPU: {torch.cuda.get_device_name(0)}")
    tracker_yaml = create_occlusion_tracker_config()

    model = YOLO(model_weights)
    model.to(device)

    # Классы COCO: 2 (car), 5 (bus), 6 (train/кузов сзади), 7 (truck)
    VEHICLE_CLASSES = [2, 5, 6, 7]
    TRUCK_CLASS_ID = 7

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Не удалось открыть видеофайл: {input_video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    mask_color_gpu = torch.tensor([80.0, 220.0, 0.0], device=device, dtype=torch.float16)
    alpha = 0.45

    confirmed_trucks: dict[int, TruckKalman2D] = {}
    max_truck_area: dict[int, int] = {}

    with open(output_csv_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "frame_number", "timestamp_sec", "truck_id", "status",
            "confidence", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
            "silhouette_area_px", "raw_centroid_x", "raw_centroid_y",
            "filtered_centroid_x", "filtered_centroid_y"
        ])

        frame_number = 0

        with torch.inference_mode():
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_number += 1
                timestamp_sec = round(frame_number / fps, 3)

                # Трекинг с низким порогом conf=0.08 для захвата сзади и на краях кадра
                results = model.track(
                    source=frame,
                    classes=VEHICLE_CLASSES,
                    conf=0.08,
                    iou=0.55,
                    imgsz=img_size,
                    device=0,
                    half=True,
                    retina_masks=True,
                    augment=True,      # Улучшает распознавание масштаба (ось Z) и углов поворота
                    agnostic_nms=True,
                    tracker=tracker_yaml,
                    persist=True,
                    verbose=False
                )

                result = results[0]
                seen_ids_in_frame = set()
                valid_indices = []

                if result.masks is not None and result.boxes is not None and result.boxes.id is not None:
                    boxes = result.boxes
                    masks_gpu = result.masks.data
                    if masks_gpu.shape[1:] != (height, width):
                        masks_gpu = F.interpolate(
                            masks_gpu.unsqueeze(1).float(),
                            size=(height, width),
                            mode="bilinear",
                            align_corners=False
                        ).squeeze(1)
                    bool_masks_gpu = masks_gpu > 0.5

                    # Проверка и первичная регистрация объектов
                    for i in range(len(boxes)):
                        track_id = int(boxes.id[i].item())
                        cls_id = int(boxes.cls[i].item())
                        conf = float(boxes.conf[i].item())
                        x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
                        area_px = int(bool_masks_gpu[i].sum().item())

                        near_edge = is_near_frame_edge(x1, y1, x2, y2, width, height)
                        
                        # Условия валидации нового грузовика (с учетом вида сзади и краев кадра)
                        is_valid_start = (
                            track_id in confirmed_trucks or
                            (cls_id == TRUCK_CLASS_ID and conf >= 0.25) or
                            (near_edge and conf >= 0.10 and area_px > 300) or
                            (cls_id in [5, 6] and conf >= 0.20 and area_px > 500)
                        )

                        if is_valid_start:
                            if track_id not in confirmed_trucks:
                                y_idx, x_idx = torch.where(bool_masks_gpu[i])
                                if len(x_idx) > 0:
                                    init_cx = float(x_idx.float().mean().item())
                                    init_cy = float(y_idx.float().mean().item())
                                    confirmed_trucks[track_id] = TruckKalman2D(init_cx, init_cy)
                                    max_truck_area[track_id] = area_px

                            if track_id in confirmed_trucks:
                                valid_indices.append(i)
                                seen_ids_in_frame.add(track_id)

                    # Наложение масок на GPU
                    if valid_indices:
                        frame_gpu = torch.from_numpy(frame).to(device, non_blocking=True).to(torch.float16)
                        selected_masks = bool_masks_gpu[valid_indices]
                        combined_mask = selected_masks.any(dim=0)
                        frame_gpu[combined_mask] = (
                            frame_gpu[combined_mask] * (1.0 - alpha) + mask_color_gpu * alpha
                        )
                        annotated_frame = frame_gpu.to(torch.uint8).cpu().numpy()
                    else:
                        annotated_frame = frame.copy()

                    # Обработка активных детекций
                    for i in valid_indices:
                        track_id = int(boxes.id[i].item())
                        single_mask = bool_masks_gpu[i]
                        area_px = int(single_mask.sum().item())
                        if area_px == 0:
                            continue

                        max_truck_area[track_id] = max(max_truck_area.get(track_id, area_px), area_px)
                        conf = round(float(boxes.conf[i].item()), 4)
                        x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())

                        is_occluded = (area_px < 0.60 * max_truck_area[track_id]) or (conf < 0.30)
                        status = "occluded" if is_occluded else "visible"

                        y_indices, x_indices = torch.where(single_mask)
                        raw_cx = int(x_indices.float().mean().item())
                        raw_cy = int(y_indices.float().mean().item())

                        filt_cx, filt_cy = confirmed_trucks[track_id].update(raw_cx, raw_cy, is_occluded)

                        if i < len(result.masks.xy) and len(result.masks.xy[i]) >= 3:
                            pts = np.array(result.masks.xy[i], dtype=np.int32).reshape((-1, 1, 2))
                            color = (0, 165, 255) if is_occluded else (0, 255, 100)
                            cv2.polylines(annotated_frame, [pts], isClosed=True, color=color, thickness=2)

                        cv2.circle(annotated_frame, (raw_cx, raw_cy), 4, (255, 100, 0), -1)
                        cv2.circle(annotated_frame, (filt_cx, filt_cy), 6, (0, 0, 255), -1)

                        label = f"Truck #{track_id} [{status}] ({conf:.2f})"
                        cv2.putText(
                            annotated_frame, label, (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                        )

                        writer.writerow([
                            frame_number, timestamp_sec, track_id, status,
                            conf, x1, y1, x2, y2, area_px,
                            raw_cx, raw_cy, filt_cx, filt_cy
                        ])
                else:
                    annotated_frame = frame.copy()

                # Прогноз вслепую (когда грузовик полностью пропал или скрыт перекрытием)
                for track_id, kf_tracker in list(confirmed_trucks.items()):
                    if track_id not in seen_ids_in_frame:
                        if kf_tracker.missed_frames < max_blind_frames:
                            pred_cx, pred_cy = kf_tracker.predict_blind()
                            if 30 <= pred_cx < (width - 30) and 30 <= pred_cy < (height - 30):
                                cv2.circle(annotated_frame, (pred_cx, pred_cy), 6, (0, 255, 255), 2)
                                cv2.putText(
                                    annotated_frame, f"Truck #{track_id} [predicted]",
                                    (pred_cx + 10, pred_cy),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2
                                )
                                writer.writerow([
                                    frame_number, timestamp_sec, track_id, "predicted_blind",
                                    0.0, None, None, None, None, 0,
                                    None, None, pred_cx, pred_cy
                                ])
                        else:
                            del confirmed_trucks[track_id]

                out_video.write(annotated_frame)

    cap.release()
    out_video.release()
    if os.path.exists(tracker_yaml):
        os.remove(tracker_yaml)
    torch.cuda.empty_cache()
    print(f"Готово! Обработанное видео сохранено в: {output_video_path}")
    print(f"Пофреймовые данные сохранены в CSV: {output_csv_path}")

if __name__ == "__main__":
    process_final_truck_pipeline(
        input_video_path="C:\\Андрей\\SpaceHack\\Way1.FinalImage.CineCameraActor2.mp4",
        output_video_path="output2_final_segmented.mp4",
        output_csv_path="truck2_final_data.csv",
        model_weights="yolo11l-seg.pt",
        img_size=960
    )




    