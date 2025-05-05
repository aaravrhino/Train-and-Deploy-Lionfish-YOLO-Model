import os
import sys
import argparse
import glob
import time
import cv2
import numpy as np
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv5 Inference Script")

    parser.add_argument('--model', required=True,
                        help='Path to YOLO model (e.g., runs/detect/train/weights/best.pt)')
    parser.add_argument('--source', required=True,
                        help='Image source: image file, folder, video file, USB index (usb0), or Picamera (picamera0)')
    parser.add_argument('--thresh', type=float, default=0.5,
                        help='Minimum confidence threshold for displaying objects (default: 0.5)')
    parser.add_argument('--resolution', default=None,
                        help='Resolution in WxH (e.g., 640x480)')
    parser.add_argument('--record', action='store_true',
                        help='Record video output as demo1.avi (requires --resolution)')
    return parser.parse_args()

def validate_inputs(model_path, source, resolution, record):
    if not os.path.exists(model_path):
        print('ERROR: Model path is invalid or not found.')
        sys.exit(1)

    if os.path.isdir(source):
        return 'folder'
    elif os.path.isfile(source):
        ext = os.path.splitext(source)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            return 'image'
        elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']:
            return 'video'
        else:
            print(f'Unsupported file extension: {ext}')
            sys.exit(1)
    elif 'usb' in source:
        return 'usb'
    elif 'picamera' in source:
        return 'picamera'
    else:
        print(f'Invalid source: {source}')
        sys.exit(1)

def load_source(source_type, source, resolution):
    if source_type == 'folder':
        return sorted([f for f in glob.glob(os.path.join(source, '*')) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])

    if source_type == 'image':
        return [source]

    if source_type in ['video', 'usb']:
        cap = cv2.VideoCapture(source if source_type == 'video' else int(source[3:]))
        if resolution:
            w, h = map(int, resolution.split('x'))
            cap.set(3, w)
            cap.set(4, h)
        return cap

    if source_type == 'picamera':
        from picamera2 import Picamera2
        w, h = map(int, resolution.split('x'))
        picam = Picamera2()
        picam.configure(picam.create_video_configuration(main={"format": 'RGB888', "size": (w, h)}))
        picam.start()
        return picam

def main():
    args = parse_args()
    source_type = validate_inputs(args.model, args.source, args.resolution, args.record)
    model = YOLO(args.model, task='detect')
    labels = model.names

    source = load_source(source_type, args.source, args.resolution)
    record = args.record
    resolution = args.resolution
    thresh = args.thresh

    if resolution:
        w, h = map(int, resolution.split('x'))
    else:
        w, h = None, None

    recorder = None
    if record:
        if not resolution or source_type not in ['video', 'usb']:
            print('Recording requires --resolution and a video/camera source.')
            sys.exit(1)
        recorder = cv2.VideoWriter('demo1.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (w, h))

    bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
                   (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]
    
    fps_history = []
    fps_avg_len = 200
    frame_count = 0

    while True:
        t_start = time.perf_counter()
        if source_type in ['image', 'folder']:
            if frame_count >= len(source):
                print("Done processing all images.")
                break
            frame = cv2.imread(source[frame_count])
            frame_count += 1
        elif source_type == 'video' or source_type == 'usb':
            ret, frame = source.read()
            if not ret:
                print("End of video/camera stream.")
                break
        elif source_type == 'picamera':
            frame = source.capture_array()

        if frame is None:
            print("Failed to read frame.")
            break

        if resolution:
            frame = cv2.resize(frame, (w, h))

        results = model(frame, verbose=False)[0]
        object_count = 0

        for box in results.boxes:
            conf = box.conf.item()
            if conf < thresh:
                continue
            xyxy = box.xyxy.cpu().numpy().squeeze().astype(int)
            classid = int(box.cls.item())
            label = f"{labels[classid]}: {int(conf * 100)}%"

            cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), bbox_colors[classid % 10], 2)
            cv2.putText(frame, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

            object_count += 1

        fps = 1 / (time.perf_counter() - t_start)
        fps_history.append(fps)
        if len(fps_history) > fps_avg_len:
            fps_history.pop(0)

        avg_fps = np.mean(fps_history)
        if source_type in ['video', 'usb', 'picamera']:
            cv2.putText(frame, f"FPS: {avg_fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        cv2.putText(frame, f"Objects: {object_count}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.imshow("YOLO Detection", frame)

        if recorder:
            recorder.write(frame)

        key = cv2.waitKey(0 if source_type in ['image', 'folder'] else 5)
        if key in [ord('q'), ord('Q')]:
            break
        elif key in [ord('s'), ord('S')]:
            cv2.waitKey()
        elif key in [ord('p'), ord('P')]:
            cv2.imwrite('capture.png', frame)

    print(f'Average FPS: {np.mean(fps_history):.2f}')
    if isinstance(source, cv2.VideoCapture):
        source.release()
    elif source_type == 'picamera':
        source.stop()
    if recorder:
        recorder.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
