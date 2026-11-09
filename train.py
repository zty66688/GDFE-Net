import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('yaml/yolo11n.yaml') 
    model.train(data='ultralytics/cfg/datasets/NEU-DET.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=32,
                close_mosaic=10,
                device='0',
                workers=4, 
                optimizer='SGD', # using SGD
                patience=50, 
                # resume=True,
                project='runs/train',
                name='exp',
                )