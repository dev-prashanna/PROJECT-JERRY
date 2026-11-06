"""Check the actual pretrained detector on an official Ultralytics reference image."""
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import cv2

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'realtime_depth_app'))
from src.robot_detection import ObjectPrediction,annotate_objects,decode_boxes


def main():
    directory=ROOT/'simulation/artifacts/yolo';directory.mkdir(parents=True,exist_ok=True)
    config=ROOT/'.cache/yolo';config.mkdir(parents=True,exist_ok=True)
    os.environ['YOLO_CONFIG_DIR']=str(config)
    from ultralytics import YOLO,settings
    import torch
    settings.update({'sync':False})
    image=cv2.imread(str(directory/'bus.jpg'))
    assert image is not None,'Download the official Ultralytics bus.jpg reference image first'
    model=YOLO(str(ROOT/'models/yolo/yolo11n.pt'))
    device='0' if torch.cuda.is_available() else 'cpu'
    model.predict(image,imgsz=416,device=device,half=device!='cpu',verbose=False)
    started=time.monotonic()
    result=model.predict(image,imgsz=416,conf=.3,device=device,half=device!='cpu',verbose=False)[0]
    prediction=ObjectPrediction(SimpleNamespace(id=1,bgr=image),decode_boxes(result),(time.monotonic()-started)*1000,device)
    labels={box.label for box in prediction.boxes}
    assert 'bus' in labels and 'person' in labels,labels
    cv2.imwrite(str(directory/'pretrained_check.png'),annotate_objects(prediction))
    report={'model':'yolo11n.pt','pretrained_classes':len(model.names),'device':device,
            'inference_ms':prediction.inference_ms,
            'reference_image':'https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg',
            'objects':[{'class':b.label,'confidence':b.confidence,'xyxy':b.xyxy} for b in prediction.boxes]}
    (directory/'pretrained_check.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
