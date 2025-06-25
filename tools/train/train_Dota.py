from ultralytics import YOLO
 
def main():
    model = YOLO('/workspace/PHA_YOLO/ultralytics/cfg/models/v9/yolov9c_PHA-uvc.yaml', task="obb").load('/workspace/PHA_YOLO/yolov9c.pt')  # build from YAML and transfer weights
    model.train(data='/workspace/PHA_YOLO/ultralytics/cfg/datasets/DOTA.yaml', epochs=300, imgsz=640, batch=4, device=0,name="PHA-YOLO_DOTA")
if __name__ == '__main__':
    main()
