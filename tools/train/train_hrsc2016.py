from ultralytics import YOLO
 
def main():
    model = YOLO('/workspace/PHA_YOLO/ultralytics/cfg/models/v9/yolov9c_PHA.yaml', task="obb").load('/workspace/PHA_YOLO/yolov9c.pt')  # build from YAML and transfer weights
    model.train(data='/workspace/PHA_YOLO/ultralytics/cfg/datasets/hrsc.yaml', epochs=300, imgsz=640, batch=16, device=0,name="PHA-YOLO_hrsc2016")
if __name__ == '__main__':
    main()
    
