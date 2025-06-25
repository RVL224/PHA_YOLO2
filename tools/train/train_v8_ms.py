from ultralytics import YOLO
 
def main():
    # model = YOLO('/workspace/yolov10/ultralytics/cfg/models/v8/yolov8-obb_.yaml')
    model.train(data='/workspace/yolov10/ultralytics/cfg/datasets/HRSC2016_MS.yaml', epochs=300, imgsz=640, batch=64,device=[0,1,2,3] , name="v8_ms" )
if __name__ == '__main__':
    main()
