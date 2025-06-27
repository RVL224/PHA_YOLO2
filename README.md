# PHA_YOLO
## Installation
  - Download program
```
git clone https://github.com/RVL224/PHA_YOLO.git
Change the folder name to PHA_YOLO
```
  - Create by Docker
```
cd PHA_YOLO/docker
docker build -t pha .
docker run -it -d --gpus all --name [YourContainerName] -v /path/to/dataset:/workspace/PHA_YOLO/datasets -v /path/to/program:/workspace/PHA_YOLO --shm-size=64g pha
docker exec -it [YourContainerName] /bin/bash
```
  - install env from souce.
```
pip install -e .
```
## Train
  - Datasets prepare
```
go to rvlab NAS 163.13.132.234 ,_實驗室成員/PHA_YOLO資料集, acount and password in my ftp
```
  - Start Training
```
python3 tools/train/train_Dota.py
python3 tools/train/train_hrsc2016.py
python3 tools/train/train_hrsc2016ms.py
python3 tools/train/train_cspdarknet_ms.py
python3 tools/train/train_cspdarknet_mvtec.py
python3 tools/train/train_v8_ms.py
python3 tools/train/train_v8_mvtec.py
```
  - The trained weight file will appear in /runs/obb
## test and Demo
  - Modify the path and execute the program
```
python3 tools/demo.py
python3 tools/tests/test_dota.py
python3 tools/tests/test_hrsc_map07.py
python3 tools/tests/test_ms_map12.py
python3 tools/tests/test_mvtec.py
```
