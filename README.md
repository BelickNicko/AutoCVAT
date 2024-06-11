# YoloToCoco
Auxiliary tool, auto-labling. 
Intended use: for tasks related to the training of custom solutions based on [Yolo models][1]  
What it does: implements the inference and converts the output to the Coco 1.0 format supported by CVAT

## Local Setup
You need to have Python 3.10 or newer installed.
Run the following commands sequentially in your terminal:

1. Clone this repository to your local machine.
2. Navigate to the created folder using the command cd.

Install all necessary libraries:

`pip install -r requirements.txt`

This repository supports all relevant models from Ultralytics: YOLOv8 for detection and segmentation, FastSAM, and YOLO-World (Real-Time Open-Vocabulary Object Detection).

## Adaptive Commands
To interact with the repository, you need to set the following command in the cmd terminal:

`python main.py --img_folder=image_cars  --weights=yolov8m.pt  --classes=car --classes_to_detect=2 --annotations_zip=cars_annotations `


**Table 1. Explanation of clip command values**

| № | Command               | Description                                                                                     |
|---|-----------------------|-------------------------------------------------------------------------------------------------|
| 1 | `--img_folder=`       | Path to the folder containing images                                                            |
| 2 | `--weights=`          | Path to the model weights file                                                                  |
| 5 | `--annotations_zip=`  | Path to the zip archive with annotations for images to be uploaded to CVAT                      |
| 6 | `--use_yaml=`         | Использовать ли yaml файл для передачи классов COCO и их названия для CVAT                      |

[COCO classes supported by YOLO models][2] 

[1]: https://docs.ultralytics.com/ru/models/
[2]: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml
