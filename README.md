# Auto-labling CVAT
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

`python main.py --img_folder=image_cars  --weights=yolov8m.pt  --annotations_zip=cars_annotations `


**Table 1. Explanation of clip command values**

| № | Command               | Description                                                                                     |
|---|-----------------------|-------------------------------------------------------------------------------------------------|
| 1 | `--img_folder=`       | Path to the folder containing images                                                            |
| 2 | `--weights=`          | Path to the model weights file                                                                  |
| 5 | `--annotations_zip=`  | Path to the zip archive with annotations for images to be uploaded to CVAT                      |

The project also provides a configuration file where the parameter each class, confidentiality for each class, and the iou parameter are set.
An example of configuring a configuration file to configure defined classes and make the model confident in their presence:
```
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  14: bird
confs:
  0: 0.7
  1: 0.4
  2: 0.4
  3: 0.5
  14: 0.6
iou: 0.7
save_foto: True # whether to create a file .zip photos to upload to CVAT
classes_CVAT: True # should I create a json file with classes for CVAT
```

It is important to note that the number of confidentiality parameters must match the number of class names.

[COCO classes supported by YOLO models][2] 

The repository also optionally generates a json file for creating classes in the cvat project.

Example:
```
[
  {
    "name": "person",
    "id": 0,
    "color": "#CC3D90",
    "type": "any",
    "attributes": []
  },
  {
    "name": "bicycle",
    "id": 1,
    "color": "#6CC42C",
    "type": "any",
    "attributes": []
  },
  {
    "name": "car",
    "id": 2,
    "color": "#FAFCD4",
    "type": "any",
    "attributes": []
  },
  {
    "name": "motorcycle",
    "id": 3,
    "color": "#0E7D25",
    "type": "any",
    "attributes": []
  },
  {
    "name": "bird",
    "id": 4,
    "color": "#3738B7",
    "type": "any",
    "attributes": []
  }
]
```
The repository also generates a zip file with images to upload to the CVAT task

[1]: https://docs.ultralytics.com/ru/models/
[2]: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml
