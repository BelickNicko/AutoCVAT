import numpy as np


class Element:
    # Class containing information about a specific crop
    def __init__(
        self, image: np.ndarray, image_id: int, file_name: str, width: float, height: float
    ) -> None:
        self.image_id = image_id
        self.file_name = file_name
        self.width = width
        self.height = height
        self.image = image  # Original image
        self.category_id = None  # List of detected classes
        self.bbox = None  # List of lists with xyxy box coordinates
        self.detected_masks = []  # List of np arrays with masks in case of yolo-seg
        self.annotations_id = None
        self.areas = None  # List of areas of bbox/masks depending on the task
        self.iscrowd = 0  # 0 | 1 object is not a group | group
