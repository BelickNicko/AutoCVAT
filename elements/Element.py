import numpy as np

class Element:
    # Класс, содержаций информацию о конкретном кропе
    def __init__(
        self, image: np.ndarray, image_id: int, file_name: str, width: float, height: float
    ) -> None:
        self.image_id = image_id
        self.file_name = file_name
        self.width = width
        self.height = height
        self.image = image  # Исходное изображение
        self.category_id = None  # Список детектируемых классов
        self.bbox = None  # Список списков с координатами xyxy боксов
        self.detected_masks = []  # Список np массивов с масками в случае yolo-seg
        self.annotations_id = None
        self.areas = None  # Список площадей bbox/масок в зависимости от решаемой задачи
        self.iscrowd = 0  # 0 | 1 объект не группа|группа
