import numpy as np
from ultralytics import YOLO
from shapely.geometry import Polygon
import cv2


class Inferencer:
    """
    Класс для выполнения вывода с использованием модели YOLO и обработки результатов.

    Args:
        elements (DataGen): Набор данных для вывода.
        model_path (str, optional): Путь к файлу модели YOLO. По умолчанию "yolov8m.pt".
        imgsz (int, optional): Размер входных изображений. По умолчанию 640.
        conf (float, optional): Порог уверенности. По умолчанию 0.7.
        iou (float, optional): Порог пересечения по объединению (IOU). По умолчанию 0.8.
        model (YOLO, optional): Предварительно инициализированная модель YOLO. По умолчанию None.
        classes_list (list, optional): Список меток классов. По умолчанию None.
    """

    def __init__(
        self,
        elements,
        model_path="yolov8m.pt",
        imgsz=640,
        conf=0.4,
        iou=0.8,
        model=None,
        classes_list=None,
        conf_dict={},
    ) -> None:

        self.model_path = model_path
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        if model is None:
            self.model = YOLO(model_path)  # Load the model from the specified path
        else:
            self.model = model
        self.elements = elements
        self.classes = classes_list
        self.conf_dict = conf_dict

    def process(self):
        """
        Обрабатывает набор данных для вывода.

        Returns:
            DataGen: Обработанный набор данных с результатами вывода.
        """
        mask_id = 0

        for element in self.elements:
            try:
                predictions = self.model.predict(
                    element.image,
                    imgsz=self.imgsz,
                    conf=self.conf,
                    iou=self.iou,
                    verbose=False,
                    classes=self.classes,
                    retina_masks=True,
                )
            except:
                predictions = self.model.predict(
                    element.image,
                    imgsz=self.imgsz,
                    conf=self.conf,
                    iou=self.iou,
                    verbose=False,
                    classes=self.classes,
                    device="CPU",
                    retina_masks=True,
                )

            predictions = predictions[0]
            # фильтрация по confidence
            if len(self.conf_dict) != 0:
                filtered_indices = [
                    i
                    for i, (conf, classs) in enumerate(
                        zip(
                            predictions.boxes.conf.cpu().float().tolist(),
                            predictions.boxes.cls.cpu().int().tolist(),
                        )
                    )
                    if self.conf_dict[classs] <= conf
                ]

            else:
                filtered_indices = [i for i in range(len(predictions))]
            # переводим объекты boxes в формат используемый в COCO
            element.bbox = [
                [box[0], box[1], box[2] - box[0], box[3] - box[1]]
                for i, box in enumerate(predictions.boxes.xyxy.cpu().float().tolist())
                if i in filtered_indices
            ]
            # вычисляем id класса для каждого детекктируемого объекта на фотографии
            element.category_id = [
                cls + 1
                for i, cls in enumerate(predictions.boxes.cls.cpu().int().tolist())
                if i in filtered_indices
            ]
            # Для каждой детекции находим ее номер относительно всего набора фотографий
            element.annotations_id = [mask_id + i for i in range(len(filtered_indices))]
            try:
                # список масок под формат COCO
                detected_masks = [
                    mask.flatten()
                    for i, mask in enumerate(predictions.masks.xy)
                    if i in filtered_indices
                ]
                element.detected_masks = [
                    [
                        float(detected_masks[i][j]) if j % 2 == 1 else float(detected_masks[i][j])
                        for j in range(len(detected_masks[i]))
                    ]
                    for i in range(len(detected_masks))
                ]
                element.areas = [
                    Polygon(mask).area
                    for i, mask in enumerate(predictions.masks.xy)
                    if i in filtered_indices
                ]
                # ставим флаг на групповой объект
                element.isscrowd = 0
            except AttributeError:
                # если модель не сегментационная, то список масок пуст и мы вычисляем площади по bbox
                element.detected_masks = []
                element.areas = [box[2] * box[3] for box in element.bbox]

            mask_id += len(element.annotations_id)

        return self.elements
