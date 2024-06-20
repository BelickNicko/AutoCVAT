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
        segment=False,
        model_path="yolov8m.pt",
        imgsz=640,
        conf=0.4,
        iou=0.8,
        model=None,
        classes_list=None,
        minimize_points=True,
        conf_dict={},
    ) -> None:
        self.segment = segment
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
        self.minimize_points= minimize_points
    def process(self):
        """
        Обрабатывает набор данных для вывода.

        Returns:
            DataGen: Обработанный набор данных с результатами вывода.
        """
        mask_id = 0
        printed = False

        for element in self.elements:
            predictions = self.model.predict(
                element.image,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                verbose=False,
                classes=self.classes,
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
            if self.segment:
                try:
                    # список масок под формат COCO
                    if self.minimize_points:
                        element.detected_masks = self.minimize_contours(
                        predictions.masks.data.cpu().numpy(), filtered_indices, element.image
                    )
                    else:
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
                    if not printed:
                        print('Модель, веса которой вы указали, не поддерживает сегментацию!')
                        printed = True
                    element.detected_masks = []
                    element.areas = [box[2] * box[3] for box in element.bbox]
            else:
                element.detected_masks = []
                element.areas = [box[2] * box[3] for box in element.bbox]
            mask_id += len(element.annotations_id)
        return self.elements

    def minimize_contours(self, predictions_masks_xy, filtered_indices, image):
        """
        Преобразует маски в минимизированные контуры и сохраняет точки в требуемом формате.

        Args:
            predictions_masks_xy (list): Список сегментов в пиксельных координатах, представленных как тензоры.
            filtered_indices (list): Индексы масок, которые нужно обработать.
        Returns:
            list: Список минимизированных контуров в формате списка точек.
        """
        minimized_contours = []
        predictions_masks_xy = np.array(predictions_masks_xy)

        for i, mask in enumerate(predictions_masks_xy):
            if i in filtered_indices:
                mask_resized = cv2.resize(
                    np.array(mask),
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                # Находим все контуры без иерархии
                mask_contours, _ = cv2.findContours(
                    mask_resized.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
                )
                # Выбираем контур с наибольшей площадью
                max_contour = max(mask_contours, key=cv2.contourArea)
                # Упрощаем контур
                epsilon = 0.002 * cv2.arcLength(max_contour, True)  # Задаем точность аппроксимации
                approx = cv2.approxPolyDP(
                    max_contour, epsilon, True
                )  # Получаем аппроксимированный контур
                # Преобразуем все значения координат в целые числа
                mask_contour_int = list(map(int, approx.reshape(-1)))
                minimized_contours.append(mask_contour_int)
        return minimized_contours
