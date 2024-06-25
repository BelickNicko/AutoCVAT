import json
import numpy as np
from collections import defaultdict


class COCOConverter:
    """Класс для конвертации данных о детекциях в формат COCO (Common Objects in Context).

    Attributes:
        elements (list): Список элементов, содержащих информацию о детекциях.
        category_names (list): Список имен категорий объектов.

    Methods:
        __init__: Инициализирует объект класса COCOConverter.
        convert_to_coco: Конвертирует данные в формат COCO.
    """

    def __init__(self, elements, category_names, category_id):
        """Инициализация объекта класса COCOConverter.

        Args:
            elements (list): Список элементов, содержащих информацию о детекциях.
            category_names (list): Список имен категорий объектов.
        """
        self.elements = elements
        category_dict = defaultdict(list)
        for name, id_ in zip(category_names, category_id):
            category_dict[name].append(id_)
        self.category_dict = dict(category_dict)
    
    def convert_to_coco(self):
        """Конвертирует данные в формат COCO.

        Returns:
            str: JSON-строка, представляющая данные в формате COCO.
        """
        # Создание списка категорий
        categories = [
            {"id": self.category_dict[name][0] + 1, "name": name, "supercategory": ""}
            for name in self.category_dict
        ]

        # Создание списка изображений
        images = [
            {
                "id": elem.image_id,
                "width": elem.width,
                "height": elem.height,
                "file_name": elem.file_name,
                "license": 0,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": 0,
            }
            for elem in self.elements
        ]

        # Создание списка аннотаций
        annotations = []
        annotation_id = 1
        for elem in self.elements:
            
            counter = 0
            for bbox, area, category_id in zip(elem.bbox, elem.areas, elem.category_id):
                annotation = {
                    "id": annotation_id,
                    "image_id": elem.image_id,
                    "category_id": int(list(filter(lambda sublist: category_id-1 in sublist, self.category_dict.values()))[0][0] + 1),
                    "segmentation": [],
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": elem.iscrowd,
                }
                if elem.detected_masks:
                    annotation["segmentation"] = [list(elem.detected_masks[counter])]
                    annotation["attributes"] = {"occluded": False}
                else:
                    annotation["attributes"] = {"occluded": False, "rotation": 0}
                annotations.append(annotation)
                annotation_id += 1
                counter += 1

        # Создание словаря COCO
        coco_data = {
            "licenses": [{"name": "", "id": 0, "url": ""}],
            "info": {
                "contributor": "",
                "date_created": "",
                "description": "",
                "url": "",
                "version": "",
                "year": "",
            },
            "categories": categories,
            "images": images,
            "annotations": annotations,
        }

        # Преобразование в строку JSON
        coco_json = json.dumps(coco_data)

        return coco_json
