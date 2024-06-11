# Библиотеки для работы с файлами и директориями
import os
import shutil
import yaml
import json

# Библиотека для создания CLI (Command Line Interface) интерфейсов
import click

# Кастомные модули и классы
from nodes.Datagen import DataGen
from nodes.Inference import Inferencer
from nodes.AnnotMaker import COCOConverter

# вспомогательные библиотеки
import random


class LengthMismatchError(Exception):
    """Кастомное исключение для несоответствия длины списка классов и словаря порогов уверенности."""

    def __init__(self, message):
        super().__init__(message)


def generate_and_save_class_list(original_classes, file_name="new_classes.json"):
    """
    Генерирует список словарей классов с рандомными цветами и сохраняет его в JSON файл.

    Параметры:
        original_classes (list): Список классов.
        file_name (str, optional): Имя файла, в который будет сохранен список классов.
            По умолчанию используется "new_classes.json".

    Возвращает:
        None
    """
    # Генерация случайного неповторяющегося цвета для каждого класса
    colors = ["#" + "".join(random.choices("0123456789ABCDEF", k=6)) for _ in original_classes]

    # Создание нового списка словарей с рандомными цветами и индексами в качестве id
    new_classes = []
    for idx, cls in enumerate(original_classes):
        cls_dict = {"name": cls, "id": idx, "color": colors[idx], "type": "any", "attributes": []}
        new_classes.append(cls_dict)

    # Сохранение нового списка в файл
    with open(file_name, "w") as file:
        json.dump(new_classes, file, indent=2)
    print(f"JSON для CVAT сохранен в файл {file_name}")


@click.command()
@click.option(
    "--img_folder",
    default="img_folder",
    help="Папка с изображениями (task из CVAT)",
    type=str,
)
@click.option(
    "--weights",
    default="yolov8m.pt",
    help="Путь к весам модели Yolo расширения .pt",
    type=str,
)
@click.option(
    "--annotations_zip",
    default="annotations",
    help="Названия zip архива аннотаций формата COCO CVAT",
    type=str,
)
@click.option(
    "--yaml_pth",
    default="coco_classes.yaml",
    help="Pairs of classes - ther indexes for COCO",
    type=str,
)
def main(**kwargs):

    result_folder = kwargs["annotations_zip"]
    MODEL_PTH = kwargs["weights"]
    input_folder = kwargs["img_folder"]
    use_yaml = kwargs["yaml_pth"]
    # Загрузка данных из YAML файла
    with open(use_yaml, "r") as yaml_file:
        data = yaml.safe_load(yaml_file)

    # Получение всех ключей и всех значений
    classes_CVAT = list(data["names"].values())
    classes_COCO = list(data["names"].keys())
    try:
        dict_confs = data["conf"]
        if classes_COCO != list(dict_confs):
            raise LengthMismatchError(
                f"Cписок классов и список ключей словаря порогов уверенности не совпадают. Каждый класс должен соответствовать порогу уверенности."
            )
    except KeyError:
        dict_confs = {}
    # Если результирующая папка уже существует, удаляем ее и создаем новую
    if os.path.exists(result_folder):
        shutil.rmtree(result_folder)
    # Если папка для выгрузки изображений существует, удаляем ее и создаем новую
    if os.path.exists("images_for_cvat"):
        shutil.rmtree("images_for_cvat")

    # Создаем результирующую папку для аннотаций и изображений для загрузки в cvat
    os.mkdir(result_folder)
    os.mkdir(result_folder + "/annotations")
    os.mkdir(result_folder + "/images")
    os.mkdir("images_for_cvat")

    # Получаем список файлов в исходной папке
    files = os.listdir(input_folder)

    # Сохраняем ли изображения в zip - архив для загрузки в CVAT
    save_foto = input("Сохранить изображения для загрузки в CVAT (YES/ДА -> сохраняем): ").lower()

    # Копируем каждый файл из исходной папки в целевую папку
    for file_name in files:
        source_file = os.path.join(input_folder, file_name)
        destination_file = os.path.join(result_folder + "/images", file_name)
        shutil.copy2(
            source_file, destination_file
        )  # Используем shutil.copy2 для копирования с метаданными
        if save_foto == "yes" or save_foto == "да":
            shutil.copy2(source_file, os.path.join("images_for_cvat", file_name))

    # Создаем zip ахрив для загрузки в CVAT
    shutil.make_archive("images_for_cvat", "zip", "images_for_cvat")
    # Удаляем папку изображений для загрузки в CVAT
    shutil.rmtree("images_for_cvat")
    # в терминале прописываем путь к результирующей папке
    if save_foto == "yes" or save_foto == "да":
        print(f"zip ахрив для загрузки в CVAT: images_for_cvat.zip")
    # Создаем строку JSON под формат COCO
    # Извлекаем каждое изображение из папки и создаем список elements
    datagen = DataGen(input_folder)
    elements = datagen.process()
    # Инференс каждой фотографии
    inferencer = Inferencer(
        elements, model_path=MODEL_PTH, classes_list=classes_COCO, conf_dict=dict_confs
    )
    elements = inferencer.process()
    # Создаем JSON объект
    converter = COCOConverter(elements, classes_CVAT, classes_COCO)
    results = converter.convert_to_coco()

    # Записываем JSON в файл
    output_file_path = os.path.join(result_folder + "/annotations", "instances_default.json")

    with open(output_file_path, "w") as output_file:
        output_file.write(results)

    # Создаем zip-архив из результирующей папки
    shutil.make_archive(result_folder, "zip", result_folder)

    # в терминале прописываем путь к результирующей папке
    print(f"Аннотации находятся по указанному пути: {result_folder}.zip")

    # Удаляем папку результирующую папку
    shutil.rmtree(result_folder)

    # Создадим json под CVAT проект
    generate_and_save_class_list(classes_CVAT)


if __name__ == "__main__":
    main()
