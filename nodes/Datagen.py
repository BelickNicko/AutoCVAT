import os
import cv2
from elements.Element import Element


class DataGen:
    """Класс для генерации данных изображений.

    Attributes:
        folder_path (str): Путь к папке с изображениями.
    """

    def __init__(self, folder_path):
        """Инициализация объекта класса DataGen.

        Args:
            folder_path (str): Путь к папке с изображениями.
        """
        # Проверяем, существует ли папка
        self.folder_path = folder_path

    def process(self):
        """Обработка изображений в папке и создание элементов.

        Returns:
            list: Список объектов Element, содержащих информацию об изображениях.
        """
        if not os.path.isdir(self.folder_path):
            print(f'Папка "{self.folder_path}" не существует')
            return

        data_all_elements = []
        # Проходим по всем файлам в папке
        for num, filename in enumerate(os.listdir(self.folder_path)):
            file_parh = os.path.join(self.folder_path, filename)
            # Проверяем, является файл изображением
            if os.path.isfile(file_parh) and any(
                filename.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png"]
            ):
                try:
                    # считываем изображение
                    img = cv2.imread(file_parh)
                    if img is not None:
                        image_id = num + 1
                        height, width, _ = img.shape
                        # добавляем элемент в список
                        data_all_elements.append(
                            Element(
                                image=img,
                                image_id=image_id,
                                file_name=filename,
                                width=width,
                                height=height,
                            )
                        )

                except Exception as e:
                    print(f"Ошибка при обработке файла '{filename}': {e}")

        return data_all_elements
