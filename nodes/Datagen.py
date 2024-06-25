import os
import cv2
from elements.Element import Element


class DataGen:
    """Class for generating image data.

    Attributes:
        folder_path (str): Path to the folder with images.
    """

    def __init__(self, folder_path):
        """Initialization of the DataGen object.

        Args:
            folder_path (str): Path to the folder with images.
        """
        # Check if the folder exists
        self.folder_path = folder_path

    def process(self):
        """Processing images in the folder and creating elements.

        Returns:
            list: List of Element objects containing image information.
        """
        if not os.path.isdir(self.folder_path):
            print(f'Folder "{self.folder_path}" does not exist')
            return

        data_all_elements = []
        # Iterate through all files in the folder
        for num, filename in enumerate(os.listdir(self.folder_path)):
            file_parh = os.path.join(self.folder_path, filename)
            # Check if the file is an image
            if os.path.isfile(file_parh) and any(
                filename.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png"]
            ):
                try:
                    # Read the image
                    img = cv2.imread(file_parh)
                    if img is not None:
                        image_id = num + 1
                        height, width, _ = img.shape
                        # Add the element to the list
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
                    print(f"Error processing file '{filename}': {e}")

        return data_all_elements
