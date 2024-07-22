import numpy as np
from ultralytics import YOLO
from shapely.geometry import Polygon
import cv2


class Inferencer:
    """
    Class for performing inference using a YOLO model and processing the results.

    Args:
        elements (DataGen): Dataset for inference.
        model_path (str, optional): Path to the YOLO model file. Defaults to "yolov8m.pt".
        imgsz (int, optional): Input image size. Defaults to 640.
        conf (float, optional): Confidence threshold. Defaults to 0.7.
        iou (float, optional): Intersection over Union (IoU) threshold. Defaults to 0.8.
        model (YOLO, optional): Pre-initialized YOLO model. Defaults to None.
        classes_list (list, optional): List of class labels. Defaults to None.
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
        use_box_propt_sam=False,
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
        self.minimize_points = minimize_points

        self.use_box_propt_sam = use_box_propt_sam
        if self.use_box_propt_sam and self.segment:
            from ultralytics.models.fastsam import FastSAMPrompt

            self.FastSAMPrompt = FastSAMPrompt
            print(
                "Instance segmentation will be performed using a zero-shot approach "
                "thanks to feeding detected boxes through a pre-trained SAM network."
            )
            self.model_sam = YOLO("FastSAM-x.pt")

    def process(self):
        """
        Processes the dataset for inference.

        Returns:
            DataGen: Processed dataset with inference results.
        """
        mask_id = 0

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
            # Filter by confidence
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
            # Convert boxes to COCO format
            element.bbox = [
                [box[0], box[1], box[2] - box[0], box[3] - box[1]]
                for i, box in enumerate(predictions.boxes.xyxy.cpu().float().tolist())
                if i in filtered_indices
            ]
            # Calculate class IDs for each detected object in the image
            element.category_id = [
                cls + 1
                for i, cls in enumerate(predictions.boxes.cls.cpu().int().tolist())
                if i in filtered_indices
            ]
            # Find annotation ID for each detection relative to the entire dataset
            element.annotations_id = [mask_id + i for i in range(len(filtered_indices))]
            if self.segment:
                if self.use_box_propt_sam:
                    # process boxes as input pompt for sam

                    detected_masks = []

                    for box in element.bbox:
                        everything_results = self.model_sam(
                            element.image,
                            retina_masks=True,
                            imgsz=1024,
                            conf=0.1,
                            iou=0.9,
                            verbose=False,
                        )

                        # Prepare a Prompt Process object
                        prompt_process = self.FastSAMPrompt(element.image, everything_results)

                        # Bounding box prompt
                        ann = prompt_process.box_prompt(
                            bbox=[
                                int(box[0]),
                                int(box[1]),
                                int(box[2] + box[0]),
                                int(box[3] + box[1]),
                            ]
                        )[0]

                        # List of masks in COCO format
                        if self.minimize_points:
                            detected_mask = self.minimize_contours(
                                ann.masks.data.cpu().numpy(), [0], element.image
                            )[0]
                        else:
                            detected_mask = ann.masks.xy[0].flatten()

                        detected_masks.append(detected_mask)

                    element.detected_masks = [
                        [
                            (
                                float(detected_masks[i][j])
                                if j % 2 == 1
                                else float(detected_masks[i][j])
                            )
                            for j in range(len(detected_masks[i]))
                        ]
                        for i in range(len(detected_masks))
                    ]

                    element.areas = [0 for i, _ in enumerate(element.detected_masks)]
                    # Set flag for group object
                    element.isscrowd = 0
                else:
                    try:
                        # List of masks in COCO format
                        if self.minimize_points:
                            element.detected_masks = self.minimize_contours(
                                predictions.masks.data.cpu().numpy(),
                                filtered_indices,
                                element.image,
                            )
                        else:
                            detected_masks = [
                                mask.flatten()
                                for i, mask in enumerate(predictions.masks.xy)
                                if i in filtered_indices
                            ]
                            element.detected_masks = [
                                [
                                    (
                                        float(detected_masks[i][j])
                                        if j % 2 == 1
                                        else float(detected_masks[i][j])
                                    )
                                    for j in range(len(detected_masks[i]))
                                ]
                                for i in range(len(detected_masks))
                            ]

                        element.areas = [
                            Polygon(mask).area
                            for i, mask in enumerate(predictions.masks.xy)
                            if i in filtered_indices
                        ]
                        # Set flag for group object
                        element.isscrowd = 0
                    except AttributeError:
                        # If the model is not a segmentation model, the list of masks is empty and we calculate areas by bbox
                        element.detected_masks = []
                        element.areas = [box[2] * box[3] for box in element.bbox]
            else:
                element.detected_masks = []
                element.areas = [box[2] * box[3] for box in element.bbox]
            mask_id += len(element.annotations_id)
        return self.elements

    def minimize_contours(self, predictions_masks_xy, filtered_indices, image):
        """
        Converts masks into minimized contours and saves points in the required format.

        Args:
            predictions_masks_xy (list): List of segments in pixel coordinates represented as tensors.
            filtered_indices (list): Indices of masks to process.
        Returns:
            list: List of minimized contours in the form of a list of points.
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
                # Find all contours without hierarchy
                mask_contours, _ = cv2.findContours(
                    mask_resized.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
                )
                # Select the contour with the largest area
                max_contour = max(mask_contours, key=cv2.contourArea)
                # Simplify the contour
                epsilon = 0.002 * cv2.arcLength(max_contour, True)  # Set the approximation accuracy
                approx = cv2.approxPolyDP(
                    max_contour, epsilon, True
                )  # Get the approximated contour
                # Convert all coordinate values to integers
                mask_contour_int = list(map(int, approx.reshape(-1)))
                minimized_contours.append(mask_contour_int)
        return minimized_contours
