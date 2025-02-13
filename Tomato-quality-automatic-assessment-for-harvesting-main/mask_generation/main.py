import os
import cv2
import torch
import numpy as np

from PIL import Image
from tqdm import tqdm
from typing import Union, List, Optional, Tuple
from utils import detect, plot_detections, init_dataloader
from dataclass import BoundingBox, DetectionResult

from torchvision.transforms import v2



def grounded_segmentation(
        image: torch.Tensor,
        labels: List[str],
        threshold: float = 0.3,
        polygon_refinement: bool = False,
        detector_id: Optional[str] = None,
        segmenter_id: Optional[str] = None
) -> Tuple[np.ndarray, List]:

    detections = segment(image, detections, polygon_refinement, segmenter_id)
    return np.array(image), detections





def segment(
    images: Union[List[Image.Image], Image.Image],
    detection_results: List[DetectionResult],
    polygon_refinement: bool = False,
    device: Optional[str] = "cpu",
    batch_size: Optional[int] = 1,
    model_path: Optional[str] = "facebook/sam-vit-base",
) -> List[DetectionResult]:
    def get_boxes(results: DetectionResult) -> List[List[List[float]]]:
        boxes = []
        for result in results:
            xyxy = result.box.xyxy
            boxes.append(xyxy)

        return [boxes]

    def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
        # Find contours in the binary mask
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the contour with the largest area
        largest_contour = max(contours, key=cv2.contourArea)

        # Extract the vertices of the contour
        polygon = largest_contour.reshape(-1, 2).tolist()

        return polygon

    def polygon_to_mask(polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Convert a polygon to a segmentation mask.

        Args:
        - polygon (list): List of (x, y) coordinates representing the vertices of the polygon.
        - image_shape (tuple): Shape of the image (height, width) for the mask.

        Returns:
        - np.ndarray: Segmentation mask with the polygon filled.
        """
        # Create an empty mask
        mask = np.zeros(image_shape, dtype=np.uint8)

        # Convert polygon to an array of points
        pts = np.array(polygon, dtype=np.int32)

        # Fill the polygon with white color (255)
        cv2.fillPoly(mask, [pts], color=(255,))

        return mask

    def refine_masks(masks: torch.BoolTensor, polygon_refinement: bool = False) -> List[np.ndarray]:
        masks = masks.cpu().float()
        masks = masks.permute(0, 2, 3, 1)
        masks = masks.mean(axis=-1)
        masks = (masks > 0).int()
        masks = masks.numpy().astype(np.uint8)
        masks = list(masks)

        if polygon_refinement:
            for idx, mask in enumerate(masks):
                shape = mask.shape
                polygon = mask_to_polygon(mask)
                mask = polygon_to_mask(polygon, shape)
                masks[idx] = mask
        return masks

    segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(device)
    processor = AutoProcessor.from_pretrained(segmenter_id)

    boxes = get_boxes(detection_results)
    inputs = processor(images=image, input_boxes=boxes, return_tensors="pt").to(device)

    outputs = segmentator(**inputs)
    masks = processor.post_process_masks(
        masks=outputs.pred_masks,
        original_sizes=inputs.original_sizes,
        reshaped_input_sizes=inputs.reshaped_input_sizes
    )[0]

    masks = refine_masks(masks, polygon_refinement)

    for detection_result, mask in zip(detection_results, masks):
        detection_result.mask = mask
    return detection_results


def main() -> None:
    device = "cuda"
    save_root = "/home/trong/Downloads/Local/Source/Python/semester_8/DSP391m/mask_generation/outputs"
    data_root = "/home/trong/Downloads/Dataset/Tomato/Detection/Relabeled/final_dataset/Tomato_det/"
    cache_dir = []

    dataloader = init_dataloader(root=data_root, batch_size=16)

    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        imgs, _ = batch["data"]
        fnames = batch["fnames"]

        labels: List[List[str]] = [["a tomato."] for _ in range(imgs.shape[0])]
        imgs: List[Image.Image] = [v2.ToPILImage()(img) for img in imgs]

        batch_detections: List[List[DetectionResult]] = detect(imgs, labels, 0.2, device)
        # segment(imgs, detections, True, "cuda")

        # Path to dir right before image file
        for img, detection, fname in zip(imgs, batch_detections, fnames):
            intermidate_dir = fname.split(os.sep)[:-2]

            if intermidate_dir not in cache_dir:
                cache_dir.append(cache_dir)
                os.makedirs(os.path.join(save_root, *intermidate_dir, "images"), exist_ok=True)
                os.makedirs(os.path.join(save_root, *intermidate_dir, "masks"), exist_ok=True)

            fname: str = fname.split(os.sep)[-1]
            detection_save_path = os.path.join(save_root, *intermidate_dir, "images", fname)
            mask_save_path = os.path.join(save_root, *intermidate_dir, "masks", fname)

            plot_detections(img, detection, detection_save_path)

        # if i == 3:
        #     break
    return None


if __name__ == '__main__':
    main()