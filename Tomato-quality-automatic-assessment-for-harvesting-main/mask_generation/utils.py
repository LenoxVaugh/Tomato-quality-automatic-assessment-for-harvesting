import cv2
import torch
import torchvision
import numpy as np

from PIL import Image
from mask_generation.dataclass import DetectionResult
from typing import (
    Union,
    List,
    Dict,
    Optional
)

from transformers import (
    pipeline,
    AutoModelForMaskGeneration
)

from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import StackDataset
from torchvision.transforms import Compose, v2, InterpolationMode

__all__ = ["detect",
           "plot_detections",
           "init_dataloader"
           ]


def detect(batch_images: Union[Image.Image, List[Image.Image]],
           batch_labels: Union[str, List[str], List[List[str]]],
           nms: bool = True,
           topk: Optional[int] = None,
           threshold: Optional[float] = 0.3,
           device: Optional[str] = "cpu",
           model_path: Optional[str] = "IDEA-Research/grounding-dino-tiny"
           ) -> List[List[DetectionResult]]:
    # Chua post-processing bbox

    # Batching single input
    if isinstance(batch_images, Image.Image):
        # Single input
        batch_images = [batch_images]

    if isinstance(batch_labels, str):
        batch_labels = [[batch_labels]]
    elif all(isinstance(label, str) for label in batch_labels):
        batch_labels = [batch_labels]

    # ZeroShotObjectDetectionPipeline
    # from transformers import ZeroShotObjectDetectionPipeline
    pipe = pipeline(model=model_path,
                    task="zero-shot-object-detection",
                    device=device,
                    batch_size=len(batch_images),
                    torch_dtype=torch.float32
                    )

    batch_labels: List[List[str]] = [[label if label.endswith(".") else f"{label}." for label in item] for item in batch_labels]

    # See: ZeroShotObjectDetectionPipeline() from huggingface
    inputs: List[dict] = [{"image": img, "candidate_labels": label} for img, label in zip(batch_images, batch_labels)]

    # Output format: [[img1_dectection_dicts], [img2_dectection_dicts], ...]
    batch_outputs: List[List[Dict]] = pipe(inputs, threshold=threshold, topk=topk)

    batch_outputs: List[List[DetectionResult]] = [
        [DetectionResult.from_dict(detection_output) for detection_output in detection_outputs] \
        for detection_outputs in batch_outputs]

    if nms:
        boxes, scores = None, None
        for i in range(len(batch_outputs)):
            for detection in batch_outputs[i]:
                box: torch.Tensor[torch.float32] = torch.tensor(detection.box.xyxy, dtype=torch.float32)
                score: torch.Tensor[torch.float32] = torch.tensor(detection.score, dtype=torch.float32)

                boxes = box if boxes is None else torch.vstack([boxes, box])
                scores = score if scores is None else torch.vstack([scores, score])

            scores = scores.flatten()
            nms_results: torch.Tensor[torch.int64] = torchvision.ops.nms(boxes, scores, .3)
            batch_outputs[i] = list(map(batch_outputs[i].__getitem__, nms_results))
    return batch_outputs


########################################################################################################################
def plot_detections(
    image: Union[Image.Image, np.ndarray],
    detections: List[DetectionResult],
    save_name: str
) -> None:
    # Convert PIL Image to OpenCV format
    if isinstance(image, Image.Image):
        image: np.ndarray = np.array(image)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Iterate over detections and add bounding boxes and masks
    for detection in detections:
        label = detection.label
        score = detection.score
        box = detection.box
        mask = detection.mask

        # Sample a random color for each detection
        color = np.random.randint(0, 256, size=3)

        # Draw bounding box
        image = cv2.rectangle(image, (box.xmin, box.ymin), (box.xmax, box.ymax), color.tolist(), 2)
        image = cv2.putText(image, f'{label}: {score:.2f}', (box.xmin, box.ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, color.tolist(), 2)

        # If mask is available, apply it
        if mask is not None:
            # Convert mask to uint8
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            image = cv2.drawContours(image, contours, -1, color.tolist(), 2)

    if save_name:
        cv2.imwrite(save_name, image)
########################################################################################################################


def init_dataloader(root: str,
                    batch_size: int = 32,
                    num_workers: int = 4
                    ) -> DataLoader:
    dataset = ImageFolder(root=root,
                          transform=Compose([
                              v2.ToImage(),
                              v2.Resize((640, 640), InterpolationMode.BICUBIC),
                              v2.ToDtype(torch.float32, scale=True)])
                          )

    img_names: List[str] = [img_path.replace(root, "") for img_path, _ in dataset.imgs]
    stacked_dataset = StackDataset(**{"data": dataset, "fnames": img_names})
    dataloader = DataLoader(stacked_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dataloader
