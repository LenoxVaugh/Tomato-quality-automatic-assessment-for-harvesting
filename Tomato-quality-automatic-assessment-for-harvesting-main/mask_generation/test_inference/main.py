import os
import torch
import torchvision

from tqdm import tqdm
from PIL import Image
from typing import List
from pprint import pprint as pp
from mask_generation.dataclass import DetectionResult
from mask_generation.utils import plot_detections, detect


def test_single_inference() -> None:
    os.makedirs("single_inference", exist_ok=True)

    for img_fname in tqdm(os.listdir(f"..{os.sep}samples")):
        img: Image.Image = Image.open(f"..{os.sep}samples{os.sep}{img_fname}").convert("RGB").resize((640, 640))

        labels = ["a separated reddish tomato",
                  "a separated pinkish tomato",
                  "a separated greenish tomato",
                  "a separated rotten tomato",

                  "a separated reddish tomato hanging in the limb",
                  "a separated pinkish tomato hanging in the limb",
                  "a separated greenish tomato hanging in the limb",
                  "a separated rotten tomato hanging in the limb",

                  "occluded tomato",
                  ]

        detections: List[DetectionResult] = detect(img, labels, threshold=.2, device="cuda")[0]

        for i in range(len(detections)):
            detections[i].label = "tomato"

        save_name = os.path.join("single_inference", img_fname)
        plot_detections(img, detections, save_name)

        # print("Detection result")
        # print(img_fname)
        # pp(detections)
        # print("######################################################################################")
        # break

def test_batch_inference():
    pass


test_single_inference()