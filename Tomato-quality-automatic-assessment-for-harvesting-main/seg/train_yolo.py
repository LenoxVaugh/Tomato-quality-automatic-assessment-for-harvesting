import re
import os
import gc
import yaml
import torch
from ultralytics import YOLO
from ultralytics.engine import trainer, validator


def _load_yaml(path: str):
    # Customized yaml loader for scientific notation
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))

    with open(path, "r") as f:
        config = yaml.load(f, loader)
    return config


def train_yolo(model_name: str,
               weight_path: str = None,
               ) -> None:
    hyperpara: dict = _load_yaml("./train.yaml")
    hyperpara["name"] = model_name

    if weight_path is None:
        hyperpara["pretrained"] = False

    model = YOLO(task="det", verbose=True)
    model.train(**hyperpara)
    return None


def train_yolov8() -> None:
    for model_size in ("n", "s", "m", "l", "x"):
        model_name = f"yolov8{model_size}"
        weight_path = f"/home/trong/Downloads/Local/Pretrained_models/YOLOv8/Detection/{model_name}.pt"
        train_yolo(model_name, weight_path)
    return None


def train_yolov9(train_data_path: str) -> None:
    for model_size in ("t", "s", "m", "c", "e"):
        model_name = f"yolov9{model_size}"
        weight_path = f"/home/trong/Downloads/Local/Pretrained_models/YOLOv9/Detection/{model_name}.pt"
        train_yolo(model_name, weight_path, train_data_path)
    return None


def val_yolo(model_path: str, data_path: str, hyperpara_path: str) -> None:
    # Ref: https://docs.ultralytics.com/modes/val/#usage-examples
    hyperpara: dict = _load_yaml(hyperpara_path)

    # hyperpara = {
    #     "batch": .9,
    #     "imgsz": 640,
    #     "save_hybrid": False,
    #     "conf": 1e-3,
    #     "iou": .6,
    #     "half": True,
    #     "device": 0,
    #     "plots": True,
    # }

    model = YOLO(model=model_path, task="det", verbose=True)
    model.val(data=data_path, **hyperpara)
    return None


def main() -> None:
    from roboflow import Roboflow
    rf = Roboflow(api_key="VdRKOfGigHt8W6czx0JQ")
    project = rf.workspace("tomatosegmentation-uaevh").project("final-segmentation-dataset")
    version = project.version(4)
    dataset = version.download("coco")
    # test_data_path = "/home/trong/Downloads/Dataset/Tomato/Detection/laboro_rob2pheno/test.yaml"

    # train_yolov8()
    # train_yolov9()
    # train_yolov11()

    # val_yolo(
    #     "./DSP391m_tomato_detection/yolov8/yolov8x/weights/best.pt",
    #     test_data_path,
    #     "./test.yaml"
    # )


if __name__ == '__main__':
    main()