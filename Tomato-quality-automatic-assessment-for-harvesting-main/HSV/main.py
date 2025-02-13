import cv2 as cv
import numpy as np

class YOLOv8Dataset(Dataset):
    def __init__(self, data_paths, transform=None):
        self.data_paths = data_paths
        self.transform = transform

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        image, target = load_data(self.data_paths[idx])  # Implement load_data function
        if self.transform:
            image = self.transform(image)
        return image, target



def main() -> None:
    ds = YOLODataset(
        img_path="/home/trong/Downloads/Dataset/Tomato/Detection/Relabeled/seperated_dataset/YOLO/Real_dataset/train",
        # data="/home/trong/Downloads/Dataset/Tomato/Detection/Relabeled/seperated_dataset/YOLO/Real_dataset/data.yaml"
    )


    return None

if __name__ == '__main__':
    main()