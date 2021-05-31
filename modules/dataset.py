from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
import cv2 as cv
import os

class ImageDataset(Dataset):

    def load_images(self, path):
        data = {}
        print('Loading images and labels')
        for i, filename in tqdm(enumerate(os.listdir(path))):
            if filename.endswith(".jpg"):
                img = cv.imread(path+filename)
                data[i] = img
                continue
            else:
                continue
        return data

    def __init__(self, images_path, label_path, image_transforms):
        self.images = self.load_images(images_path)
        self.labels = pd.read_csv(label_path)
        self.cats = list(set(self.labels))
        self.transforms = image_transforms

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transforms != None:
            image = self.transforms(image)

        return image, label

    def __len__(self):
        return len(self.images)