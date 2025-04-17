import pandas as pd
from glob import glob
import os

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from skimage import exposure

import torch

import torchvision
import torch.utils.data as data
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader

from sklearn.utils import shuffle

import pydicom


# 파일 경로 가져오기
# dcm파일, 2000장이니까 그냥 그대로 가져와서 읽기
#paths = glob("./data/*/*/*.dcm")
paths = glob("./data/**/*/*.dcm", recursive=True) # **는 재귀적으로 하위폴더 다 찾음

# CSV 파일 로드
df = pd.read_csv("./sorted_output_final.csv")

# 전체 데이터 저장할 리스트 초기화
img_paths = []
labels = []

# ID별로 해당하는 파일 경로 및 라벨 매칭
# chest 먼저
for _, row in df.iterrows():
    file_id = os.path.splitext(row["id"])[0]  # .png 제거
    label = row["chest"]  # 레이블 가져오기
    
    # 정상(0) 또는 비정상(1) 데이터만 저장
    if label in [0, 1]:
        matching_files = [p for p in paths if os.path.splitext(os.path.basename(p))[0] == file_id]
        
        # 리스트에 추가
        img_paths.extend(matching_files)
        labels.extend([label] * len(matching_files))  # 각 파일에 해당하는 라벨 추가

# 결과 출력 (확인용)
print("Total images (Normal + Abnormal):", len(img_paths))
print("Total labels (Normal + Abnormal):", len(labels))
print("Normal count:", labels.count(0)) # Normal X-rays:
print("Abnormal count:", labels.count(1)) # Abnormal X-rays:

# 데이터 스플릿
from sklearn.model_selection import train_test_split

labels = [int(l) for l in labels]

# 1차 스플릿
train_paths, temp_paths, train_labels, temp_labels = train_test_split(img_paths, labels, train_size=0.8, random_state=42, stratify=labels)

# 2차 스플릿
val_paths, test_paths, val_labels, test_labels = train_test_split(temp_paths, temp_labels, train_size=0.5, random_state=42, stratify=temp_labels)

# transforms

img_size = 512

from torchvision.transforms import Lambda

train_transform = transforms.Compose([
    Lambda(lambda img: transforms.functional.crop(img, top=0, left=0, height=img.size[1] * 2 // 3, width=img.size[0])),
    #transforms.Lambda(lambda img: transforms.functional.crop(img, img.size[1] // 4, 0, img.size[1] * 3 // 4, img.size[0])),  # 윗부분 1/4 제거
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    Lambda(lambda img: transforms.functional.crop(img, top=0, left=0, height=img.size[1] * 2 // 3, width=img.size[0])),
    #transforms.Lambda(lambda img: transforms.functional.crop(img, img.size[1] // 4, 0, img.size[1] * 3 // 4, img.size[0])),  # 윗부분 1/4 제거
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    Lambda(lambda img: transforms.functional.crop(img, top=0, left=0, height=img.size[1] * 2 // 3, width=img.size[0])),
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),    
])

# Dataset class
# 외부에서 path만들어서 넘기기
class NICUDataset(data.Dataset):
    def __init__(self, img_paths, labels, transform=None):
        # path
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):

        img_path = self.img_paths[index]

        label = self.labels[index]

        dicom = pydicom.dcmread(img_path)
        img_array = dicom.pixel_array.astype(np.float32)

        # 정규화 (0~1 사이)
        img_array -= img_array.min()
        img_array /= (img_array.max() + 1e-5)

        img_array = exposure.equalize_adapthist(img_array, clip_limit=0.03)

        # (0~255 정수형으로 변환 후 PIL.Image로 변환) - Grayscale
        img_uint8 = (img_array * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_uint8).convert("L")  # "L" = 1채널 흑백

        # transform
        img_transformed = self.transform(img_pil)

        #print(f"Label type: {type(label)}, value: {label}")

        return img_transformed, label
    
def dataloader(batch_size):
    train_dataset = NICUDataset(train_paths, train_labels, transform = train_transform)
    val_dataset = NICUDataset(val_paths, val_labels, transform=val_transform)
    test_dataset = NICUDataset(test_paths, test_labels, transform=test_transform)

    # train_dataset 데이터 로더 작성
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # validation_dataset 데이터 로더 작성
    validation_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # test_dataset 데이터 로더 작성
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle=True)

    return train_dataloader, validation_dataloader, test_dataloader


"""
from torchvision.utils import save_image

dataset = NICUDataset(train_paths, train_labels, transform=train_transform)
img_tensor, label = dataset[0]  # 0번째 샘플

# 이미지 저장
save_image(img_tensor, f"sample_{label}.png")
"""