import cv2
import numpy as np
import os
from torch.utils.data import Dataset

class makeDataset(Dataset):
    def __init__(self, data_path, data_list, transform=None, distillation=False,
                 teacher_data_path=None, image_dir_a='T1', image_dir_b='T2', label_dir='GT',
                 patch_size=0):
        self.data_path = data_path
        self.data_list = data_list
        self.transform = transform
        self.use_distillation = distillation
        self.teacher_data_path = teacher_data_path
        self.image_dir_a = image_dir_a
        self.image_dir_b = image_dir_b
        self.label_dir = label_dir
        self.patch_size = patch_size
        if patch_size < 0:
            raise ValueError('patch_size must be zero or a positive integer')

        self.samples = []
        for name in data_list:
            image_path = os.path.join(self.data_path, self.image_dir_a, name)
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f'Could not read image: {image_path}')
            height, width = image.shape[:2]
            if patch_size and (height > patch_size or width > patch_size):
                if height % patch_size or width % patch_size:
                    raise ValueError(
                        f'{image_path} is {width}x{height}; dimensions must be divisible by '
                        f'patch_size={patch_size}'
                    )
                for top in range(0, height, patch_size):
                    for left in range(0, width, patch_size):
                        self.samples.append((name, top, left, height, width,
                                             patch_size, patch_size))
            else:
                self.samples.append((name, 0, 0, height, width, height, width))

    def __getitem__(self, i):
        name, top, left, height, width, crop_h, crop_w = self.samples[i]
        pre_path = os.path.join(self.data_path, self.image_dir_a, name)
        post_path = os.path.join(self.data_path, self.image_dir_b, name)
        label_path = os.path.join(self.data_path, self.label_dir, name)

        pre_img = cv2.imread(pre_path)
        post_img = cv2.imread(post_path)
        label = cv2.imread(label_path, 0)
        if pre_img is None or post_img is None or label is None:
            raise ValueError(f'Could not read paired images and label for {name}')
        if pre_img.shape[:2] != post_img.shape[:2] or pre_img.shape[:2] != label.shape[:2]:
            raise ValueError(
                f'Spatial dimensions differ for {name}: '
                f'A={pre_img.shape}, B={post_img.shape}, label={label.shape}'
            )
        if self.patch_size and (height > self.patch_size or width > self.patch_size):
            size = self.patch_size
            pre_img = pre_img[top:top + size, left:left + size]
            post_img = post_img[top:top + size, left:left + size]
            label = label[top:top + size, left:left + size]
        img = np.concatenate((pre_img, post_img), axis=2)
        if self.transform:
            [img, label] = self.transform(img, label)
        else:
            img = img.transpose((2, 0, 1))
        metadata = {
            'name': name,
            'top': top,
            'left': left,
            'height': height,
            'width': width,
            'crop_height': crop_h,
            'crop_width': crop_w,
        }
        return img[0:3], img[3:6], label, metadata

    def __len__(self):
        return len(self.samples)

