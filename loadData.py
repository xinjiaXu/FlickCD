import cv2
import numpy as np
import os
from torch.utils.data import Dataset

class makeDataset(Dataset):
    def __init__(self, data_path, data_list, transform=None, distillation=False,
                 teacher_data_path=None, image_dir_a='T1', image_dir_b='T2', label_dir='GT'):
        self.data_path = data_path
        self.data_list = data_list
        self.transform = transform
        self.use_distillation = distillation
        self.teacher_data_path = teacher_data_path
        self.image_dir_a = image_dir_a
        self.image_dir_b = image_dir_b
        self.label_dir = label_dir

    def __getitem__(self, i):
        pre_path = os.path.join(self.data_path, self.image_dir_a, self.data_list[i])
        post_path = os.path.join(self.data_path, self.image_dir_b, self.data_list[i])
        label_path = os.path.join(self.data_path, self.label_dir, self.data_list[i])

        pre_img = cv2.imread(pre_path)
        post_img = cv2.imread(post_path)
        label = cv2.imread(label_path, 0)
        img = np.concatenate((pre_img, post_img), axis=2)
        if self.transform:
            [img, label] = self.transform(img, label)
        else:
            img = img.transpose((2, 0, 1))
        data_idx = self.data_list[i]
        return img[0:3], img[3:6], label, data_idx

    def __len__(self):
        return len(self.data_list)

