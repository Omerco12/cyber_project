import albumentations as alb
import os
import cv2
augmentor = alb.Compose([alb.RandomCrop(width=160, height=160),
                         alb.HorizontalFlip(p=0.5),
                         alb.RandomBrightnessContrast(p=0.2),
                         alb.RandomGamma(p=0.2),
                         alb.RGBShift(p=0.2),
                         alb.VerticalFlip(p=0.5)]
                        )


data_dirs = ["dataset/train", "dataset/test"]

import cv2
import os

for data_dir in ['train', 'test']:
    for subdir in os.listdir(os.path.join('dataset', data_dir)):
        subdir_path = os.path.join('dataset', data_dir, subdir)
        for filename in os.listdir(subdir_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img_path = os.path.join(subdir_path, filename)
                img = cv2.imread(img_path)
                for num in range(100):
                    augmented = augmentor(image=img)
                    augmented_img_path = os.path.join(subdir_path, f"{filename}_augmented_{num}{os.path.splitext(filename)[-1]}")
                    cv2.imwrite(augmented_img_path, augmented['image'])



