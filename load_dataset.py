import os
import PIL as pil
import glob
import numpy as np
import imageio

def crop_center(img, cropx, cropy):
    (y, x, z) = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx]

def load_dataset_kid(size=(224, 224), dataset_path="",
                     excludes=[
                         'ampulla',
                         # 'inflammatory',
                         'polypoid',
                         'normalstom',
                         'normaleso',
                         'normalcolon',
                         'vascular',
                         # 'normalsb'
                     ]):
    train_dataset = []
    train_labels = []
    files = glob.glob(os.path.join(dataset_path,'abnormal/')+"*.png")
    files = files + glob.glob(os.path.join(dataset_path,'normal/')+"*.png")
    for file in files:
        load_file = True
        for exclude in excludes:
            if exclude in file:
                load_file = False
                break
        if load_file:
            img = imageio.imread(file)
            if img.shape == (360, 360, 3):
                img = crop_center(img, 320, 320)
                img = np.array(pil.Image.fromarray(img).resize(size))
                train_dataset.append(img)
                train_labels.append(1 if "abnormal" in file else 0)
    return np.array(train_dataset), np.array(train_labels)



