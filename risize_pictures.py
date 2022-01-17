#!/usr/bin/python
from PIL import Image
import os, sys
from pathlib import Path

path_test = "./data/test/" #Change path to the one where pictures are
path_train = "./data/train/"

mashroom_kinds = os.listdir( path_test )

def resize(kind, type):
    if type == "test":
        path = path_test + kind + "/"
        destination_path = './data/resized_pictures/test/' + kind + '/'
    else:
        path = path_train + kind + "/"
        destination_path = './data/resized_pictures/train/' + kind + '/'

    dirs = os.listdir( path )

    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            file_name = item.split('.')[0]

            Path(destination_path).mkdir(parents=True, exist_ok=True)

            imResize = im.resize((256,256), Image.ANTIALIAS)
            imResize.save(destination_path + file_name + '.jpg', 'JPEG', quality=90)



for kind in mashroom_kinds:
    resize(kind, 'test')
    resize(kind, 'train')


print("Resize done")
