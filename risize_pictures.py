#!/usr/bin/python
from PIL import Image
import os, sys

path = "./data/resize_test/" #Change path to the one where pictures are
dirs = os.listdir( path )

def resize():
    for item in dirs:
        print(os.path.dirname(item))

        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((256,256), Image.ANTIALIAS)
            imResize.save(f + '.jpg', 'JPEG', quality=90)

    print("Resize done")

resize()
