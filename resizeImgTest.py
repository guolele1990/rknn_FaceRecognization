import os

import numpy as np
import cv2
from PIL import Image

#---------------------------------------------------#
#   对输入图像进行resize
#---------------------------------------------------#
def letterbox_image(image, size):
    ih, iw, _   = np.shape(image)
    w, h        = size
    scale       = min(w/iw, h/ih)
    nw          = int(iw*scale)
    nh          = int(ih*scale)

    image       = cv2.resize(image, (nw, nh))
    new_image   = np.ones([size[1], size[0], 3],np.uint8) * 255
    new_image[(h-nh)//2:nh+(h-nh)//2, (w-nw)//2:nw+(w-nw)//2] = image
    return new_image

srcPath = "img/"
dstPath = "img2/"
list_dir = os.listdir(srcPath)
image_paths = []
names = []
idx = 0

for name in list_dir:
    image_paths.append(srcPath+name)
    img_raw = Image.open(srcPath+name).convert('RGB')
    img_raw = cv2.cvtColor(np.asarray(img_raw),cv2.COLOR_RGB2BGR)
    img_raw = letterbox_image(img_raw, [640, 640])
    if os.path.exists(dstPath) == False:
        os.mkdir(dstPath)
    cv2.imwrite(dstPath+str(idx)+".jpg",img_raw)
    idx += 1
    names.append(name.split("_")[0])


