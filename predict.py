import tensorflow as tf
from PIL import Image
import cv2
from FaceRecognition import Facenet
import os
import numpy as np
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


if __name__ == "__main__":
    model = Facenet(facenet_threhold=12)

    exit
    totalImg = os.listdir("./img")
    for i in range(len(totalImg)):

        # image_1 = input('Input image_1 filename:')
        image_path = "img/"+totalImg[i]
        try:
            image_1 = Image.open(image_path).convert('RGB')
        except:
            print('Image_1 Open Error! Try again!')
            continue

        # image_2 = input('Input image_2 filename:')
        # try:
        #     image_2 = Image.open(image_2)
        # except:
        #     print('Image_2 Open Error! Try again!')
        #     continue
        img_raw = cv2.imread(image_path)
        # img_raw = cv2.resize(img_raw, (640, 640))
        img_raw = letterbox_image(img_raw, [640, 640])
        img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

        r_image = model.detect_image(img)
        r_image = cv2.cvtColor(r_image, cv2.COLOR_RGB2BGR)
        cv2.imshow("after", r_image)
        cv2.waitKey(0)
        # print(probability)
