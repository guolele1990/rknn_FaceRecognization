import tensorflow as tf
from PIL import Image
import cv2
from FaceRecognition import Facenet
import os
import numpy as np
from timeit import default_timer as timer
import argparse
import os
import glob
import random
import time
from rtspdec import RTSCapture

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

ENABLE_CACHE_IMG = False
def detect_rtsp(model):

    #rtscap = RTSCapture.create("rtsp://172.16.3.44:10554/analyse_full")
    rtscap = RTSCapture.create("rtsp://172.16.3.194:554/live/av0")
    rtscap.start_read()
    accum_time = 0
    curr_fps = 0
    prev_time = timer()
    while rtscap.isStarted():
        ok, frame = rtscap.read_latest_frame()

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        if not ok:
            continue


        # inhere
        test = Image.fromarray(frame)#test is BGR
        img = np.asarray(test, np.uint8)
        if ENABLE_CACHE_IMG:
            cv2.imshow("test",img)
            r = cv2.waitKey()
            if r == ord('s'):
                cv2.imwrite("face_dataset/gyl_1.jpg",img)
                print("save img")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = letterbox_image(img, [640, 640])
        # r_image = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

        r_image = model.detect_image(img)
        # r_image = cv2.cvtColor(r_image, cv2.COLOR_RGB2BGR)



        # result = np.asarray(r_image)
        #
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
            print("fps %s"%fps)
        # cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #             fontScale=0.50, color=(255, 0, 0), thickness=2)
        # cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        # cv2.imshow("result", result)

        cv2.imshow("after", r_image)
        # cv2.waitKey(0)
        #r_image.show()

        #cv2.imshow("cam", r_image)
        #cv2.destroyAllWindows()


    rtscap.stop_read()
    rtscap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    # text_retinaface = "{:.4f}".format(0.88777)
    # text = "{:.4f}".format(0.98456)
    # img_raw = cv2.imread("img/4.jpeg")
    # # img_raw = cv2.resize(img_raw, (640, 640))
    # old_image = letterbox_image(img_raw, [640, 640])
    #
    # b = [0,0]
    # cx = b[0]
    # cy = b[1] + 12
    #
    # cv2.putText(old_image, text, (cx, cy),
    #             cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
    # cx = b[0] + 60
    # cy = b[1] + 32
    # cv2.putText(old_image, text_retinaface, (cx, cy),
    #             cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
    # cv2.imshow("test",old_image)
    # cv2.waitKey()
    if ENABLE_CACHE_IMG != True:
        model = Facenet(facenet_threhold=0.6)
    else:
        model = []
    detect_rtsp(model)
