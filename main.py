from __future__ import print_function
import sys
# sys.path.insert(0, "/opt/caffe-1.0/python")
# sys.path.insert(0, "/opt/caffe_plus/python")
# sys.path.insert(1, "../")
import caffe
import argparse
# import torch
import numpy as np
import cv2
import time


class CaffeInference(caffe.Net):
    """docstring for ClassName"""

    def __init__(self, model_file, pretrained_file, mean=None, use_gpu=False, device_id=0):
        self.__mean = mean
        if use_gpu:
            caffe.set_mode_gpu()
            caffe.set_device(device_id)
        else:
            caffe.set_mode_cpu()

        self.__net = caffe.Net(model_file, pretrained_file, caffe.TEST)

    def predict(self, img, input_name="data", output_name=["BboxHead_Concat", "ClassHead_Softmax", "LandmarkHead_Concat"]):
        img -= (self.__mean)
        # img *= 0.0078125
        if 3 == len(self.__mean):
            img = img.transpose(2, 0, 1)#hwc > chw
            new_shape = [1, img.shape[0], img.shape[1], img.shape[2]]
        else:
            new_shape = [1, 1, img.shape[0], img.shape[1]]

        img = img.reshape(new_shape)
        self.__net.blobs[input_name].reshape(*new_shape)
        self.__net.blobs[input_name].data[...] = img

        self.__net.forward()

        res = []

        res.append(self.__net.blobs[output_name].data)

        return (*res, img)


def demo(args):

    net = CaffeInference(args.deploy, args.trained_model, mean=(0, 0, 0), use_gpu=not args.cpu, device_id=0)
    print('Finished loading model!')

    # device = torch.device("cpu" if args.cpu else "cuda")
    data_layer = 'data'
    resize = 1

    # testing begin
    img_raw = cv2.imread(args.img_path, cv2.IMREAD_COLOR)
    img_raw = cv2.resize(img_raw,(112,112))

    img = np.float32(img_raw)



    im_height, im_width, _ = img.shape

    # scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    # scale = scale.to(device)

    tic = time.time()
    # loc, conf, landms, img = net.predict(img)  # forward pass
    #for gyl test
    # output_name = ["BboxHead_Concat", "ClassHead_Softmax", "LandmarkHead_Concat"]
    result, img = net.predict(img,input_name="data", output_name="fc1")  # forward pass
    print('net forward time: {:.4f}'.format(time.time() - tic))

    print(result)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Retinaface')

    parser.add_argument('-m', '--trained_model', default='./mobilefacenet.caffemodel',
                        type=str, help='Trained caffemodel path')
    parser.add_argument('--deploy', default='./mobilefacenet.prototxt', help='Path of deploy file')
    parser.add_argument('--img_path', default='./9.jpg', help='Path of test image')
    parser.add_argument('--cpu', action="store_true", default=True, help='Use cpu inference')
    parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
    parser.add_argument('--top_k', default=5000, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
    parser.add_argument('-s', '--save_image', action="store_true", default=True, help='save detection results')
    parser.add_argument('--show_image', action="store_true", default=True, help='show detection results')
    parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
    args = parser.parse_args()


    demo(args)
