import numpy as np
import cv2
import os
import matplotlib
matplotlib.use('Agg')
# import urllib.request
# from matplotlib import gridspec
# from matplotlib import pyplot as plt
# from PIL import Image
# from tensorflow.python.platform import gfile
from rknn.api import RKNN
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from utils.box_utils import decode, decode_landm
import time
import argparse
import torch

# os.environ['RKNN_DRAW_DATA_DISTRIBUTE'] = "1"
#os.environ['NN_LAYER_DUMP'] = "1"

def compute_cos_dis(x, y):
    cos_dist = (x * y) / (np.linalg.norm(x) * (np.linalg.norm(y)))
    return cos_dist.sum()

cfg_mnet = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retinaface')

    parser.add_argument('-m', '--trained_model', default='/opt/deeplearning/ONNXToCaffe/model/face.caffemodel',
                        type=str, help='Trained caffemodel path')
    parser.add_argument('--deploy', default='/opt/deeplearning/ONNXToCaffe/model/face.prototxt',
                        help='Path of deploy file')
    parser.add_argument('--img_path', default='../curve/t1.jpg', help='Path of test image')
    parser.add_argument('--cpu', action="store_true", default=True, help='Use cpu inference')
    parser.add_argument('--confidence_threshold', default=0.2, type=float, help='confidence_threshold')
    parser.add_argument('--top_k', default=5000, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
    parser.add_argument('-s', '--save_image', action="store_true", default=True, help='save detection results')
    parser.add_argument('--show_image', action="store_true", default=True, help='show detection results')
    parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
    args = parser.parse_args()

    BUILD_QUANT = True
    RKNN_MODEL_PATH = './retinaface.rknn'
    if BUILD_QUANT:
        RKNN_MODEL_PATH='./retinaface_quant.rknn'
    im_file = './img/4.jpeg'

    # Create RKNN object
    rknn = RKNN()

    NEED_BUILD_MODEL = True
    if NEED_BUILD_MODEL:
        print('--> config model')
        rknn.config(reorder_channel='2 1 0', mean_values=[[104, 117, 123]], std_values=[[1, 1, 1]],
                    target_platform=['rv1126'], batch_size=1)
        print('done')

        # Load tensorflow model
        # print('--> Loading model')
        # ret = rknn.load_caffe(model='./face.prototxt',proto='caffe',blobs='./face.caffemodel')
        # if ret != 0:
        #     print('Load model failed! Ret = {}'.format(ret))
        #     exit(ret)
        # print('done')

        # # Load mxnet model
        # symbol = './mobilefacenet-symbol.json'
        # params = './mobilefacenet-0000.params'
        # input_size_list = [[3, 112, 112]]
        # print('--> Loading model')
        # ret = rknn.load_mxnet(symbol, params, input_size_list)
        # if ret != 0:
        #     print('Load mxnet model failed!')
        #     exit(ret)
        # print('done')

        # Load keras model
        # print('--> Loading model')
        # ret = rknn.load_keras(model='./facenet_mobilenet_all.h5')
        # if ret != 0:
        #     print('Load keras model failed!')
        #     exit(ret)
        # print('done')
        # print('--> Loading model')
        ret = rknn.load_onnx(model='./weights/retinaface.onnx',
                             inputs='input0',
                             input_size_list=[[3,640,640]],
                             outputs=['output0','590','589'])
        if ret != 0:
            print('Load retinaface failed!')
            exit(ret)
        print('done')

        # Build model
        print('--> Building model')
        ret = rknn.build(do_quantization=BUILD_QUANT, dataset='./dataset_retinaface.txt')
        if ret != 0:
            print('Build model failed!')
            exit(ret)
        print('done')

        if BUILD_QUANT:
            print('--> Accuracy analysis')
            rknn.accuracy_analysis(inputs='./dataset_retinaface.txt', output_dir="./retinaface_result", target='rv1126')
            print('done')

        # Export rknn model
        print('--> Export RKNN model')
        ret = rknn.export_rknn(RKNN_MODEL_PATH)
        if ret != 0:
            print('Export rknn failed!')
            exit(ret)
        print('done')
    else:
        # Direct load rknn model
        print('Loading RKNN model')
        ret = rknn.load_rknn(RKNN_MODEL_PATH)
        if ret != 0:
            print('load rknn model failed.')
            exit(ret)
        print('done')

    print('--> Init runtime environment')
    ret = rknn.init_runtime(target='rv1126', device_id='d81352278dd4de31', rknn2precompile=False)
    # ret = rknn.init_runtime(target='rv1126')
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    # Set inputs
    cfg = cfg_mnet
    device = torch.device("cpu")
    img_raw = cv2.imread(im_file)
    # img_raw = cv2.resize(img_raw, (640, 640))
    img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    img_raw = letterbox_image(img_raw, [640, 640])

    cv2.imshow("Src", img_raw)
    img_raw = np.asarray(img_raw)
    # img = np.array(img, np.uint8)

    print('--> inter')
    # print(x2)
    # 增加一个维度
    # img = img[:, :, :, np.newaxis]
    # 转换为模型需要的输入维度(640, 640 ,3)
    #opencv读的就是hwc格式，如果转换成其他img.transpose([3, 2, 0, 1])  nchw，推理时间会加长
    # img = img.transpose([3, 2, 0, 1])
    # img = img.transpose([2, 0, 1])
    print(img_raw.shape)


    im_height, im_width, _ = img_raw.shape
    scale = torch.Tensor([img_raw.shape[1], img_raw.shape[0], img_raw.shape[1], img_raw.shape[0]])
    scale = scale.to(device)


    # inference
    for i in range(1):
        print('--> inference')
        # loc, conf, landms = rknn.inference(data_format='nchw',inputs=[img])
        loc, conf, landms = rknn.inference(inputs=[img_raw])
        print('done')
        rknn.eval_perf()
        # rknn.accuracy_analysis()
        # rknn.eval_memory()


    img = img_raw.transpose(2, 0, 1)
    new_shape = [1, img.shape[0], img.shape[1], img.shape[2]]
    img = img.reshape(new_shape)

    # print(outputs)
    resize = 1
    loc = torch.tensor(loc)
    conf = torch.tensor(conf)
    landms = torch.tensor(landms)
    loc = loc.view(loc.shape[1], -1, 4)
    conf = conf.view(conf.shape[1], -1, 2)
    landms = landms.view(landms.shape[1], -1, 10)

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1 / resize
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > args.confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, args.nms_threshold)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:args.keep_top_k, :]
    landms = landms[:args.keep_top_k, :]

    dets = np.concatenate((dets, landms), axis=1)
    faces = 0
    # show image
    if True:
        for b in dets:
            if b[4] < args.vis_thres:
                continue
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            print(b)
            cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(img_raw, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            # landms
            cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
        # save image

        name = "test.jpg"
        cv2.imwrite(name, img_raw)
        if args.show_image:
            cv2.imshow("Demo", img_raw)
            cv2.waitKey(0)
    rknn.release()

