import numpy as np
import cv2
import os
# import matplotlib
# matplotlib.use('Agg')
# import urllib.request
# from matplotlib import gridspec
# from matplotlib import pyplot as plt
# from PIL import Image
# from tensorflow.python.platform import gfile
from rknn.api import RKNN
from PIL import Image
from sklearn import preprocessing
from scipy.spatial.distance import pdist, squareform

os.environ['RKNN_DRAW_DATA_DISTRIBUTE']="1"
def compute_cos_dis(x, y):
    cos_dist= (x* y)/(np.linalg.norm(x)*(np.linalg.norm(y)))
    return cos_dist.sum()


cfg_facenet_pytorch_onnx = {
    'modelType':"onnx",
    'model': './weights/mobilefacenet2.onnx',
    'inputs' : "input0",
    'input_size_list':[[3, 160, 160]],
    'outputs':['output0'],
    'reorder_channel':'0 1 2',
    'mean_values':[[0, 0, 0]],
    'std_values':[[255, 255, 255]],
    'input_img_size':(160,160)
}

cfg_facenet_mxnet_caffe = {
    'modelType':"caffe",
    'model': './weights/mobilefacenet.prototxt',
    'blob' : './weights/mobilefacenet.caffemodel',
    'inputs' : "input0",
    'input_size_list':[[3, 112, 112]],
    'outputs':['output0'],
    'reorder_channel':'2 1 0',
    'mean_values':[[127.5, 127.5, 127.5]],
    'std_values':[[128, 128, 128]],
    'input_img_size':(112,112)
}
cfg_facenet_mxnet = {
    'modelType':"mxnet",
    'inputs' : "input0",
    'input_size_list':[[3, 112, 112]],
    'outputs':['output0'],
    'reorder_channel':'2 1 0',
    'mean_values':[[0, 0, 0]],
    'std_values':[[1, 1, 1]],
    'symbol' : './model-symbol.json',
    'params' : './model-0000.params',
    'input_img_size':(112,112)
}

if __name__ == '__main__':

    cfg = cfg_facenet_mxnet
    im_file = './9.jpg'
    BUILD_QUANT = False
    RKNN_MODEL_PATH = './mobilefacenet.rknn'
    if BUILD_QUANT:
        RKNN_MODEL_PATH = './mobilefacenet_quant.rknn'

    # Create RKNN object
    rknn = RKNN()

    NEED_BUILD_MODEL = True
    if NEED_BUILD_MODEL:
        print('--> config model')

        rknn.config(reorder_channel=cfg['reorder_channel'], mean_values=cfg['mean_values'], std_values=cfg['std_values'],target_platform=['rv1126'],batch_size=1,quantized_dtype='dynamic_fixed_point-i16')

        print('done')
        print('--> Loading model')
        if cfg['modelType'] == "caffe":
            # Load caffe model
            print("load caffe model proto[%s] weights[%s]"%(cfg['model'],cfg['blob']))
            ret = rknn.load_caffe(model=cfg['model'],proto='caffe',blobs=cfg['blob'])
            if ret != 0:
                print('Load model failed! Ret = {}'.format(ret))
                exit(ret)
        elif cfg['modelType'] == "onnx":
            print("load onnx model model[%s] inputs[%s] input_size_list[%s] outputs[%s]"
                  % (cfg['model'],cfg['inputs'],cfg['input_size_list'],cfg['outputs']))
            ret = rknn.load_onnx(model=cfg['model'],
                                 inputs=cfg['inputs'],
                                 input_size_list=cfg['input_size_list'],
                                 outputs=cfg['outputs'])
            if ret != 0:
                print('Load retinaface failed!')
                exit(ret)
        elif cfg['modelType'] == "mxnet":# # Load mxnet model
            print("load mxnet model symbol[%s] params[%s] input_size_list[%s]" % (cfg['symbol'], cfg['params'], cfg['input_size_list']))
            ret = rknn.load_mxnet(cfg['symbol'], cfg['params'], cfg['input_size_list'])
            if ret != 0:
                print('Load mxnet model failed!')
                exit(ret)
            print('done')
        elif cfg['modelType'] == "keras":# # Load mxnet model
            ret = rknn.load_keras(model=cfg['model'])
            if ret != 0:
                print('Load keras model failed!')
                exit(ret)
            print('done')
        else:
            print('Load mxnet failed!')
            exit(-1)
        print('done')

        # Build model
        print('--> Building model')
        ret = rknn.build(do_quantization=BUILD_QUANT, dataset='./dataset.txt')
        if ret != 0:
            print('Build model failed!')
            exit(ret)
        print('done')

        if BUILD_QUANT:
            print('--> Accuracy analysis')
            rknn.accuracy_analysis(inputs='./dataset.txt',output_dir="./result",target='rv1126')
            print('done')

        # Export rknn model

        if False:#BUILD_QUANT:
            print('--> Export RKNN precompile model')
            ret = rknn.export_rknn_precompile_model(RKNN_MODEL_PATH)
        else:
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
    ret = rknn.init_runtime(target='rv1126', device_id='d81352278dd4de31',rknn2precompile=False)
    # ret = rknn.init_runtime(target='rv1126')
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    # Set inputs
    img = cv2.imread(im_file)
    img = cv2.resize(img, cfg['input_img_size'])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = np.random.randint(low=0, high=255, size=(112,112,3), dtype=np.uint8)


    image_1 = Image.open(im_file)
    image_1 = image_1.resize(cfg['input_img_size'], Image.BICUBIC)
    img = np.asarray(image_1, np.uint8)
    # img = np.float32(img)
    # img -= (128.0, 128.0, 128.0)
    print(img.shape)

    # img *= 0.0078125


    # inference

    print('--> inference')
    # outputs = rknn.inference(inputs=[img])
    outputs = rknn.inference(data_format='nhwc',inputs=[img])
    print('done')
    # outputs = np.expand_dims(outputs,axis=1)

    # outputs = preprocessing.normalize(outputs[0], norm='l2')


    print(outputs)
    image_1 = Image.open("1_001.jpg")
    image_1 = image_1.resize(cfg['input_img_size'], Image.BICUBIC)
    img = np.asarray(image_1, np.uint8)
    outputs0 = np.array(rknn.inference(data_format='nhwc', inputs=[img])[0])


    image_1 = Image.open("1_002.jpg")
    image_1 = image_1.resize(cfg['input_img_size'], Image.BICUBIC)
    img = np.asarray(image_1, np.uint8)
    outputs1 = np.array(rknn.inference(data_format='nhwc', inputs=[img])[0])


    l1 = np.linalg.norm(outputs1 - outputs0, axis=1)
    print("l1 %f"%l1)
    cosSim = 1 - pdist(np.vstack([outputs1, outputs0]), 'cosine')
    print("pdist %f"%cosSim)
    outputs1 = preprocessing.normalize(outputs1, norm='l2')
    outputs0 = preprocessing.normalize(outputs0, norm='l2')
    l1 = np.linalg.norm(outputs1 - outputs0, axis=1)
    print("after l2 l1 %f" % l1)

    rknn.eval_perf()
    rknn.release()

