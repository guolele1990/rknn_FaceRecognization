# RV1126的人脸检测项目

首先需要把相应的训练好的retinaface模型以及mobilefaceNet模型拷贝到weights/下
保证NTP正常需要，使用USB ADB方式
(rknn) python -m rknn.bin.list_devices

# 量化Retinaface

python RetinafaceConvertTest.py

注意修改模型路径，以及输入输出层
代码中使用的是
ret = rknn.load_onnx(model='./weights/retinaface.onnx',
                             inputs='input0',
                             input_size_list=[[3,640,640]],
                             outputs=['output0','590','589'])
输入输出层可以使用netron查看，基本output0  590 589要分别对应loc, conf, landms。错了运行会报错

把不量化直接运行的结果判断一下，简单先判断是否人脸检测正常。
BUILD_QUANT = False
NEED_BUILD_MODEL = True
量化结果会打印每层的量化精度，再判断是否人脸检测正常
BUILD_QUANT = True
NEED_BUILD_MODEL = True

# MobilefaceNet量化

修改

cfg = cfg_facenet_mxnet

选择对应的模型配置

把不量化直接运行的结果判断一下，结果与pytorch工程里的输出对比
BUILD_QUANT = False
NEED_BUILD_MODEL = True
量化结果会打印每层的量化精度，结果与pytorch工程里的输出对比
BUILD_QUANT = True
NEED_BUILD_MODEL = True

测试：
先编码

python encoding.py

图片预测：

python predict.py

rtsp预测：

python rtspPredict.py









