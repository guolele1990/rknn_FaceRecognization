import numpy as np
import cv2
import os
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
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn import preprocessing
from scipy.spatial.distance import pdist, squareform

def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)
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


# -----------------------------------------------------------------#
#   将输出调整为相对于原图的大小
# -----------------------------------------------------------------#
def retinaface_correct_boxes(result, input_shape, image_shape):
    new_shape = image_shape * np.min(input_shape / image_shape)

    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape

    scale_for_boxs = [scale[1], scale[0], scale[1], scale[0]]
    scale_for_landmarks = [scale[1], scale[0], scale[1], scale[0], scale[1], scale[0], scale[1], scale[0], scale[1],
                           scale[0]]

    offset_for_boxs = [offset[1], offset[0], offset[1], offset[0]]
    offset_for_landmarks = [offset[1], offset[0], offset[1], offset[0], offset[1], offset[0], offset[1], offset[0],
                            offset[1], offset[0]]

    result[:, :4] = (result[:, :4] - np.array(offset_for_boxs)) * np.array(scale_for_boxs)
    result[:, 5:] = (result[:, 5:] - np.array(offset_for_landmarks)) * np.array(scale_for_landmarks)

    return result


#---------------------------------#
#   计算人脸距离
#---------------------------------#
def face_distance(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty((0))
        # 已知所有人脸的特征向量和当前人脸的特征向量的欧氏距离
    cosSim = []
    for i , face_encode in enumerate(face_encodings):
        cosSim.append(pdist(np.vstack([face_encode, face_to_compare]), 'cosine')[0])
        # i+=1
    cosSim = np.array(cosSim)

    return cosSim
    # return np.linalg.norm(face_encodings - face_to_compare, axis=1)

#---------------------------------#
#   比较人脸
#---------------------------------#
def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=1):
    # (n)
    dis = face_distance(known_face_encodings, face_encoding_to_check)
    # for i in range(dis.size):
    #     print("dis "+dis[i])
    print("dis ",format(dis))

    return list(dis <= tolerance), dis


# --------------------------------------#
#   写中文需要转成PIL来写。
# --------------------------------------#
def cv2ImgAddText(img, label, left, top, textColor=(255, 255, 255)):
    img = Image.fromarray(np.uint8(img))
    # ---------------#
    #   设置字体
    # ---------------#
    font = ImageFont.truetype(font='model_data/simhei.ttf', size=20)

    draw = ImageDraw.Draw(img)
    label = label.encode('utf-8')
    draw.text((left, top), str(label, 'UTF-8'), fill=textColor, font=font)
    return np.asarray(img)



class Facenet(object):
    _defaults = {
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
        'out_channel': 64,

        'confidence_threshold' : 0.2,#预先框的阈值
        'nms_threshold' : 0.4,#nms 阈值
        'vis_thres' : 0.8,#人脸置信值
        'retinaface_rknn_model_path' : './retinaface_quant.rknn',
        'mobilefacenet_rknn_model_path': './mobilefacenet_quant.rknn',
        # 'retinaface_rknn_model_path': './retinaface.rknn',
        # 'mobilefacenet_rknn_model_path': './mobilefacenet.rknn',
        'target' : 'rv1126',
        'device_id' : 'd81352278dd4de31',
        # ----------------------------------------------------------------------#
        #   是否需要进行图像大小限制。
        #   输入图像大小会大幅度地影响FPS，想加快检测速度可以减少input_shape。
        #   开启后，会将输入图像的大小限制为input_shape。否则使用原图进行预测。
        #   keras代码中主干为mobilenet时存在小bug，当输入图像的宽高不为32的倍数
        #   会导致检测结果偏差，主干为resnet50不存在此问题。
        #   可根据输入图像的大小自行调整input_shape，注意为32的倍数，如[640, 640, 3]
        # ----------------------------------------------------------------------#
        "retinaface_input_shape": [640, 640, 3],
        # ----------------------------------------------------------------------#
        #   facenet所使用到的输入图片大小
        # ----------------------------------------------------------------------#
        "facenet_input_shape": [112, 112, 3],
        "letterbox_image": False,
        "facenet_l2_norm": False,
        # ----------------------------------------------------------------------#
        #   facenet所使用的人脸距离门限
        # ----------------------------------------------------------------------#
        "facenet_threhold": 1.0
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化Retinaface+facenet
    # ---------------------------------------------------#
    def __init__(self, encoding=0,**kwargs):
        #更新参数
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        self.generate()
        try:
            self.known_face_encodings = np.load("model_data/face_encoding.npy".format())
            self.known_face_names     = np.load("model_data/names.npy".format())
        except:
            if not encoding:
                print("载入已有人脸特征失败，请检查model_data下面是否生成了相关的人脸特征文件。")
            pass
        show_config(**self._defaults)

    def generate(self):
        # Create retinaface RKNN object
        self.retinaface_rknn = RKNN()
        self.mobilefacenet_rknn = RKNN()
        print('Loading retinaface_rknn model')
        ret = self.retinaface_rknn.load_rknn(self.retinaface_rknn_model_path)
        if ret != 0:
            print('load retinaface_rknn model failed.')
            exit(ret)
        print('done')

        print('--> Init retinaface_rknn runtime environment')
        ret = self.retinaface_rknn.init_runtime(target=self.target, device_id=self.device_id, rknn2precompile=False)
        if ret != 0:
            print('Init retinaface_rknn runtime environment failed')
            exit(ret)
        print('done')

        ret = self.mobilefacenet_rknn.load_rknn(self.mobilefacenet_rknn_model_path)
        if ret != 0:
            print('load mobilefacenet_rknn model failed.')
            exit(ret)
        print('done')

        print('--> Init mobilefacenet_rknn runtime environment')
        ret = self.mobilefacenet_rknn.init_runtime(target=self.target, device_id=self.device_id, rknn2precompile=False)
        if ret != 0:
            print('Init mobilefacenet_rknn runtime environment failed')
            exit(ret)
        print('done')

    #TODO : need to change result with true img size
    #现在是都缩放在640x640上实现的，后面需要映射到原始图片中
    def detect_one_image(self,image):

        # cv2.imshow("src", image)
        # cv2.waitKey()
        img = np.asarray(image)
        # img = np.array(img, np.uint8)

        print('--> inter')

        # ---------------------------------------------------#
        #   计算输入图片的高和宽
        # ---------------------------------------------------#
        resize = 1
        device = torch.device("cpu")
        im_height, im_width, _ = np.shape(image)
        scale = torch.Tensor([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
        scale = scale.to(device)

        # ---------------------------------------------------#
        #   图片预处理，归一化 在RKNN模型里实现，只需要输入RGB
        # ---------------------------------------------------#
        resize = 1
        # ---------------------------------------------------#
        #   将处理完的图片传入Retinaface网络当中进行预测
        # ---------------------------------------------------#
        loc, conf, landms = self.retinaface_rknn.inference(inputs=[img])

        img = image.transpose(2, 0, 1)
        new_shape = [1, img.shape[0], img.shape[1], img.shape[2]]
        img = img.reshape(new_shape)
        # ---------------------------------------------------#
        #   Retinaface网络的解码，最终我们会获得预测框
        #   将预测结果进行解码和非极大抑制
        # ---------------------------------------------------#
        loc = torch.tensor(loc)
        conf = torch.tensor(conf)
        landms = torch.tensor(landms)
        loc = loc.view(loc.shape[1], -1, 4)
        conf = conf.view(conf.shape[1], -1, 2)
        landms = landms.view(landms.shape[1], -1, 10)
        cfg_mnet = {
            'min_sizes': [[16, 32], [64, 128], [256, 512]],
            'steps': [8, 16, 32],
            'variance': [0.1, 0.2],
            'clip': False,
            'loc_weight': 2.0,
            'image_size': 640,
        }
        cfg = cfg_mnet
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
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
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        # anchor 最大5000个
        top_k = 5000
        order = scores.argsort()[::-1][:top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS 最大分析750个
        keep_top_k = 750
        dets = dets[:keep_top_k, :]
        landms = landms[:keep_top_k, :]
        # x1,y1,x2,y2,score,landm x1,y1,x2,y2...x5,y5
        results = np.concatenate((dets, landms), axis=1)
        return results

    def encode_face_dataset(self, image_paths, names):
        face_encodings = []
        for index, path in enumerate(tqdm(image_paths)):
            # ---------------------------------------------------#
            #   打开人脸图片
            # ---------------------------------------------------#
            image = np.array(Image.open(path).convert('RGB'), np.uint8)  # if not use .convert(‘RGB’) it will be RGBA
            # image = cv2.resize(image, (self.retinaface_input_shape[1], self.retinaface_input_shape[0]))
            image = letterbox_image(image, (self.retinaface_input_shape[1], self.retinaface_input_shape[0]))

            # ---------------------------------------------------#
            #   对输入图像进行一个备份
            # ---------------------------------------------------#
            old_image = image.copy()  # old_image is rgb
            # ---------------------------------------------------#
            #   计算输入图片的高和宽
            # ---------------------------------------------------#
            im_height, im_width, _ = np.shape(image)
            # ---------------------------------------------------#
            #   计算scale，用于将获得的预测框转换成原图的高宽
            # ---------------------------------------------------#
            scale = [
                np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]
            ]
            scale_for_landmarks = [
                np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
                np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
                np.shape(image)[1], np.shape(image)[0]
            ]

            # image = np.array(Image.open(path), np.float32)

            # image = cv2.imread(path)#这个打不开中文路径或者名字的图片
            if self.letterbox_image:
                image = letterbox_image(image, [self.retinaface_input_shape[1], self.retinaface_input_shape[0]])

            # image = cv2.resize(image, (self.retinaface_input_shape[0], self.retinaface_input_shape[1]))

            # image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)  # rknn need to be RGB in and handled means and std in rknn model


            results = self.detect_one_image(image)
            if len(results) <= 0:
                print(names[index], "：未检测到人脸")
                continue
            # ---------------------------------------------------#
            #   4人脸框置信度
            #   :4是框的坐标
            #   5:是人脸关键点的坐标
            # ---------------------------------------------------#
            # 将结果映射到原来图像大小
            # results[:, :4] = results[:, :4] / scale_in * scale
            # results[:, 5:] = results[:, 5:] / scale_for_landmarks_in * scale_for_landmarks
            # ---------------------------------------------------------#
            #   如果使用了letterbox_image的话，要把灰条的部分去除掉。
            # ---------------------------------------------------------#
            # if self.letterbox_image:
            #     results = retinaface_correct_boxes(results, np.array(
            #         (self.retinaface_input_shape[0], self.retinaface_input_shape[1])),
            #                                        np.array([im_height, im_width]))



            faces = 0
            # show image
            # ---------------------------------------------------#
            #   选取最大的人脸框。
            # ---------------------------------------------------#

            best_face_location = None
            biggest_area = 0
            if True:
                for b in results:
                    if b[4] < self.vis_thres:
                        continue
                    text = "{:.4f}".format(b[4])
                    faces += 1
                    b = list(map(int, b))
                    print(b)
                    if True:#测试显示结果
                        tmpImage = old_image.copy()
                        tmpImage = cv2.cvtColor(tmpImage,cv2.COLOR_RGB2BGR)
                        cv2.rectangle(tmpImage, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                        cx = b[0]
                        cy = b[1] + 12
                        cv2.putText(tmpImage, text, (cx, cy),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                        # landms
                        cv2.circle(tmpImage, (b[5], b[6]), 1, (0, 0, 255), 4)
                        cv2.circle(tmpImage, (b[7], b[8]), 1, (0, 255, 255), 4)
                        cv2.circle(tmpImage, (b[9], b[10]), 1, (255, 0, 255), 4)
                        cv2.circle(tmpImage, (b[11], b[12]), 1, (0, 255, 0), 4)
                        cv2.circle(tmpImage, (b[13], b[14]), 1, (255, 0, 0), 4)
                        # cv2.imshow("Detect", tmpImage)
                        # cv2.waitKey(0)

                    # ---------------------------------------------------#
                    #   选取最大的人脸框。
                    # ---------------------------------------------------#
                    left, top, right, bottom = b[0:4]

                    w = right - left
                    h = bottom - top
                    if w * h > biggest_area:
                        biggest_area = w * h
                        best_face_location = b


            if faces == 0:
                print(names[index], "：未检测到人脸")
                continue
            # results = np.array(results)


            # ---------------------------------------------------#
            #   截取图像 old_image (RGB)[h1:h2,w1:w2]=old_image[y1:y2,x1:x2]
            # ---------------------------------------------------#
            crop_img = old_image[int(best_face_location[1]):int(best_face_location[3]),
                       int(best_face_location[0]):int(best_face_location[2])]

            landmark = np.reshape(best_face_location[5:], (5, 2)) - np.array(
                [int(best_face_location[0]), int(best_face_location[1])])
            # crop_img, _ = Alignment_1(crop_img, landmark)
            if True:#self.letterbox_image:
                crop_img = np.array(
                    letterbox_image(np.uint8(crop_img), (self.facenet_input_shape[1], self.facenet_input_shape[0])),np.uint8)
            else:
                crop_img = cv2.resize(crop_img,(self.facenet_input_shape[1], self.facenet_input_shape[0]))
            # cv2.imshow("encFaceView", crop_img)
            # cv2.waitKey(0)
            # crop_img = np.expand_dims(crop_img, 0)
            # ---------------------------------------------------#
            #   利用图像算取长度为128的特征向量
            # ---------------------------------------------------#
            print(crop_img.shape)
            if self.facenet_l2_norm == False:
                face_encoding = self.mobilefacenet_rknn.inference(data_format='nhwc',inputs=[crop_img])[0][0]
                print(face_encoding)
                face_encodings.append(face_encoding)
            else:
                face_encoding = self.mobilefacenet_rknn.inference(data_format='nhwc', inputs=[crop_img])
                face_encoding = preprocessing.normalize(face_encoding[0], norm='l2')
                print(face_encoding)
                face_encodings.append(face_encoding[0])

        np.save("model_data/face_encoding.npy".format(), face_encodings)
        np.save("model_data/names.npy".format(), names)

    #   检测图片 输入image是RGB格式
    # ---------------------------------------------------#
    def detect_image(self, image):
        # ---------------------------------------------------#
        #   对输入图像进行一个备份，后面用于绘图
        # ---------------------------------------------------#
        # cv2.imshow("Src",image)

        # image = np.asarray(image, np.uint8)
        old_image   = image.copy()
        # ---------------------------------------------------#
        #   计算输入图片的高和宽
        # ---------------------------------------------------#
        im_height, im_width, _ = np.shape(image)
        # ---------------------------------------------------#
        #   计算scale，用于将获得的预测框转换成原图的高宽
        # ---------------------------------------------------#
        scale = [
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]
        ]
        scale_for_landmarks = [
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0]
        ]
        # ---------------------------------------------------#
        #   把图像转换成numpy的形式
        # ---------------------------------------------------#
        if self.letterbox_image:
            image = letterbox_image(image, [self.retinaface_input_shape[1], self.retinaface_input_shape[0]])

        # image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)  # rknn need to be RGB in and handled means and std in rknn model

        # ---------------------------------------------------#
        #   Retinaface检测部分-开始
        # ---------------------------------------------------#
        results = self.detect_one_image(image)
        # ---------------------------------------------------#
        #   4人脸框置信度
        #   :4是框的坐标
        #   5:是人脸关键点的坐标
        # ---------------------------------------------------#
        # 将结果映射到原来图像大小
        # results[:, :4] = results[:, :4] / scale_in * scale
        # results[:, 5:] = results[:, 5:] / scale_for_landmarks_in * scale_for_landmarks

        # ---------------------------------------------------#
        #   如果没有预测框则返回原图
        # ---------------------------------------------------#
        if len(results) <= 0:
            print("未检测到人脸")
            return old_image
        # ---------------------------------------------------------#
        #   如果使用了letterbox_image的话，要把灰条的部分去除掉。
        # ---------------------------------------------------------#
        # if self.letterbox_image:
        #     results = retinaface_correct_boxes(results, np.array(
        #         (self.retinaface_input_shape[0], self.retinaface_input_shape[1])),
        #                                        np.array([im_height, im_width]))

        # # ---------------------------------------------------#
        # #   4人脸框置信度
        # #   :4是框的坐标
        # #   5:是人脸关键点的坐标
        # # ---------------------------------------------------#
        # results[:, :4] = results[:, :4] * scale
        # results[:, 5:] = results[:, 5:] * scale_for_landmarks

        #results = np.array(results)

        # ---------------------------------------------------#
        #   Retinaface检测部分-结束
        # ---------------------------------------------------#

        # -----------------------------------------------#
        #   Facenet编码部分-开始
        # -----------------------------------------------#
        face_encodings = []
        idxCount = 0
        plt.figure()
        detectResult = []

        tmpImage = old_image.copy()
        for result in results:
            # ----------------------#
            #   图像截取，人脸矫正
            # ----------------------#
            if result[4] < self.vis_thres:
                continue
            detectResult.append(result)
            result = np.maximum(result, 0)

            crop_img = np.array(old_image)[int(result[1]):int(result[3]), int(result[0]):int(result[2])]
            landmark = np.reshape(result[5:], (5, 2)) - np.array([int(result[0]), int(result[1])])
            # crop_img, _ = Alignment_1(crop_img, landmark)
            # cv2.imwrite("out.jpg", crop_img)
            idxCount = idxCount + 1
            pltShowNp = np.array(crop_img, np.uint8(Image.BILINEAR))
            pltShowImg = Image.fromarray(pltShowNp)

            plt.subplot(len(results), len(results), idxCount)
            # pltShowImg = cv2.cvtColor(pltShowImg, cv2.COLOR_BGR2RGB)  # 改变显示的颜色
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(pltShowImg)

            text = "{:.4f}".format(result[4])
            b = list(map(int, result))
            print(b)
            if False:  # 测试显示结果
                tmpImage = cv2.cvtColor(tmpImage, cv2.COLOR_RGB2BGR)
                cv2.rectangle(tmpImage, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(tmpImage, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                # landms
                cv2.circle(tmpImage, (b[5], b[6]), 1, (0, 0, 255), 4)
                cv2.circle(tmpImage, (b[7], b[8]), 1, (0, 255, 255), 4)
                cv2.circle(tmpImage, (b[9], b[10]), 1, (255, 0, 255), 4)
                cv2.circle(tmpImage, (b[11], b[12]), 1, (0, 255, 0), 4)
                cv2.circle(tmpImage, (b[13], b[14]), 1, (255, 0, 0), 4)

            # ----------------------#
            #   人脸编码
            # ----------------------#

            # -----------------------------------------------#
            #   不失真的resize，然后进行归一化
            # -----------------------------------------------#
            crop_img = np.array(
                letterbox_image(np.uint8(crop_img), (self.facenet_input_shape[1], self.facenet_input_shape[0])),np.uint8)
            # crop_img = np.expand_dims(crop_img, 0)
            # cv2.imshow("faceEnc", crop_img)
            # cv2.waitKey(0)
            print(crop_img.shape)
            # -----------------------------------------------#
            #   利用图像算取长度为128的特征向量
            # -----------------------------------------------#
            if self.facenet_l2_norm == False:
                face_encoding = self.mobilefacenet_rknn.inference(data_format='nhwc', inputs=[crop_img])[0][0]
                print(face_encoding)
                face_encodings.append(face_encoding)
            else:
                face_encoding = self.mobilefacenet_rknn.inference(data_format='nhwc', inputs=[crop_img])
                face_encoding = preprocessing.normalize(face_encoding[0], norm='l2')
                print(face_encoding)
                face_encodings.append(face_encoding[0])
        # -----------------------------------------------#
        #   Facenet编码部分-结束
        # -----------------------------------------------#
        # plt.show()
        # cv2.imshow("faceDetect",tmpImage)
        # cv2.waitKey()
        # -----------------------------------------------#
        #   人脸特征比对-开始
        # -----------------------------------------------#
        face_names = []
        face_dist = []
        for face_encoding in face_encodings:
            # -----------------------------------------------------#
            #   取出一张脸并与数据库中所有的人脸进行对比，计算得分
            # -----------------------------------------------------#
            matches, face_distances = compare_faces(self.known_face_encodings, face_encoding,
                                                    tolerance=self.facenet_threhold)
            name = "Unknown"

            # -----------------------------------------------------#
            #   找到已知最贴近当前人脸的人脸序号
            # -----------------------------------------------------#
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
                face_dist.append(face_distances[best_match_index])
            else:
                face_dist.append(0)
            face_names.append(name)
        # -----------------------------------------------#
        #   人脸特征比对-结束
        # -----------------------------------------------#

        for i, b in enumerate(detectResult):
            text_retinaface = "{:.4f}".format(b[4])
            text = "{:.4f}".format(face_dist[i])
            b = list(map(int, b))
            # ---------------------------------------------------#
            #   b[0]-b[3]为人脸框的坐标，b[4]为得分
            # ---------------------------------------------------#

            old_image = cv2.cvtColor(np.asarray(old_image), cv2.COLOR_RGB2BGR)
            cv2.rectangle(old_image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(old_image, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
            cx = b[0]
            cy = b[1] + 32
            cv2.putText(old_image, text_retinaface, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            # ---------------------------------------------------#
            #   b[5]-b[14]为人脸关键点的坐标
            # ---------------------------------------------------#
            cv2.circle(old_image, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(old_image, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(old_image, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(old_image, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(old_image, (b[13], b[14]), 1, (255, 0, 0), 4)

            name = face_names[i]
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.putText(old_image, name, (b[0] , b[3] - 15), font, 0.75, (255, 255, 255), 2)
            # --------------------------------------------------------------#
            #   cv2不能写中文，加上这段可以，但是检测速度会有一定的下降。
            #   如果不是必须，可以换成cv2只显示英文。
            # --------------------------------------------------------------#
            old_image = cv2ImgAddText(old_image, name, b[0] + 5, b[3] - 25)
        return old_image


