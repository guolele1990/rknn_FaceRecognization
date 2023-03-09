import cv2
from PIL import Image
import numpy as np

def getCoordinate(img):
    rectangle = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度图
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # 二值化

    element3 = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))  # 设置膨胀和腐蚀操作
    dilation = cv2.dilate(binary, element3, iterations=1)  # 膨胀一次，让轮廓突出
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)  # 检测轮廓
    cv2.drawContours(img, contours, -1, (0, 0, 255), 3)  # 参数值为1， 给contours[1]绘制轮廓。 -1: 给所有的contours绘制轮廓
    cv2.imshow("img", img)
    cv2.waitKey()

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        rectangle.append((x, y, x + w, y + h))
    print(f'rectangle: {rectangle}')
    return rectangle


def savePic(rectangle):
    for i in range(len(rectangle)):
        imgPath = "D:\\PythonWork\\Contour\\Photos\\" + str(i + 1) + ".PNG"  # notes: 图片的扩展名要一致
        im = Image.open(defaultImgPath)
        im = im.crop(rectangle[i])  # 对图片进行切割 im.crop(top_x, top_y, bottom_x, bottom_y)
        im.save(imgPath)


if __name__ == '__main__':

    # 创建一个长度为 14 的 Python 列表 list_data
    list_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    # 使用 array 函数将 list_data 转换为形状为 (14) 的 NumPy 数组
    arr = np.array(list_data)
    print (arr.ndim)
    # 打印 arr 的形状，输出 (14,)
    print(arr.shape)
    exit(0)
    defaultImgPath = './t1.jpg'
    img = cv2.imread(defaultImgPath)
    img_crop = img[0:100,100:200]
    cv2.imshow("img", img_crop)
    cv2.waitKey()
