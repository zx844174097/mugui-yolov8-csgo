import threading

from ultralytics import YOLO
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon, QPixmap
# 配置本地代理
import os

from ui.window import Ui_MainWindow
from ultralytics import settings

os.environ['http_proxy'] = 'http://127.0.0.1:12083'
os.environ['https_proxy'] = 'http://127.0.0.1:12083'


class YoloPose(object):
    img = 'C:\\Users\\mugui\\AppData\\Local\\Programs\\Python\\Python310\\Lib\\site-packages\\ultralytics\\assets\\bus.jpg'

    def __init__(self):
        # 初始化YOLO，加载模型
        # self.model: YOLO = YOLO('yolov8x-pose.pt')
        self.model: YOLO = YOLO('runs/detect/train/weights/best.pt')

    def onInitYOLO(self, ui):
        print(settings)
        ui.mainui.drawImgPath(self.img, ui)
        ui.mainui.initAction.setEnabled(False)
        # 推理模式
        self.model.eval()

    def onStartYOLO(self, ui):
        # 检测边框
        results = self.model(self.img)
        # 打印检测结果
        annotated_frame = results[0].plot()
        # 将图像数据转换为QImage格式
        height, width, channel = annotated_frame.shape
        bytes_per_line = 3 * width
        qimage = QtGui.QImage(annotated_frame.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
        # 将QImage转换为QPixmap
        pixmap = QtGui.QPixmap.fromImage(qimage)
        # 显示检测结果
        ui.mainui.drawImg(pixmap, ui)

    def onSelectImg(self, ui):
        # 选择图片
        self.img, _ = QtWidgets.QFileDialog.getOpenFileName(ui.window, '选择图片', '', 'Image files(*.jpg *.gif *.png)')
        # 显示图片
        ui.updateDrawImgPath(self.img)

    def onTraining(self, ui):
        self.model = YOLO('yolov8n.yaml')
        # 训练模型
        self.model.train(data='E:\\ai\\yolov8\\csgo3.v2i.yolov8\\data.yaml', epochs=3, resume=True, batch=4, lr0=0.01)

    def showUI(self):
        ui = Ui_MainWindow()
        ui.setupUi()

        ui.setInitYOLO(self.onInitYOLO)
        ui.setStartYOLO(self.onStartYOLO)
        ui.setSelectImg(self.onSelectImg)
        ui.setTraining(self.onTraining)
        # 显示窗体
        ui.show()


if __name__ == '__main__':
    YoloPose().showUI()
