import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QFileDialog
from Exp3_VAE_utils import *

from Exp3_VAE_utils import *

MODEL_PATH = "models/VAE.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {
    'input_size': 256,
    'first_ch': 32,
    'latent_channels': 40,
    'conv1': {'kernel_size': 4, 'stride': 2, 'padding': 1}, # 256 -> 128
    'conv2': {'kernel_size': 2, 'stride': 2, 'padding': 0}, # 128 -> 64
    'conv3': {'kernel_size': 4, 'stride': 2, 'padding': 1}, # 64 -> 32
    'conv4': {'kernel_size': 3, 'stride': 1, 'padding': 1}, # 32 -> 16
    'conv5': {'kernel_size': 4, 'stride': 2, 'padding': 1}, # 16 -> 8
    'deconv1': {'kernel_size': 4, 'stride': 2, 'padding': 1}, # 8 -> 16
    'deconv2': {'kernel_size': 3, 'stride': 1, 'padding': 1}, # 16 -> 32
    'deconv3': {'kernel_size': 4, 'stride': 2, 'padding': 1}, # 32 -> 64
    'deconv4': {'kernel_size': 4, 'stride': 2, 'padding': 1}, # 64 -> 128
    'deconv5': {'kernel_size': 2, 'stride': 2, 'padding': 0}, # 128 -> 256
    'out': {'kernel_size': 3, 'stride': 1, 'padding': 1}, # 256 -> 256
}

model = Exp3VariationalAutoEncoder(config)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE)['model_state_dict'])
model.to(DEVICE)


def generate_label(image_channels):
    # Stack and process the image channels
    image = np.stack(image_channels, axis=0)
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
    image = image.to(DEVICE)
    model.eval()
    with torch.no_grad():
        (q1, q2), _, _ = model(image)
    reject_mask = calculate_rejection_mask(image, q2, q2 - q1, threshold=0.05)[0].squeeze()
    return reject_mask


class ImageLoaderLabel(QtWidgets.QLabel):
    def __init__(self, parent=None):
        super(ImageLoaderLabel, self).__init__(parent)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setText("Click to load image")
        self.setStyleSheet("border: 2px dashed #aaa; background-color: #f0f0f0;")
        self.setScaledContents(True)
        self.image = None

    def loadImage(self):
        filePath, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if filePath:
            img = plt.imread(filePath)
            img = img[:, :, 0]
            self.image = img
            self.setPixmap(QtGui.QPixmap(filePath).scaled(self.width(), self.height()))


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(790, 800)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # Image loader labels with click-to-load functionality
        self.Image1 = ImageLoaderLabel(self.centralwidget)
        self.Image1.setGeometry(QtCore.QRect(10, 10, 250, 250))
        self.Image1.setObjectName("ImageT1")
        self.LabelT1 = QtWidgets.QLabel("T1", self.centralwidget)
        self.LabelT1.setGeometry(QtCore.QRect(10, 260, 250, 20))
        self.LabelT1.setAlignment(QtCore.Qt.AlignCenter)
        self.Image1.mousePressEvent = lambda event: self.Image1.loadImage()

        self.Image2 = ImageLoaderLabel(self.centralwidget)
        self.Image2.setGeometry(QtCore.QRect(270, 10, 250, 250))
        self.Image2.setObjectName("ImageT2")
        self.LabelT2 = QtWidgets.QLabel("T2", self.centralwidget)
        self.LabelT2.setGeometry(QtCore.QRect(270, 260, 250, 20))
        self.LabelT2.setAlignment(QtCore.Qt.AlignCenter)
        self.Image2.mousePressEvent = lambda event: self.Image2.loadImage()

        self.Image3 = ImageLoaderLabel(self.centralwidget)
        self.Image3.setGeometry(QtCore.QRect(530, 10, 250, 250))
        self.Image3.setObjectName("ImageFLAIR")
        self.LabelFLAIR = QtWidgets.QLabel("FLAIR", self.centralwidget)
        self.LabelFLAIR.setGeometry(QtCore.QRect(530, 260, 250, 20))
        self.LabelFLAIR.setAlignment(QtCore.Qt.AlignCenter)
        self.Image3.mousePressEvent = lambda event: self.Image3.loadImage()

        self.pushButton_1 = QtWidgets.QPushButton("Compute Rejection Mask", self.centralwidget)
        self.pushButton_1.setGeometry(QtCore.QRect(270, 300, 250, 50))
        self.pushButton_1.clicked.connect(self.compute_rejection_mask)

        self.MaskDisplay = QtWidgets.QLabel(self.centralwidget)
        self.MaskDisplay.setGeometry(QtCore.QRect(25, 380, 350, 350))
        self.MaskDisplay.setStyleSheet("border: 1px solid #aaa; background-color: #e0e0e0;")
        self.MaskDisplay.setText("Mask will be displayed here")
        self.MaskDisplay.setAlignment(QtCore.Qt.AlignCenter)
        self.MaskDisplay.setScaledContents(True)
        self.MaskDisplay.setObjectName("MaskDisplay")

        self.ground_truth = ImageLoaderLabel(self.centralwidget)
        self.ground_truth.setGeometry(QtCore.QRect(415, 380, 350, 350))
        self.ground_truth.setObjectName("ground_truth")
        self.ground_truth.setText("Ground Truth")
        self.ground_truth.setAlignment(QtCore.Qt.AlignCenter)
        self.ground_truth.mousePressEvent = lambda event: self.ground_truth.loadImage()

        MainWindow.setCentralWidget(self.centralwidget)

    def compute_rejection_mask(self):
        if self.Image1.image is not None and self.Image2.image is not None and self.Image3.image is not None:
            channels = [self.Image1.image, self.Image2.image, self.Image3.image]
            mask = generate_label(channels)
            mask_image = (mask * 255).astype(np.uint8)  # Convert to 8-bit image
            height, width = mask_image.shape
            bytesPerLine = 1 * width
            qImg = QtGui.QImage(mask_image.data, width, height, bytesPerLine, QtGui.QImage.Format_Grayscale8)
            self.MaskDisplay.setPixmap(QtGui.QPixmap.fromImage(qImg).scaled(width, height))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
