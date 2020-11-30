import io
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PIL.ImageQt import ImageQt
from PyQt5.QtCore import Qt
from PIL import Image as pImage
from PyQt5.QtCore import QBuffer

COLORS = [
    '#000000', '#ffffff',
]

BRUSH_SIZES = [
    2, 4, 8, 16, 32
]


class Canvas(QtWidgets.QLabel):

    def __init__(self):
        super().__init__()
        pixmap = QtGui.QPixmap(512, 512)
        self.setPixmap(pixmap)

        self.last_x, self.last_y = None, None
        self.pen_color = QtGui.QColor('#000000')
        self.brush_size = 4

    def set_pen_color(self, c):
        self.pen_color = QtGui.QColor(c)

    def set_brush_size(self, s):
        self.brush_size = s

    def mouseMoveEvent(self, e):
        if self.last_x is None:  # First event.
            self.last_x = e.x()
            self.last_y = e.y()
            return  # Ignore the first time.

        painter = QtGui.QPainter(self.pixmap())
        p = painter.pen()
        p.setWidth(self.brush_size)
        p.setColor(self.pen_color)
        painter.setPen(p)
        # painter.drawLine(self.last_x, self.last_y, e.x(), e.y())
        painter.drawLine(e.x(), self.last_y, e.x(), e.y())
        painter.drawLine(self.last_x, e.y(), e.x(), e.y())
        painter.end()
        self.update()

        # Update the origin for next time.
        self.last_x = e.x()
        self.last_y = e.y()

    def drawImage(self, img):
        painter = QtGui.QPainter(self.pixmap())
        painter.drawImage(QtCore.QPoint(), img)
        painter.end()
        self.update()

    def mouseReleaseEvent(self, e):
        self.last_x = None
        self.last_y = None

    def getPixMap(self):
        return self.pixmap()


class ImageDisplayer(QtWidgets.QLabel):

    def __init__(self, img):
        super().__init__()
        pixmap = QtGui.QPixmap(512, 512)
        self.setPixmap(pixmap)
        self.drawImage(img)

    def drawImage(self, img):
        painter = QtGui.QPainter(self.pixmap())
        painter.drawImage(QtCore.QPoint(), img)
        painter.end()
        self.update()


class QPaletteButton(QtWidgets.QPushButton):

    def __init__(self, color):
        super().__init__()
        self.setFixedSize(QtCore.QSize(24, 24))
        self.color = color
        self.setStyleSheet("background-color: %s;" % color)


class BrushSizeButton(QtWidgets.QPushButton):

    def __init__(self, size):
        super().__init__()
        self.setFixedSize(QtCore.QSize(64, 24))
        self.size = size
        self.setText(str(size) + " Brush")


class ShowOriginalImageButton(QtWidgets.QPushButton):

    def __init__(self):
        super().__init__()
        self.setText("TOGGLE ORIGINAL IMAGE OVERLAY")


class MainWindow(QtWidgets.QMainWindow):
    save_and_move_image = bool

    def __init__(self, p_img, o_img):
        super().__init__()

        self.save_and_move_image = False

        #
        # image display area
        #
        canvas_area = QtWidgets.QWidget()
        canvas_area_layout = QtWidgets.QVBoxLayout()
        canvas_area.setLayout(canvas_area_layout)
        canvas_area_layout.setContentsMargins(0, 0, 0, 0)

        # original image static display
        self.original_image_display = ImageDisplayer(o_img)
        canvas_area_layout.addWidget(self.original_image_display)

        # original image overlay display, hidden by default
        self.original_image_overlay = ImageDisplayer(o_img)
        self.original_image_overlay.hide()
        canvas_area_layout.addWidget(self.original_image_overlay)

        # prediction canvas display/editor
        self.canvas = Canvas()
        self.canvas.drawImage(p_img)
        canvas_area_layout.addWidget(self.canvas)

        #
        # tools area
        #
        tools = QtWidgets.QWidget()
        tools_layout = QtWidgets.QVBoxLayout()
        tools.setLayout(tools_layout)

        # show original image overlay button
        show_original_image_button = ShowOriginalImageButton()
        show_original_image_button.pressed.connect(lambda: self.swap_image_overlay())
        tools_layout.addWidget(show_original_image_button)

        # color selection palatte
        palette = QtWidgets.QHBoxLayout()
        self.add_palette_buttons(palette)
        tools_layout.addLayout(palette)

        # brush selection buttons
        brushes = QtWidgets.QHBoxLayout()
        self.add_bush_size_buttons(brushes)
        tools_layout.addLayout(brushes)

        #
        # save/quit buttons
        #
        action_buttons = QtWidgets.QWidget()
        action_buttons_layout = QtWidgets.QHBoxLayout()
        action_buttons.setLayout(action_buttons_layout)

        # save button
        save_button = QtWidgets.QPushButton()
        save_button.setText("SAVE PREDICTION AND CONTINUE")
        action_buttons_layout.addWidget(save_button)
        save_button.clicked.connect(self.save_button_press)

        close_button = QtWidgets.QPushButton()
        close_button.setText("CONTINUE WITHOUT SAVING")
        action_buttons_layout.addWidget(close_button)
        close_button.clicked.connect(lambda: self.close())

        #
        # main window layout
        #
        window = QtWidgets.QWidget()
        window_layout = QtWidgets.QVBoxLayout()
        window.setLayout(window_layout)

        window_layout.addWidget(canvas_area)
        window_layout.addWidget(tools)
        window_layout.addWidget(action_buttons)

        self.setCentralWidget(window)

    def save_button_press(self):
        self.save_and_move_image = True
        self.close()

    def swap_image_overlay(self):
        if not self.canvas.isHidden():
            self.canvas.hide()
            self.original_image_overlay.show()
        else:
            self.original_image_overlay.hide()
            self.canvas.show()

    def add_palette_buttons(self, layout):
        for c in COLORS:
            b = QPaletteButton(c)
            b.pressed.connect(lambda c=c: self.canvas.set_pen_color(c))
            layout.addWidget(b)

    def add_bush_size_buttons(self, layout):
        for s in BRUSH_SIZES:
            b = BrushSizeButton(s)
            b.pressed.connect(lambda s=s: self.canvas.set_brush_size(s))
            layout.addWidget(b)


def qPixmapToPILImage(qimg):
    buffer = QBuffer()
    buffer.open(QBuffer.ReadWrite)
    qimg.save(buffer, "PNG")
    pimg = pImage.open(io.BytesIO(buffer.data()))
    pimg = pImage.Image.convert(pimg, mode="L")
    return pimg


class UserImageFeedbackWindow:
    edited_image = None

    def __init__(self, prediction_img, original_img):
        p_img = prediction_img.tobytes("raw", "L")
        p_qimg = QtGui.QImage(p_img, 512, 512, QtGui.QImage.Format_Grayscale8)

        o_img = original_img.tobytes("raw", "L")
        o_qimg = QtGui.QImage(o_img, 512, 512, QtGui.QImage.Format_Grayscale8)

        app = QtWidgets.QApplication(sys.argv)
        self.window = MainWindow(p_qimg, o_qimg)
        self.window.show()
        app.exec_()

        # convert edited image to PIL image format
        self.edited_image = qPixmapToPILImage(self.window.canvas.getPixMap().toImage())

    def get_edited_image(self):
        return self.edited_image

    def should_move_image(self):
        return self.window.save_and_move_image
