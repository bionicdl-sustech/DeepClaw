# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'demo-v1.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1117, 708)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("./gui/icons/logo.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Form.setWindowIcon(icon)
        self.widget = QtWidgets.QWidget(Form)
        self.widget.setGeometry(QtCore.QRect(0, 0, 301, 60))
        self.widget.setObjectName("widget")
        self.layoutWidget = QtWidgets.QWidget(self.widget)
        self.layoutWidget.setGeometry(QtCore.QRect(1, 1, 298, 62))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.logo_label = QtWidgets.QLabel(self.layoutWidget)
        self.logo_label.setMinimumSize(QtCore.QSize(60, 60))
        self.logo_label.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.logo_label.setText("")
        self.logo_label.setPixmap(QtGui.QPixmap("./gui/icons/logo.png"))
        self.logo_label.setScaledContents(True)
        self.logo_label.setObjectName("logo_label")
        self.horizontalLayout.addWidget(self.logo_label)
        self.title_label = QtWidgets.QLabel(self.layoutWidget)
        self.title_label.setMinimumSize(QtCore.QSize(230, 50))
        self.title_label.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.title_label.setObjectName("title_label")
        self.horizontalLayout.addWidget(self.title_label)
        self.widget_2 = QtWidgets.QWidget(Form)
        self.widget_2.setGeometry(QtCore.QRect(0, 70, 301, 71))
        self.widget_2.setObjectName("widget_2")
        self.objects_checkBox_2 = QtWidgets.QCheckBox(self.widget_2)
        self.objects_checkBox_2.setEnabled(False)
        self.objects_checkBox_2.setGeometry(QtCore.QRect(0, 24, 150, 25))
        self.objects_checkBox_2.setMinimumSize(QtCore.QSize(150, 25))
        self.objects_checkBox_2.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.objects_checkBox_2.setChecked(True)
        self.objects_checkBox_2.setObjectName("objects_checkBox_2")
        self.original_checkBox = QtWidgets.QCheckBox(self.widget_2)
        self.original_checkBox.setEnabled(False)
        self.original_checkBox.setGeometry(QtCore.QRect(0, 0, 150, 25))
        self.original_checkBox.setMinimumSize(QtCore.QSize(150, 25))
        self.original_checkBox.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.original_checkBox.setAutoFillBackground(False)
        self.original_checkBox.setChecked(True)
        self.original_checkBox.setObjectName("original_checkBox")
        self.checkBox_5 = QtWidgets.QCheckBox(self.widget_2)
        self.checkBox_5.setEnabled(False)
        self.checkBox_5.setGeometry(QtCore.QRect(0, 48, 150, 25))
        self.checkBox_5.setMinimumSize(QtCore.QSize(150, 25))
        self.checkBox_5.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.checkBox_5.setChecked(True)
        self.checkBox_5.setObjectName("checkBox_5")
        self.valuable_checkBox = QtWidgets.QCheckBox(self.widget_2)
        self.valuable_checkBox.setEnabled(False)
        self.valuable_checkBox.setGeometry(QtCore.QRect(150, 24, 150, 25))
        self.valuable_checkBox.setMinimumSize(QtCore.QSize(150, 25))
        self.valuable_checkBox.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.valuable_checkBox.setChecked(True)
        self.valuable_checkBox.setObjectName("valuable_checkBox")
        self.features_checkBox_2 = QtWidgets.QCheckBox(self.widget_2)
        self.features_checkBox_2.setEnabled(False)
        self.features_checkBox_2.setGeometry(QtCore.QRect(150, 0, 150, 25))
        self.features_checkBox_2.setMinimumSize(QtCore.QSize(150, 25))
        self.features_checkBox_2.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.features_checkBox_2.setChecked(True)
        self.features_checkBox_2.setObjectName("features_checkBox_2")
        self.widget_3 = QtWidgets.QWidget(Form)
        self.widget_3.setGeometry(QtCore.QRect(0, 150, 301, 181))
        self.widget_3.setObjectName("widget_3")
        self.features_label = QtWidgets.QLabel(self.widget_3)
        self.features_label.setGeometry(QtCore.QRect(0, 0, 300, 25))
        self.features_label.setMinimumSize(QtCore.QSize(300, 25))
        self.features_label.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.features_label.setObjectName("features_label")
        self.listView = QtWidgets.QListView(self.widget_3)
        self.listView.setGeometry(QtCore.QRect(0, 20, 301, 161))
        self.listView.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.listView.setObjectName("listView")
        self.widget_4 = QtWidgets.QWidget(Form)
        self.widget_4.setGeometry(QtCore.QRect(0, 330, 301, 161))
        self.widget_4.setObjectName("widget_4")
        self.level3_label = QtWidgets.QLabel(self.widget_4)
        self.level3_label.setGeometry(QtCore.QRect(0, 0, 300, 25))
        self.level3_label.setMinimumSize(QtCore.QSize(300, 25))
        self.level3_label.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.level3_label.setObjectName("level3_label")
        self.listView_2 = QtWidgets.QListView(self.widget_4)
        self.listView_2.setGeometry(QtCore.QRect(0, 20, 301, 141))
        self.listView_2.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.listView_2.setObjectName("listView_2")
        self.widget_5 = QtWidgets.QWidget(Form)
        self.widget_5.setGeometry(QtCore.QRect(310, 0, 801, 491))
        self.widget_5.setObjectName("widget_5")
        self.level1_label = QtWidgets.QLabel(self.widget_5)
        self.level1_label.setGeometry(QtCore.QRect(0, 10, 120, 25))
        self.level1_label.setMinimumSize(QtCore.QSize(120, 25))
        self.level1_label.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.level1_label.setObjectName("level1_label")
        self.rgb_displayer = QtWidgets.QLabel(self.widget_5)
        self.rgb_displayer.setGeometry(QtCore.QRect(0, 40, 800, 450))
        self.rgb_displayer.setMinimumSize(QtCore.QSize(640, 360))
        self.rgb_displayer.setObjectName("rgb_displayer")
        self.camera = QtWidgets.QPushButton(self.widget_5)
        self.camera.setCheckable(True)
        self.camera.setGeometry(QtCore.QRect(120, 10, 25, 25))
        self.camera.setStyleSheet("QPushButton{border:2px}")
        self.camera.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("./gui/icons/camera.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.camera.setIcon(icon1)
        self.camera.setIconSize(QtCore.QSize(25, 25))
        self.camera.setObjectName("camera")
        self.record = QtWidgets.QPushButton(self.widget_5)
        self.record.setCheckable(True)
        self.record.setGeometry(QtCore.QRect(150, 10, 25, 25))
        self.record.setStyleSheet("QPushButton{border:2px}")
        self.record.setText("")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("./gui/icons/record.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.record.setIcon(icon2)
        self.record.setIconSize(QtCore.QSize(25, 25))
        self.record.setObjectName("record")
        self.widget_6 = QtWidgets.QWidget(Form)
        self.widget_6.setGeometry(QtCore.QRect(0, 500, 301, 171))
        self.widget_6.setObjectName("widget_6")
        self.valuable_label = QtWidgets.QLabel(self.widget_6)
        self.valuable_label.setGeometry(QtCore.QRect(0, 0, 300, 25))
        self.valuable_label.setMinimumSize(QtCore.QSize(300, 25))
        self.valuable_label.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.valuable_label.setObjectName("valuable_label")
        self.listView_3 = QtWidgets.QListView(self.widget_6)
        self.listView_3.setGeometry(QtCore.QRect(0, 20, 301, 151))
        self.listView_3.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.listView_3.setObjectName("listView_3")
        self.widget_7 = QtWidgets.QWidget(Form)
        self.widget_7.setGeometry(QtCore.QRect(310, 500, 801, 171))
        self.widget_7.setObjectName("widget_7")
        self.actions_label = QtWidgets.QLabel(self.widget_7)
        self.actions_label.setGeometry(QtCore.QRect(0, 80, 320, 25))
        self.actions_label.setMinimumSize(QtCore.QSize(320, 25))
        self.actions_label.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.actions_label.setObjectName("actions_label")
        self.listView_4 = QtWidgets.QListView(self.widget_7)
        self.listView_4.setGeometry(QtCore.QRect(0, 20, 801, 61))
        self.listView_4.setObjectName("listView_4")
        self.listView_5 = QtWidgets.QListView(self.widget_7)
        self.listView_5.setGeometry(QtCore.QRect(0, 100, 801, 71))
        self.listView_5.setObjectName("listView_5")
        self.state_label = QtWidgets.QLabel(self.widget_7)
        self.state_label.setGeometry(QtCore.QRect(0, 0, 320, 25))
        self.state_label.setMinimumSize(QtCore.QSize(320, 25))
        self.state_label.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.state_label.setObjectName("state_label")
        self.widget_9 = QtWidgets.QWidget(Form)
        self.widget_9.setGeometry(QtCore.QRect(310, 680, 85, 25))
        self.widget_9.setObjectName("widget_9")
        self.play = QtWidgets.QPushButton(self.widget_9)
        self.play.setGeometry(QtCore.QRect(30, 0, 25, 25))
        self.play.setStyleSheet("QPushButton{border:2px}")
        self.play.setText("")
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap("./gui/icons/play.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.play.setIcon(icon3)
        self.play.setIconSize(QtCore.QSize(25, 25))
        self.play.setObjectName("play")
        self.rewind = QtWidgets.QPushButton(self.widget_9)
        self.rewind.setGeometry(QtCore.QRect(0, 0, 25, 25))
        self.rewind.setStyleSheet("QPushButton{border:2px}")
        self.rewind.setText("")
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap("./gui/icons/rewind.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.rewind.setIcon(icon4)
        self.rewind.setIconSize(QtCore.QSize(25, 25))
        self.rewind.setObjectName("rewind")
        self.forward = QtWidgets.QPushButton(self.widget_9)
        self.forward.setGeometry(QtCore.QRect(60, 0, 25, 25))
        self.forward.setStyleSheet("QPushButton{border:2px}")
        self.forward.setText("")
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap("./gui/icons/forward.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.forward.setIcon(icon5)
        self.forward.setIconSize(QtCore.QSize(25, 25))
        self.forward.setObjectName("forward")
        self.textBrowser = QtWidgets.QTextBrowser(Form)
        self.textBrowser.setGeometry(QtCore.QRect(40, 680, 261, 25))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.textBrowser.setFont(font)
        self.textBrowser.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textBrowser.setPlaceholderText("")
        self.textBrowser.setObjectName("textBrowser")
        self.open_folder = QtWidgets.QPushButton(Form)
        self.open_folder.setGeometry(QtCore.QRect(10, 680, 25, 25))
        self.open_folder.setStyleSheet("QPushButton{border:2px}")
        self.open_folder.setText("")
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap("./gui/icons/open_folder.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.open_folder.setIcon(icon6)
        self.open_folder.setIconSize(QtCore.QSize(25, 25))
        self.open_folder.setObjectName("open_folder")
        self.playingSlider = QtWidgets.QSlider(Form)
        self.playingSlider.setGeometry(QtCore.QRect(410, 680, 631, 25))
        self.playingSlider.setOrientation(QtCore.Qt.Horizontal)
        self.playingSlider.setObjectName("playingSlider")
        self.playingPercentage = QtWidgets.QLabel(Form)
        self.playingPercentage.setGeometry(QtCore.QRect(1050, 680, 51, 25))
        self.playingPercentage.setObjectName("playingPercentage")
        self.widget_5.raise_()
        self.widget.raise_()
        self.widget_2.raise_()
        self.widget_3.raise_()
        self.widget_4.raise_()
        self.widget_6.raise_()
        self.widget_7.raise_()
        self.widget_9.raise_()
        self.textBrowser.raise_()
        self.open_folder.raise_()
        self.playingSlider.raise_()
        self.playingPercentage.raise_()

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Data Collection Platfrom"))
        self.title_label.setText(_translate("Form", "<html><head/><body><p align=\"center\"><span style=\" font-size:12pt; font-weight:600; font-style:italic;\">Bionic Design &amp; Learning Lab<br/>DeepClaw 2.0: DCP</span></p></body></html>"))
        self.objects_checkBox_2.setText(_translate("Form", "6D Pose Info"))
        self.original_checkBox.setText(_translate("Form", "Raw Sensor Data"))
        self.checkBox_5.setText(_translate("Form", "State/Actions"))
        self.valuable_checkBox.setText(_translate("Form", "Meaningful Info"))
        self.features_checkBox_2.setText(_translate("Form", "2D Features"))
        self.features_label.setText(_translate("Form", "2D Features"))
        self.level3_label.setText(_translate("Form", "6D Pose Info"))
        self.level1_label.setText(_translate("Form", "Raw Sensor Data"))
        self.rgb_displayer.setText(_translate("Form", "<html><head/><body><p align=\"center\"><span style=\" font-size:20pt;\">Original Data Stream</span></p><p align=\"center\"><span style=\" font-size:12pt; font-style:italic;\">Please Open the Camera</span></p><p align=\"center\"><span style=\" font-size:12pt; font-style:italic;\">or Load a Data Package</span></p></body></html>"))
        self.valuable_label.setText(_translate("Form", "Meaningful Info"))
        self.actions_label.setText(_translate("Form", "Actions"))
        self.state_label.setText(_translate("Form", "States"))
        self.playingPercentage.setText(_translate("Form", "<html><head/><body><p align=\"center\"><span style=\" font-size:8pt;\">xxxx/xxxx</span></p></body></html>"))

