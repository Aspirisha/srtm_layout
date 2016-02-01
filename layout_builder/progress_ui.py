# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'progress.ui'
#
# Created: Sun Jan 31 15:03:18 2016
#      by: pyside-uic 0.2.15 running on PySide 1.2.1
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(335, 183)
        Dialog.setModal(True)
        self.current_progress = QtGui.QProgressBar(Dialog)
        self.current_progress.setGeometry(QtCore.QRect(20, 40, 291, 23))
        self.current_progress.setProperty("value", 0)
        self.current_progress.setObjectName("current_progress")
        self.overall_progress = QtGui.QProgressBar(Dialog)
        self.overall_progress.setGeometry(QtCore.QRect(20, 100, 291, 23))
        self.overall_progress.setProperty("value", 0)
        self.overall_progress.setObjectName("overall_progress")
        self.pause = QtGui.QPushButton(Dialog)
        self.pause.setGeometry(QtCore.QRect(50, 140, 99, 27))
        self.pause.setObjectName("pause")
        self.cancel = QtGui.QPushButton(Dialog)
        self.cancel.setGeometry(QtCore.QRect(170, 140, 99, 27))
        self.cancel.setObjectName("cancel")
        self.current_label = QtGui.QLabel(Dialog)
        self.current_label.setGeometry(QtCore.QRect(20, 15, 281, 20))
        self.current_label.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.current_label.setObjectName("current_label")
        self.overall_label = QtGui.QLabel(Dialog)
        self.overall_label.setGeometry(QtCore.QRect(20, 75, 221, 20))
        self.overall_label.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.overall_label.setObjectName("overall_label")

        self.retranslateUi(Dialog)
        QtCore.QObject.connect(self.cancel, QtCore.SIGNAL("clicked()"), Dialog.on_cancel)
        QtCore.QObject.connect(self.pause, QtCore.SIGNAL("clicked()"), Dialog.on_pause)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QtGui.QApplication.translate("Dialog", "Importing SRTM DEM", None, QtGui.QApplication.UnicodeUTF8))
        self.pause.setText(QtGui.QApplication.translate("Dialog", "Pause", None, QtGui.QApplication.UnicodeUTF8))
        self.cancel.setText(QtGui.QApplication.translate("Dialog", "Cancel", None, QtGui.QApplication.UnicodeUTF8))
        self.current_label.setText(QtGui.QApplication.translate("Dialog", "TextLabel", None, QtGui.QApplication.UnicodeUTF8))
        self.overall_label.setText(QtGui.QApplication.translate("Dialog", "Overall progress:", None, QtGui.QApplication.UnicodeUTF8))

