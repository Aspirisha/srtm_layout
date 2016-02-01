from . import progress_ui
from PySide import QtGui, QtCore


class ProgressDialog(QtGui.QDialog):
    canceled = QtCore.Signal()
    paused = QtCore.Signal()

    def __init__(self, parent=None):
        super(ProgressDialog, self).__init__(parent)
        self.ui = progress_ui.Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.retranslateUi(self)
        print('here')

    def on_cancel(self):
        self.canceled.emit()

    def on_pause(self):
        self.paused.emit()

    def set_current_label_text(self, text):
        self.ui.current_label.setText(text)

    def set_overall_progress(self, progress):
        self.ui.overall_progress.setValue(progress)

    def set_current_progress(self, progress):
        self.ui.current_progress.setValue(progress)
