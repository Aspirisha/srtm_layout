from io import BytesIO
import urllib, zipfile
import os
from math import floor, ceil
from . import util
from PySide import QtCore, QtGui

class HGTDownloader(QtCore.QThread):
    update_progress = QtCore.Signal(int)

    def __init__(self, min_lat, min_lon, max_lat, max_lon, elevation_data):
        QtCore.QThread.__init__(self)
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.min_lon = min_lon
        self.max_lon = max_lon
        self.elevation_data = elevation_data
        self.hgt_names = self.get_hgt_names()
        self.percent_per_file = 100.0 / len(self.hgt_names)
        self.downloaded_files = 0
        self.stopped = False

    def stop_running(self):
        self.stopped = True

    def continue_run(self):
        self.stopped = False

    def run(self):
        for name in self.hgt_names:
            print('downloading ' + name + '...')
            self.elevation_data.retrieve_or_load_file_data(name)
            self.downloaded_files += 1
            self.update_progress.emit(int(self.downloaded_files * self.percent_per_file))

    def get_hgt_names(self):
        step = 0.01
        hgt_names = set()
        for lat in util.frange(self.min_lat, self.max_lat, step):
            for lon in util.frange(self.min_lon, self.max_lon, step):
                hgt_names.add(self.elevation_data.get_file_name(lat, lon))

        return hgt_names

