from io import BytesIO
import urllib, zipfile
import os
from math import floor, ceil
from . import util
from PySide import QtCore, QtGui
import srtm
import PhotoScan as ps

class HGTDownloader(QtCore.QThread):
    update_current_progress = QtCore.Signal(int)
    update_overall_progress = QtCore.Signal(int)
    set_current_task_name = QtCore.Signal(str)

    def __init__(self, min_lat, min_lon, max_lat, max_lon, hgts_folder):
        QtCore.QThread.__init__(self)
        self.hgts_folder = hgts_folder
        handler = util.SpecificFolderFileHandler(hgts_folder)
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.min_lon = min_lon
        self.max_lon = max_lon
        self.elevation_data = srtm.get_data(file_handler=handler)
        self.hgt_names = self.get_hgt_names()
        self.percent_per_file = 100.0 / len(self.hgt_names)
        self.stopped = False
        self.merged_tif = os.path.join(self.hgts_folder, 'result.tif')
        self.full_hgt_names = ''

    def stop_running(self):
        self.stopped = True

    def continue_run(self):
        self.stopped = False

    def download_files(self):
        downloaded_files = 0
        for name in self.hgt_names:
            print('downloading ' + name + '...')
            self.elevation_data.retrieve_or_load_file_data(name)
            downloaded_files += 1
            self.update_current_progress.emit(int(downloaded_files * self.percent_per_file))

    def apply_offset_to_files(self):
        processed_files = 0
        for hgt_file in self.hgt_names:
            print('applying offset to ' + hgt_file + '...')
            full_hgt_name = os.path.join(self.hgts_folder, hgt_file)
            util.apply_egm_offset(full_hgt_name)
            self.full_hgt_names += full_hgt_name + ' '
            self.update_current_progress.emit(int(processed_files * self.percent_per_file))

    def merge_hgts_to_tiff(self):
        clip_bounds = ' -te {} {} {} {} '.format(
                self.min_lon, self.min_lat, self.max_lon, self.max_lat)

        gdal_command = 'gdalwarp ' + clip_bounds + self.full_hgt_names + self.merged_tif
        print('using gdal to produce tiff. Gdal commands is "' + gdal_command + '"')
        os.system(gdal_command) # + ' "+proj=longlat +ellps=WGS84"'

    def run(self):
        self.set_current_task_name.emit("Downloading .hgt files...")
        self.download_files()

        self.update_overall_progress.emit(33)
        self.update_current_progress.emit(0)
        self.set_current_task_name.emit("Converting .hgt files from EGM to WGS-84")
        self.apply_offset_to_files()

        self.update_overall_progress.emit(66)
        self.update_current_progress.emit(0)
        self.set_current_task_name.emit("Merging .hgt files into tif")
        self.merge_hgts_to_tiff()

    def get_hgt_names(self):
        step = 0.01
        hgt_names = set()
        for lat in util.frange(self.min_lat, self.max_lat, step):
            for lon in util.frange(self.min_lon, self.max_lon, step):
                hgt_names.add(self.elevation_data.get_file_name(lat, lon))

        return hgt_names

