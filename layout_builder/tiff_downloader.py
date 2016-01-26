from io import BytesIO
import urllib, zipfile
import os
from math import floor, ceil
from . import util
from PySide import QtCore, QtGui


class TifDownloader(QtCore.QThread):
    update_progress = QtCore.Signal(int)

    def __init__(self, min_lat, min_lon, max_lat, max_lon, tif_folder='~/.geotiffs'):
        QtCore.QThread.__init__(self)
        self.tif_folder = os.path.expanduser(tif_folder)
        self.tif_names = self.get_tif_names(min_lat, min_lon, max_lat, max_lon)
        self.percent_per_tif = 100.0 / len(self.tif_names)
        self.downloaded_tifs = 0
        self.stopped = False

    def get_full_tif_names(self):
        full_tif_names = set([os.path.join(self.tif_folder, name) for name in self.tif_names])
        return full_tif_names

    def stop_running(self):
        self.stopped = True

    def continue_run(self):
        self.stopped = False

    def _download_srtm_tiff(self, srtm_filename):
        """
        Download and unzip GeoTIFF file.
        """

        if not os.path.isdir(self.tif_folder):
            os.mkdir(self.tif_folder)

        full_file_name = os.path.join(self.tif_folder, srtm_filename)
        if os.path.isfile(full_file_name):
            return

        url = 'http://gis-lab.info/data/srtm-tif/%s.zip' % srtm_filename[:-4]
        print('downloading ' + url + '...')
        zobj = BytesIO()
        response = urllib.request.urlopen(url)
        total_size = int(response.info().get('Content-Length'))
        print('total tif size: ' + str(total_size))

        block_size = 4096
        start_percent = self.downloaded_tifs * self.percent_per_tif
        percent_per_block = self.percent_per_tif / total_size
        bytes_read = 0
        while not self.stopped:
            data = response.read(block_size)
            if not data:
                break
            zobj.write(data)
            bytes_read += len(data)
            self.update_progress.emit(int(start_percent + percent_per_block * bytes_read))

        if self.stopped:
            return

        z = zipfile.ZipFile(zobj)

        srtm_path = os.path.join(self.tif_folder, srtm_filename)
        out_file = open(srtm_path, 'wb')
        out_file.write(z.read(srtm_filename))

        z.close()
        out_file.close()

    @staticmethod
    def get_srtm_filename(lat, lon):
        """
        Filename of GeoTIFF file containing data with given coordinates.
        """
        colmin = floor((6000 * (180 + lon)) / 5)
        rowmin = floor((6000 * (60 - lat)) / 5)

        ilon = ceil(colmin / 6000.0)
        ilat = ceil(rowmin / 6000.0)

        return 'srtm_%02d_%02d.tif' % (ilon, ilat)

    def run(self):
        for name in self.tif_names:
            self._download_srtm_tiff(name)
            self.downloaded_tifs += 1
            self.update_progress.emit(int(self.downloaded_tifs * self.percent_per_tif))

    @staticmethod
    def get_tif_names(min_lat, min_lon, max_lat, max_lon):
        step = 0.01
        tif_names = set()
        for lat in util.frange(min_lat, max_lat, step):
            for lon in util.frange(min_lon, max_lon, step):
                tif_names.add(TifDownloader.get_srtm_filename(lat, lon))

        return tif_names