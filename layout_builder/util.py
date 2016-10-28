import fast_layout.srtm as srtm
import os
import re
import math
import struct
import gdal
import ctypes
import site

def frange(x, y, jump):
    while x < y:
        yield x
        x += jump


class Enum(set):
    def __getattr__(self, name):
        if name in self:
            return name
        raise AttributeError


class SpecificFolderFileHandler(srtm.FileHandler):
    def __init__(self, cache_folder=None):
        self.cache_folder = cache_folder
        print(cache_folder)
        if cache_folder is not None:
            if not os.path.exists(cache_folder):
                os.makedirs(cache_folder)

    def get_srtm_dir(self):
        if self.cache_folder is None:
            return super.get_srtm_dir()
        return self.cache_folder


# @return full name of result hgt file
def apply_egm_offset(hgt_file):
    offset_dir = site.getsitepackages()[0].split('python', 1)[0]

    egm_tif_file = os.path.join(offset_dir, 'geoids', 'egm96-15.tif')
    try:
        tif = gdal.Open(egm_tif_file)
    except:
        print('Couldn\'t find "egm96-15.tif" in {}'.format(egm_tif_file))
        return None

    band1 = tif.GetRasterBand(1).ReadAsArray()

    file_size = os.path.getsize(hgt_file)
    cols = rows = int(math.sqrt(file_size / 2))
    step = 1.0 / (cols - 1) # 1 / 1200 for the world, 1 / 3600 for US

    file_name = os.path.basename(hgt_file)
    m = re.match('([NS])(\d+)([WE])(\d+)', file_name)

    # so that 90N will be 0, and 90S will be 180
    lat_sign = -1 if m.group(1) == 'N' else 1
    lon_sign = -1 if m.group(3) == 'W' else 1
    start_lat = 90 + lat_sign * int(m.group(2))
    start_lon = 180 + lon_sign * int(m.group(4))

    # output file name will be like N53E069_wgs84.hgt
    output_path = os.path.dirname(hgt_file)
    output_name = str(file_name.split('.')[0]) + u'_wgs84.hgt'
    hgt_wgs_file = os.path.join(output_path, output_name)

    processed_files = os.path.join(output_path, 'processed.txt')
    if os.path.exists(processed_files):
        with open(processed_files, 'r') as f:
            for line in f:
                if line.strip() == hgt_file:
                    return

    data = ctypes.create_string_buffer(2 * rows * cols)
    offset = 0

    with open(hgt_file, "rb") as in_file:
        for r in range(rows):
            for c in range(cols):
                lon = start_lon + lon_sign * r * step
                lat = start_lat + lat_sign * c * step
                buf = in_file.read(2)

                val = struct.unpack('>h', buf)[0]
                # find corresponing pixel in offsets map
                offset_map_col = int(720 / 180 * lat)
                offset_map_row = int(1440 / 360 * lon)
                offset_sample = band1[offset_map_col][offset_map_row]

                if val != -32768:
                    val += offset_sample
                struct.pack_into(">h", data, offset, int(val))
                offset += 2

    with open(hgt_file, "wb") as out_file:
        out_file.write(data.raw)

    with open(processed_files, "a") as f:
        f.write(hgt_file + '\n')
