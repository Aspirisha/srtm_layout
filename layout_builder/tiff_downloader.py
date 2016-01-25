from io import BytesIO
import urllib, zipfile
import os
from math import floor, ceil
from . import util


def _download_srtm_tiff(srtm_filename, tif_folder):
    """
    Download and unzip GeoTIFF file.
    """

    if not os.path.isdir(tif_folder):
        os.mkdir(tif_folder)

    full_file_name = os.path.join(tif_folder, srtm_filename)
    if os.path.isfile(full_file_name):
        return

    url = 'http://gis-lab.info/data/srtm-tif/%s.zip' % srtm_filename[:-4]
    print('downloading ' + url + '...')
    zobj = BytesIO()
    zobj.write(urllib.request.urlopen(url).read())
    z = zipfile.ZipFile(zobj)

    srtm_path = os.path.join(tif_folder, srtm_filename)
    out_file = open(srtm_path, 'wb')
    out_file.write(z.read(srtm_filename))

    z.close()
    out_file.close()

def get_srtm_filename(lat, lon):
    """
    Filename of GeoTIFF file containing data with given coordinates.
    """
    colmin = floor((6000 * (180 + lon)) / 5)
    rowmin = floor((6000 * (60 - lat)) / 5)

    ilon = ceil(colmin / 6000.0)
    ilat = ceil(rowmin / 6000.0)

    return 'srtm_%02d_%02d.tif' % (ilon, ilat)

def download_srtm_tiffs(min_lat, min_lon, max_lat, max_lon, tif_folder='~/.geotiffs'):
    step = 0.01
    tiff_names = set()
    expanded_folder = os.path.expanduser(tif_folder)
    for lat in util.frange(min_lat, max_lat, step):
        for lon in util.frange(min_lon, max_lon, step):
            tiff_names.add(get_srtm_filename(lat, lon))
    for name in tiff_names:
        _download_srtm_tiff(name, expanded_folder)

    print('tif names: ', end='')
    print(tiff_names)
    full_tif_names = set([os.path.join(expanded_folder, name) for name in tiff_names])
    return full_tif_names
