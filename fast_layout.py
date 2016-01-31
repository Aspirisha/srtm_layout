import PhotoScan as ps
import math
import srtm
import os, sys
from layout_builder.DelaunayVoronoi import computeDelaunayTriangulation
import numpy as np
from osgeo import gdal
import tempfile
import time
from layout_builder import util, hgt_downloader, gdal_merge, ProgressDialog
from PySide import QtCore, QtGui
import re
import struct
import ctypes

support_directory = os.path.dirname(os.path.realpath(__file__)) + os.sep + u'layout_builder'
support_directory = str(support_directory.encode(sys.getfilesystemencoding()), 'utf8')
gdal.UseExceptions()


def test():
    handler = util.SpecificFolderFileHandler(os.path.join(get_path_in_chunk(), '.srtm'))
    elevation_data = srtm.get_data(file_handler=handler)
    h = elevation_data.get_elevation(57.1, 95.1, approximate=True)
    print(h)
    return elevation_data


def get_translator(qtapp):
    settings = QtCore.QSettings()
    lang = settings.value('main/language')
    translator = QtCore.QTranslator(qtapp)

    trans_file = 'en_GB'
    if lang == 'ru':
        trans_file = 'ru_RU'

    translator.load(trans_file, os.path.join(support_directory, 'trans'))
    qtapp.installTranslator(translator)
    return translator


def get_path_in_chunk():
    chunk = ps.app.document.chunk
    d = os.path.splitext(ps.app.document.path)[0] + ".files"
    chunkpath = os.path.join(d, str(chunk.key))
    return chunkpath

def is_existing_project():
    return ps.app.document.path != ''

def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = (arr[:,0]**2 + arr[:,1]**2 + arr[:,2]**2) ** 0.5
    arr[:,0] /= lens
    arr[:,1] /= lens
    arr[:,2] /= lens
    return arr

def delta_vector_to_chunk(v1, v2):
    chunk = ps.app.document.chunk
    v1 = chunk.crs.unproject(v1)
    v2 = chunk.crs.unproject(v2)
    v1 = chunk.transform.matrix.inv().mulp(v1)
    v2 = chunk.transform.matrix.inv().mulp(v2)
    z = v2 - v1
    z.normalize()

    return z


def get_chunk_vectors(lat, lon):
    z = delta_vector_to_chunk(ps.Vector([lon, lat, 0]), ps.Vector([lon, lat, 1]))
    y = delta_vector_to_chunk(ps.Vector([lon, lat, 0]), ps.Vector([lon + 0.001, lat, 0]))
    x = delta_vector_to_chunk(ps.Vector([lon, lat, 0]), ps.Vector([lon, lat+0.001, 0]))
    return x,y,-z


def write_model_file(f, points, normals, faces):
    f.write('mtllib mymodel.mtl\nusemtl Solid\n')
    for p in points:
        f.write("v {:.6f} {:.6f} {:.6f}\n".format(p[0], p[1], p[2]))
    for n in normals:
        f.write("vn {:.6f} {:.6f} {:.6f}\n".format(n[0], n[1], n[2]))
    for face in faces:
        f.write("f {0}//{0} {1}//{1} {2}//{2}\n".format(face[1] + 1, face[0] + 1, face[2] + 1))


def build_mesh(output_file, min_latitude, max_latitude, min_longitude, max_longitude, lat_step, long_step, need_download=True):
    handler = util.SpecificFolderFileHandler(os.path.join(get_path_in_chunk(), '.srtm'))
    elevation_data = srtm.get_data(file_handler=handler)

    get_height = lambda x,y: elevation_data.get_elevation(x, y, approximate=True)

    print(get_height(min_latitude, min_longitude))
    if not need_download:
        chunk = ps.app.document.chunk
        if chunk.elevation is None:
            print("No elevation is provided for chunk. Downloading...")
        else:
            pass
            #get_height =


    mesh_points = np.array([[longitude, latitude, get_height(latitude, longitude) ]
        for latitude in util.frange(min_latitude, max_latitude, lat_step)
        for longitude in util.frange(min_longitude, max_longitude, long_step)])

    print('mesh points length is ' + str(mesh_points.shape))
    last_ok = 0
    for p in mesh_points:
        if p[2] is None:
            p[2] = last_ok
        else:
            last_ok = p[2]

    points = [ps.Vector(p) for p in mesh_points]
    faces = np.array(computeDelaunayTriangulation(points))
    norms = np.zeros(mesh_points.shape, dtype=mesh_points.dtype)

    tris = mesh_points[faces]
    # normals for all triangles
    n = np.cross( tris[::,1 ] - tris[::,0]  , tris[::,2 ] - tris[::,0] )
    normalize_v3(n)
    norms[ faces[:,0] ] += n
    norms[ faces[:,1] ] += n
    norms[ faces[:,2] ] += n
    normalize_v3(norms)

    with open(output_file, "w") as f:
        write_model_file(f, mesh_points, norms, faces)

    print("successfully built mesh")

def wgs_to_chunk(chunk, point):
    return chunk.transform.matrix.inv().mulp(chunk.crs.unproject(point))

def check_chunk(chunk):
    if chunk is None or len(chunk.cameras) == 0:
        print("empty chunk!")
        return False

    if chunk.crs is None:
        print("Initialize chunk coordinate system first")
        return False

    return True

def get_chunk_bounds(chunk):
    min_latitude = min(c.reference.location[1] for c in chunk.cameras)
    max_latitude = max(c.reference.location[1] for c in chunk.cameras)
    min_longitude =  min(c.reference.location[0] for c in chunk.cameras)
    max_longitude =  max(c.reference.location[0] for c in chunk.cameras)
    delta_latitude = max_latitude - min_latitude
    delta_longitude = max_longitude - min_longitude
    min_longitude -= delta_longitude
    max_longitude += delta_longitude
    min_latitude -= delta_latitude
    max_latitude += delta_latitude

    return min_latitude, min_longitude, max_latitude, max_longitude


def align_cameras(chunk, min_latitude, min_longitude, max_latitude, max_longitude):
    if chunk.transform.scale is None:
        chunk.transform.scale = 1
        chunk.transform.rotation = ps.Matrix([[1,0,0], [0,1,0], [0,0,1]])
        chunk.transform.translation = ps.Vector([0,0,0])

    delta_latitude_scale_to_meters = 40008000 / 360

    init_location = chunk.cameras[0].reference.location
    latitude = init_location[0]
    delta_longitude_scale_to_meters = 40075160 * math.cos(math.radians(latitude)) / 360

    delta_meters_scale_to_chunk = 0.1
    scales = [delta_latitude_scale_to_meters * delta_meters_scale_to_chunk,
        delta_longitude_scale_to_meters * delta_meters_scale_to_chunk, delta_meters_scale_to_chunk]

    i, j, k = get_chunk_vectors(min_latitude, min_longitude) # i || North
    for c in chunk.cameras:
        location = c.reference.location

        chunk_coordinates = wgs_to_chunk(chunk, location)
        c.transform = ps.Matrix([[i.x, j.x, k.x, chunk_coordinates[0]],
                                 [i.y, j.y, k.y, chunk_coordinates[1]],
                                 [i.z, j.z, k.z, chunk_coordinates[2]],
                                 [0, 0, 0, 1]])


def revert_changes(chunk):
    for c in chunk.cameras:
        c.transform = None


class DemImporter(QtCore.QObject):
    def __init__(self, parent=None):
        super(DemImporter, self).__init__(parent)
        qtapp = QtGui.QApplication.instance()
        self.translator = get_translator(qtapp)
        self.setObjectName("DemImporter")

    def import_dem(self):
        doc = ps.app.document
        chunk = doc.chunk

        if not check_chunk(chunk):
            return

        min_latitude, min_longitude, max_latitude, max_longitude = get_chunk_bounds(chunk)
        # TODO it makes sense to make tif cache common, so following is probably not to be used

        if not is_existing_project():
            print("Save project before importing dem!")
            return

        def download_canceled():
            downloader.stop_running()
            while not downloader.isFinished():
                time.sleep(0.1)

        progress_dialog = ProgressDialog.ProgressDialog()
        progress_dialog.canceled.connect(download_canceled)

        hgts_folder = os.path.join(get_path_in_chunk(), '.srtm')
        tif_folder = os.path.join(get_path_in_chunk(), 'geotifs')

        downloader = hgt_downloader.HGTDownloader(min_latitude, min_longitude,
                                                   max_latitude, max_longitude, hgts_folder)
        downloader.update_current_progress.connect(progress_dialog.set_current_progress)
        downloader.update_overall_progress.connect(progress_dialog.set_overall_progress)
        downloader.set_current_task_name.connect(progress_dialog.set_current_label_text)
        downloader.start()
        progress_dialog.show()

        while downloader.isRunning():
            QtGui.qApp.processEvents()
            time.sleep(0.1)

        if downloader.stopped:
            revert_changes(chunk)
            return

        chunk.importDem(downloader.merged_tif)
        '''
        full_tif_names = downloader.get_full_tif_names()

        merged_tif = os.path.join(tif_folder, 'tmp.tif')
        argv = ['gdal_merge.py', '-o', merged_tif]
        argv.extend(full_tif_names)

        if not os.path.isdir(tif_folder):
            os.mkdir(tif_folder)
        gdal_merge.main(argv)
        chunk.importDem(merged_tif)'''


        '''
        if is_existing_project():
            model_file = get_path_in_chunk() + os.sep + "mymodel.obj"
        else:
            with tempfile.NamedTemporaryFile(dir='/tmp', delete=False, suffix='.obj') as tmpfile:
                model_file = tmpfile.name
        build_mesh(model_file, min_latitude, max_latitude, min_longitude, max_longitude, lat_step=0.0005, long_step=0.0005)

        chunk.importModel(model_file)

        # delete temp file
        if not is_existing_project():
            os.remove(model_file)
        '''


def run_import():
    importer = DemImporter()
    importer.import_dem()


def run_camera_alignment():
    doc = ps.app.document
    chunk = doc.chunk

    if not check_chunk(chunk):
        return

    min_latitude, min_longitude, max_latitude, max_longitude = get_chunk_bounds(chunk)
    align_cameras(chunk, min_latitude, min_longitude, max_latitude, max_longitude)


translator = get_translator(QtGui.QApplication.instance())
ps.app.addMenuItem(translator.translate(
        'dlg', "Tools/Import/Import SRTM DEM..."), run_import)
ps.app.addMenuItem(translator.translate(
    'dlg', "Workflow/Apply Vertical Camera Alignment..."), run_camera_alignment)
