import PhotoScan as ps
import math
import srtm
import os, sys
from layout_builder.DelaunayVoronoi import computeDelaunayTriangulation
import numpy as np
from osgeo import gdal
import time
from layout_builder import util, hgt_downloader, gdal_merge, ProgressDialog
from PySide import QtCore, QtGui

support_directory = os.path.dirname(os.path.realpath(__file__)) + os.sep + u'layout_builder'
support_directory = str(support_directory.encode(sys.getfilesystemencoding()), 'utf8')
gdal.UseExceptions()


def time_measure(func):
    def wrapper(*args, **kwargs):
        t1 = time.time()
        res = func(*args, **kwargs)
        t2 = time.time()
        print("Finished processing in {} sec.".format(t2 - t1))
        return res
    return wrapper


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

def get_hgts_folder():
    return os.path.join(get_path_in_chunk(), '.srtm')


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


def build_mesh(output_file, min_latitude, max_latitude, min_longitude, max_longitude, lat_step, long_step):
    hgts_folder = get_hgts_folder()
    downloader_tasks = ['convert']
    downloader = hgt_downloader.HGTDownloader(min_latitude, min_longitude,
                                              max_latitude, max_longitude, hgts_folder,
                                              downloader_tasks)
    if not run_downloader(downloader):
        return

    def get_height(x, y):
        return downloader.elevation_data.get_elevation(x, y, approximate=True)

    mesh_points = np.array([[longitude, latitude, get_height(latitude, longitude) ]
        for latitude in util.frange(min_latitude, max_latitude, lat_step)
        for longitude in util.frange(min_longitude, max_longitude, long_step)])
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
    print("Successfully built mesh")

    ps.app.document.chunk.importModel(output_file)
    print("Mesh imported")


def wgs_to_chunk(chunk, point):
    return chunk.transform.matrix.inv().mulp(chunk.crs.unproject(point))


def show_message(msg):
    translator = get_translator(QtGui.QApplication.instance())
    translated_msg = translator.translate('dlg', msg)
    msgBox = QtGui.QMessageBox()
    print(msg)
    msgBox.setText(translated_msg)
    msgBox.exec()


def check_chunk(chunk):
    if chunk is None or len(chunk.cameras) == 0:
        show_message("Empty chunk!")
        return False

    if chunk.crs is None:
        show_message("Initialize chunk coordinate system first")
        return False

    return True


def request_integer(label, default_value):
    translator = get_translator(QtGui.QApplication.instance())
    translated_label = translator.translate('dlg', label)
    return ps.app.getInt(label=translated_label, value=default_value)


def get_photos_delta(chunk):
    mid_idx = int(len(chunk.cameras) / 2)
    if mid_idx == 0:
        return ps.Vector([0,0,0])
    c1 = chunk.cameras[:mid_idx][-1]
    c2 = chunk.cameras[:mid_idx][-2]
    offset = c1.reference.location - c2.reference.location
    for i in range(len(offset)):
        offset[i] = math.fabs(offset[i])
    return offset


def get_chunk_bounds(chunk):
    min_latitude = min(c.reference.location[1] for c in chunk.cameras if c.reference.location is not None)
    max_latitude = max(c.reference.location[1] for c in chunk.cameras if c.reference.location is not None)
    min_longitude = min(c.reference.location[0] for c in chunk.cameras if c.reference.location is not None)
    max_longitude = max(c.reference.location[0] for c in chunk.cameras if c.reference.location is not None)
    offset = get_photos_delta(chunk)
    offset_factor = 2
    delta_latitude = offset_factor * offset.y
    delta_longitude = offset_factor * offset.x

    min_longitude -= delta_longitude
    max_longitude += delta_longitude
    min_latitude -= delta_latitude
    max_latitude += delta_latitude

    return min_latitude, min_longitude, max_latitude, max_longitude


def get_xy_distance(v1, v2):
    dv = v1 - v2
    dv.z = 0
    return dv.norm()


def get_camera_calibration(chunk, min_latitude, min_longitude, same_yaw_bound):
    request = "Insert number of photos to estimate camera calibration"
    cameras_number_for_align = request_integer(request, 10)
    for c in chunk.cameras:
        c.enabled = False
        c.transform = None

    yaws_deltas_per_group, first_class_yaw_per_group = [], []
    for group in chunk.camera_groups:
        central_camera_and_max_dist = (None, None)
        different_cameras = []
        for c in chunk.cameras:
            if c.group != group:
                continue
            different_cameras.append([c, None])
            max_dist = 0
            if central_camera_and_max_dist == (None, None):
                central_camera_and_max_dist = (c, 1e300)
            for other in chunk.cameras:
                if other.group != group:
                    continue
                dist = get_xy_distance(other.reference.location, c.reference.location)
                if dist > max_dist:
                    max_dist = dist

            if max_dist < central_camera_and_max_dist[1]:
                central_camera_and_max_dist = (c, max_dist)

        central_camera_location = central_camera_and_max_dist[0].reference.location
        for cam_dist in different_cameras:
            cam_dist[1] = get_xy_distance(cam_dist[0].reference.location, central_camera_location)

        different_cameras.sort(key=lambda x: x[1])
        print('central camera name is {}'.format(central_camera_and_max_dist[0].label))
        for cam_dist in different_cameras[:cameras_number_for_align]:
            cam_dist[0].enabled = True

        chunk.matchPhotos(preselection=ps.Preselection.ReferencePreselection)
        chunk.alignCameras()
        yaws_deltas, first_class_yaw = estimate_wind_angle(chunk, min_latitude, min_longitude, same_yaw_bound)
        yaws_deltas_per_group.append(yaws_deltas)
        first_class_yaw_per_group.append(first_class_yaw)

        for cam_dist in different_cameras[:cameras_number_for_align]:
            cam_dist[0].enabled = False

    for c in chunk.cameras:
        c.enabled = True
    return yaws_deltas_per_group, first_class_yaw_per_group


def extend_3d_vector(v):
    return ps.Vector([v.x, v.y, v.z, 0])


def estimate_wind_angle(chunk, min_latitude, min_longitude, same_yaw_bound=40):
    i, j, k = get_chunk_vectors(min_latitude, min_longitude) # i || North
    i, j = extend_3d_vector(i), extend_3d_vector(j)
    yaws_deltas = [0, 0]
    first_class_yaw = None

    class_sizes = [0, 0]
    for c in chunk.cameras:
        if not c.enabled or c.transform is None:
            continue
        if first_class_yaw is None:
            first_class_yaw = c.reference.rotation.x

        fi_no_wind = c.reference.rotation.x + 90
        best_delta_fi = 0
        min_norm = 1e300
        for delta_fi in util.frange(-30, 30, 0.1):
            fi = fi_no_wind + delta_fi
            fi_rad = math.radians(fi)
            ii_w, jj_w = i * math.cos(fi_rad) + j * math.sin(fi_rad), j * math.cos(fi_rad) - i * math.sin(fi_rad)
            norm = (c.transform.col(0) - ii_w) * (c.transform.col(0) - ii_w) + \
                   (c.transform.col(1) - jj_w) * (c.transform.col(1) - jj_w)
            if norm < min_norm:
                min_norm = norm
                best_delta_fi = delta_fi
        idx = 0 if math.fabs(first_class_yaw - c.reference.rotation.x) < same_yaw_bound else 1
        yaws_deltas[idx] += best_delta_fi
        #print('{} : {}'.format(c.label, best_delta_fi))
        class_sizes[idx] += 1

    for i in range(2):
        if class_sizes[i] > 0:
            yaws_deltas[i] /= class_sizes[i]
    return yaws_deltas, first_class_yaw


@time_measure
def align_cameras(chunk, min_latitude, min_longitude):
    if chunk.transform.scale is None:
        chunk.transform.scale = 1
        chunk.transform.rotation = ps.Matrix([[1,0,0], [0,1,0], [0,0,1]])
        chunk.transform.translation = ps.Vector([0,0,0])

    for c in chunk.cameras:
        if c.group is None:
            show_message("All cameras should belong to some group.")
            return

    same_yaw_bound = 40 # within this bound all yaws are considered to be for same direction flights
    yaws_deltas, first_class_yaw = get_camera_calibration(chunk, min_latitude, min_longitude, same_yaw_bound=40)

    print(yaws_deltas)

    positive_dir = chunk.cameras[1].reference.location - chunk.cameras[0].reference.location
    positive_dir.z = 0
    positive_dir.normalize()
    i, j, k = get_chunk_vectors(min_latitude, min_longitude) # i || North

    for c in chunk.cameras:
        group_index = chunk.camera_groups.index(c.group)

        location = c.reference.location
        chunk_coordinates = wgs_to_chunk(chunk, location)
        fi = c.reference.rotation.x + 90
        idx = 0 if math.fabs(c.reference.rotation.x - first_class_yaw[group_index]) < same_yaw_bound else 1
        fi += yaws_deltas[group_index][idx]
        fi = math.radians(fi)

        ii, jj = i * math.cos(fi) + j * math.sin(fi), j * math.cos(fi) - i * math.sin(fi)
        c.transform = ps.Matrix([[ii.x, jj.x, k.x, chunk_coordinates[0]],
                                 [ii.y, jj.y, k.y, chunk_coordinates[1]],
                                 [ii.z, jj.z, k.z, chunk_coordinates[2]],
                                 [0, 0, 0, 1]])


def revert_changes(chunk):
    for c in chunk.cameras:
        c.transform = None


def run_downloader(downloader):
    def download_canceled():
        downloader.stop_running()
        downloader.terminate()
        while not downloader.isFinished():
            time.sleep(0.1)

    def download_paused():
        downloader.set_paused(not downloader.paused)

    progress_dialog = ProgressDialog.ProgressDialog()
    progress_dialog.canceled.connect(download_canceled)
    progress_dialog.paused.connect(download_paused)

    downloader.update_current_progress.connect(progress_dialog.set_current_progress)
    downloader.update_overall_progress.connect(progress_dialog.set_overall_progress)
    downloader.set_current_task_name.connect(progress_dialog.set_current_label_text)

    downloader.start()
    progress_dialog.show()

    while downloader.isRunning():
        QtGui.qApp.processEvents()
        time.sleep(0.1)

    return not downloader.stopped


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

        hgts_folder = os.path.join(get_path_in_chunk(), '.srtm')

        downloader_tasks = ['convert', 'merge']
        downloader = hgt_downloader.HGTDownloader(min_latitude, min_longitude,
                                                  max_latitude, max_longitude, hgts_folder,
                                                  downloader_tasks)

        if not run_downloader(downloader):
            revert_changes(chunk)
            return
        chunk.importDem(downloader.merged_tif)


def run_import():
    importer = DemImporter()
    importer.import_dem()


def run_camera_alignment():
    doc = ps.app.document
    chunk = doc.chunk

    if not check_chunk(chunk):
        return

    min_latitude, min_longitude, max_latitude, max_longitude = get_chunk_bounds(chunk)
    align_cameras(chunk, min_latitude, min_longitude)


def import_srtm_mesh():
    doc = ps.app.document
    chunk = doc.chunk

    if not check_chunk(chunk):
        return

    min_latitude, min_longitude, max_latitude, max_longitude = get_chunk_bounds(chunk)
    mesh_file = os.path.join(get_path_in_chunk(), '.srtm', 'model.obj')
    build_mesh(mesh_file, min_latitude, max_latitude, min_longitude, max_longitude, 0.001, 0.001)


translator = get_translator(QtGui.QApplication.instance())
ps.app.addMenuItem(translator.translate(
        'dlg', "Tools/Import/Import SRTM DEM..."), run_import)
ps.app.addMenuItem(translator.translate(
    'dlg', "Workflow/Apply Vertical Camera Alignment..."), run_camera_alignment)
ps.app.addMenuItem(translator.translate(
        'dlg', "Tools/Import/Import SRTM mesh..."), import_srtm_mesh)
