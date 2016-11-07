import PhotoScan as ps
import math
import os, sys
import numpy as np
import time
from PySide import QtCore, QtGui
import copy

support_directory = os.path.dirname(os.path.realpath(__file__)) + os.sep + u'layout_builder'
support_directory = str(support_directory.encode(sys.getfilesystemencoding()), 'utf8')


def time_measure(func):
    def wrapper(*args, **kwargs):
        t1 = time.time()
        res = func(*args, **kwargs)
        t2 = time.time()
        print("Finished processing in {} sec.".format(t2 - t1))
        return res
    return wrapper


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

# returns distance estimation between two cameras in chunk 
def get_photos_delta(chunk):
    mid_idx = int(len(chunk.cameras) / 2)
    if mid_idx == 0:
        return ps.Vector([0,0,0])
    c1 = chunk.cameras[:mid_idx][-1]
    c2 = chunk.cameras[:mid_idx][-2]
    print(c1.reference.location)
    print(c2.reference.location)
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

# returns horizontal distance between two vectors
def get_xy_distance(v1, v2):
    if v1 is None or v2 is None:
        return 1e300
    dv = v1 - v2
    dv.z = 0
    return dv.norm()

def extend_3d_vector(v):
    return ps.Vector([v.x, v.y, v.z, 0])

# Evaluates rotation matrices for cameras that have location
# algorithm is straightforward: we assume copter has zero pitch and roll,
# and yaw is evaluated from current copter direction
# current direction is evaluated simply subtracting location of 
# current camera from the next camera location 
# i and j are unit axis vectors in chunk coordinate system
# i || North
def estimate_rotation_matrices(chunk, i, j):
    groups = copy.copy(chunk.camera_groups)

    groups.append(None)
    for group in groups:
        group_cameras = list(filter(lambda c: c.group == group, chunk.cameras))

        if len(group_cameras) == 0:
            continue

        if len(group_cameras) == 1:
            if group_cameras[0].reference.rotation is None:
                group_cameras[0].reference.rotation = ps.Vector([0,0,0])
            continue
        
        
        for idx, c in enumerate(group_cameras[0:-1]):
            next_camera = group_cameras[idx+1]

            if c.reference.rotation is None:
                if c.reference.location is None or next_camera.reference.location is None:
                    continue
                direction = delta_vector_to_chunk(c.reference.location, next_camera.reference.location)

                cos_yaw = direction * j
                yaw = math.degrees(math.acos(cos_yaw)) + 90 # TODO not sure about this offset

                #print("{} direction is {} and yaw is {}".format(i, direction, yaw))

                if direction * i > 0:
                    yaw = -yaw

                c.reference.rotation = ps.Vector([yaw, 0, 0])
                #print(c.reference.rotation)
        group_cameras[-1].reference.rotation = group_cameras[-2].reference.rotation

@time_measure
def align_cameras(chunk, min_latitude, min_longitude):
    if chunk.transform.scale is None:
        chunk.transform.scale = 1
        chunk.transform.rotation = ps.Matrix([[1,0,0], [0,1,0], [0,0,1]])
        chunk.transform.translation = ps.Vector([0,0,0])

    i, j, k = get_chunk_vectors(min_latitude, min_longitude) # i || North
    estimate_rotation_matrices(chunk, i, j)

    for c in chunk.cameras:
        if c.transform is not None:
            continue

        group_index = chunk.camera_groups.index(c.group) if c.group is not None else -1

        location = c.reference.location
        if location is None:
            continue
        chunk_coordinates = wgs_to_chunk(chunk, location)
        fi = c.reference.rotation.x + 90
        fi = math.radians(fi)
        roll = math.radians(c.reference.rotation.z)
        pitch = math.radians(c.reference.rotation.y)

        roll_mat = ps.Matrix([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
        pitch_mat = ps.Matrix([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
        yaw_mat = ps.Matrix([[math.cos(fi), -math.sin(fi), 0], [math.sin(fi), math.cos(fi), 0], [0, 0, 1]])

        r = roll_mat * pitch_mat * yaw_mat
        #r =  yaw_mat * pitch_mat * roll_mat

        #ii, jj, kk = r * i, r * j, k
        ii = r[0, 0] * i + r[1, 0] * j + r[2, 0] * k
        jj = r[0, 1] * i + r[1, 1] * j + r[2, 1] * k
        kk = r[0, 2] * i + r[1, 2] * j + r[2, 2] * k
        #ii, jj, kk = i * math.cos(fi) + j * math.sin(fi), j * math.cos(fi) - i * math.sin(fi), k
        c.transform = ps.Matrix([[ii.x, jj.x, kk.x, chunk_coordinates[0]],
                                 [ii.y, jj.y, kk.y, chunk_coordinates[1]],
                                 [ii.z, jj.z, kk.z, chunk_coordinates[2]],
                                 [0, 0, 0, 1]])

        #print(c.transform)

def revert_changes(chunk):
    for c in chunk.cameras:
        c.transform = None


def run_camera_alignment():
    doc = ps.app.document
    chunk = doc.chunk

    if not check_chunk(chunk):
        return

    min_latitude, min_longitude, max_latitude, max_longitude = get_chunk_bounds(chunk)
    try:
        align_cameras(chunk, min_latitude, min_longitude)
    except Exception as e:
        print(e)

#def injectFastLayout():
translator = get_translator(QtGui.QApplication.instance())
ps.app.addMenuItem("Workflow/Apply Vertical Camera Alignment...", run_camera_alignment)
