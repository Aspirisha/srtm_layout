import PhotoScan as ps
import math
import srtm
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'delaunay'))
from DelaunayVoronoi import computeDelaunayTriangulation 
import numpy as np

def frange(x, y, jump):
	while x < y:
		yield x
		x += jump

def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = (arr[:,0]**2 + arr[:,1]**2 + arr[:,2]**2) ** 0.5
    print(lens)
    arr[:,0] /= lens
    arr[:,1] /= lens
    arr[:,2] /= lens                
    return arr


def build_mesh(output_file, min_latitude, max_latitude, min_longitude, max_longitude, lat_step, long_step):
	elevation_data = srtm.get_data()

	mesh_points = np.array([[longitude, latitude, elevation_data.get_elevation(latitude, longitude, approximate=True) ] 
		for latitude in frange(min_latitude, max_latitude, lat_step)
		for longitude in frange(min_longitude, max_longitude, long_step)])

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
	print('fefefe')
	normalize_v3(n)
	print('fefefe')
	norms[ faces[:,0] ] += n
	norms[ faces[:,1] ] += n
	norms[ faces[:,2] ] += n
	normalize_v3(norms)

	print('there')
	with open(output_file, "w") as f:
		f.write('mtllib mymodel.mtl\nusemtl Solid\n')
		for p in mesh_points:
			f.write("v {:.6f} {:.6f} {:.6f}\n".format(p[0], p[1], p[2]))
		for n in norms:
			f.write("vn {:.6f} {:.6f} {:.6f}\n".format(n[0], n[1], n[2]))
		for face in faces:
			f.write("f {0}//{0} {1}//{1} {2}//{2}\n".format(face[1] + 1, face[0] + 1, face[2] + 1))



def build_layout():
	print('here')
	doc = ps.app.document
	chunk = doc.chunk
	delta_latitude_scale_to_meters = 40008000 / 360

	'''
	init_location = chunk.cameras[0].reference.location
	latitude = init_location.location[0]
	delta_longitude_scale_to_meters = 40075160 * math.cos(math.radians(latitude)) / 360

	delta_meters_scale_to_chunk = 0.1
	scales = [delta_latitude_scale_to_meters * delta_meters_scale_to_chunk, 
		delta_longitude_scale_to_meters * delta_meters_scale_to_chunk, delta_meters_scale_to_chunk]

	min_latitude = latitude
	max_latitude = latitude
	min_longitude = init_location[1]
	max_longitude = min_longitude

	for c in chunk.cameras:
		location = c.reference.location
		chunk_coordinates = ps.Vector([(x - x0) * s for x, x0, s in zip(location, init_location, scales)])
		c.transform = ps.Matrix([[0,0,1,chunk_coordinates[0]],[0,1,0,chunk_coordinates[1]],[0,0,-1,chunk_coordinates[2]], [0,0,0,1]])

	'''
	build_mesh(output_file="mymodel.obj", min_latitude=49.12, max_latitude=49.14, min_longitude=87.90, max_longitude=87.92, lat_step=0.0003, long_step=0.0003)

ps.app.addMenuItem("Workflow/Build Fast Layout", build_layout)
#elevation_data = srtm.get_data()
#print(elevation_data.get_elevation(57.1, 95.1, approximate=True))