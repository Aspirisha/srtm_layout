# srtm_layout
This is a PhotoScan plugin which allows to import srtm DEM data into project and do some other fancy stuff.

Features: 
* <b>Importing srtm DEM</b>: go to Tools -> Import -> Import SRTM DEM. Plugin supports caching, so once the DEM (.hgt files) are downloaded, it is possible to use application offline.
* <b>Vertical camera alignment:</b> aligns all cameras so that they look vertically down (perpendicular to plane (longitude, altitude)).

##Installation##

To run the plugin, you need
* GDAL installed in your system
* extract files from this repo into 
 * under linux: ~/.local/share/data/Agisoft/PhotoScan Pro/scripts
 * under windows: 
 
