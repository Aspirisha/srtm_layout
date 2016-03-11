# srtm_layout
This is a PhotoScan plugin which allows to import srtm DEM data into project and do some other fancy stuff.

Features: 
* <b>Importing srtm DEM</b>: go to Tools -> Import -> Import SRTM DEM. Plugin supports caching, so once the DEM (.hgt files) are downloaded, it is possible to use application offline.
* <b>Importing srtm mesh</b>: go to Toosl -> Import -> Import SRTM mesh. 
* <b>Vertical camera alignment:</b> aligns all cameras so that they look vertically down (perpendicular to plane (longitude, altitude)). This step includes photoscan-performed camera calibration retrieving.

##Installation##

To run the plugin, you need
* GDAL installed in your system and available via command prompt
* extract files from this repo into 
 * under linux: ~/.local/share/data/Agisoft/PhotoScan Pro/scripts
 * under windows: /c/Users/Andy/AppData/Local/Agisoft/PhotoScan Pro/scripts
* extract files from storage-nas-1/Departments/Office/Software_development/FastLayout/site-packages int %PHOTOSCAN_HOME%/python/Lib/site-packages  
