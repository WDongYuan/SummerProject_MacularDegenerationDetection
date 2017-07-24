Put the BoundaryExtraction.py and MyDenoise.py into same directory.

Extract boundary for one image:
python BoundaryExtraction.py "the_path_for_the_image" one_image
The result is stored in ./one_image

Extract the boundary for all the images in a directory:
python BoundaryExtraction.py "the_path_for_the_directory" image_dir
The result is stored in ./image_dir

Example:
python BoundaryExtraction.py ./Sample_Image/240_R_WET_FOVEA_1.jpg one_image
python BoundaryExtraction.py ./Sample_Image image_dir