import numpy as np
import cv2
def MyDenoise(img,window_size):
	img = np.array(img)
	row = img.shape[0]
	col = img.shape[1]
	# print((row,col))
	new_img = np.zeros(img.shape,dtype=np.uint8)
	half = (window_size-1)/2
	for i in range(half):
		for j in range(col):
			new_img[i][j] = 0

	for i in range(half,row-half):
		for j in range(half):
			new_img[i][j] = 0
		for j in range(col-half,col):
			new_img[i][j] = 0

	for i in range(row-half,row):
		for j in range(col):
			new_img[i][j] = 0

	for i in range(half,row-half):
		for j in range(half,col-half):
			sub_max = np.matrix(img[i-half:i+half+1,j-half:j+half+1])
			# print(sub_max)
			tmp_var = np.sqrt(sub_max.var())
			if tmp_var>30:
				new_img[i][j] = 0
			else:
				new_img[i][j] = img[i][j]
			# print(tmp_var)
			# new_img[i][j] = tmp_var
	# print(new_img)
	# cv2.imwrite("mydenoise.jpg",new_img)
	return new_img

