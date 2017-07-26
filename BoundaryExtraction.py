import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
import os
from MyDenoise import MyDenoise
import re
import random

def FindLowerBoundary(path,mode):
	img_gray = ToGrayImage(path)
	_,img_bi = cv2.threshold(img_gray,10,255,cv2.THRESH_BINARY)
	threshold_rate = 0.8
	threshold_row = -1
	row,col = img_bi.shape
	for tmp_r in range(row-1,-1,-1):
		tmp_sum = sum(img_bi[tmp_r])
		rate = float(tmp_sum)/255/col
		if rate>threshold_rate:
			threshold_row = tmp_r
			break
	return threshold_row



# GrayScale Image Convertor
# https://extr3metech.wordpress.com
def ToGrayImage(path):
	image = cv2.imread(path)
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	return gray_image

def MyDenoiseSobely(path):
	img_gray = ToGrayImage(path)
	img_mydenoise = MyDenoise(img_gray,5)
	img_denoise = cv2.fastNlMeansDenoising(img_mydenoise,None,3,7,21)
	_,img_thre = cv2.threshold(img_denoise,100,255,cv2.THRESH_TOZERO)
	sobely = cv2.Sobel(img_thre,cv2.CV_64F,0,1,ksize=3)
	return sobely

def EdgeDetection(img):
	img = cv2.fastNlMeansDenoising(img,None,3,7,21)
	_,img = cv2.threshold(img,30,255,cv2.THRESH_TOZERO)
	denoise_img = img
	laplacian = cv2.Laplacian(img,cv2.CV_64F)
	sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)  # x
	sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)  # y
	canny = cv2.Canny(img,100,200)
	contour_image, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	return {"denoise":denoise_img,"laplacian":laplacian,"canny":canny,"sobely":sobely,"sobelx":sobelx,"contour":contour_image}

# GrayScale Image Convertor
# https://extr3metech.wordpress.com
def ToGrayImage(path):
	image = cv2.imread(path)
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	return gray_image

def MyDenoiseSobely(path):
	img_gray = ToGrayImage(path)
	img_mydenoise = MyDenoise(img_gray,5)
	img_denoise = cv2.fastNlMeansDenoising(img_mydenoise,None,3,7,21)
	_,img_thre = cv2.threshold(img_denoise,100,255,cv2.THRESH_TOZERO)
	sobely = cv2.Sobel(img_thre,cv2.CV_64F,0,1,ksize=3)
	return sobely


def UpperBoundaryUpdate(path,mode,flip=False,crop_margin=50,origin_img=False):
	img = None
	
	if origin_img==True:
		mydenoise_img = MyDenoiseSobely(path)
		print("Finish MyDenoise.")
		if flip==True:
			img = cv2.flip(mydenoise_img,1)
		else:
			img = mydenoise_img
	else:
		if flip==True:
			img = cv2.flip(cv2.imread(path,cv2.IMREAD_GRAYSCALE),1)
		else:
			img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)

	blur = cv2.GaussianBlur(img,(5,5),0).astype(np.float32)

	row,col = blur.shape

	_,threshold_img = cv2.threshold(blur,40,255,cv2.THRESH_TOZERO)

	preprocess_img = threshold_img[:,crop_margin:col-crop_margin].astype(np.float)


	row,col = preprocess_img.shape
	##Find the begin segment of the upper boundary, the segment's length is 
	##decided by the detect_range.
	detect_range = 100
	begin_row = -1
	begin_visit = np.zeros(preprocess_img.shape)
	for i in range(2,row-1):
		##Find the begin point of the boundary.
		if preprocess_img[i][0]<50:
			continue

		tmp_row = i
		begin_visit[begin_row][0] = 1
		visit_list = [[i,0]]
		tmp_col = 0
		window = 4
		devide_window = 10
		found = True
		while tmp_col<detect_range:
			neigh = []
			##Deal with the pixel at the same column
			##Up pixel

			if sum(begin_visit[tmp_row-1-window:tmp_row-1,tmp_col])==0:
				for tmp_r in range(tmp_row-window,tmp_row):
					if preprocess_img[tmp_r][tmp_col]!=0:
						neigh.append([(preprocess_img[tmp_r][tmp_col]-preprocess_img[tmp_r-1][tmp_col])-\
							np.average(preprocess_img[tmp_r-devide_window-1:tmp_r-1,tmp_col]),tmp_r-tmp_row,0])
			##Down pixel
			if sum(begin_visit[tmp_row+1:tmp_row+1+window,tmp_col])==0:
				for tmp_r in range(tmp_row+1,tmp_row+1+window):
					if preprocess_img[tmp_r][tmp_col]!=0:
						neigh.append([(preprocess_img[tmp_r][tmp_col]-preprocess_img[tmp_r-1][tmp_col])-\
							np.average(preprocess_img[tmp_r-devide_window-1:tmp_r-1,tmp_col]),tmp_r-tmp_row,0])

			##Right column pixel
			for tmp_r in range(tmp_row-window,tmp_row+window+1):
				for tmp_c in range(tmp_col+1,tmp_col+window+1):
					if preprocess_img[tmp_r][tmp_c]!=0:
						neigh.append([(preprocess_img[tmp_r][tmp_c]-preprocess_img[tmp_r-1][tmp_c])-\
							np.average(preprocess_img[tmp_r-devide_window-1:tmp_r-1,tmp_c]),tmp_r-tmp_row,tmp_c-tmp_col])

			neigh = sorted(neigh,key=lambda tup: tup[0],reverse=True)
			if len(neigh)==0:
				found = False
				break
			tmp_row = tmp_row+neigh[0][1]
			tmp_col = tmp_col+neigh[0][2]
			visit_list.append([tmp_row,tmp_col])
			begin_visit[tmp_row][tmp_col] = 1
			
		if found:
			begin_row = i
			break

		for visit_point in visit_list:
			begin_visit[visit_point[0]][visit_point[1]]=0

	##Begin to track the upper boundary
	# origin_img = ToGrayImage(path).astype(np.int)
	found_flag = True
	visit = np.zeros(preprocess_img.shape)
	if begin_row == -1:
		print("No start point!")
		return [False,[],preprocess_img]
	else:
		up_bd = [[begin_row,0]]
		tmp_row = begin_row
		visit[begin_row][0] = 1
		tmp_col = 0
		window = 4
		devide_window = 10
		while tmp_col<col-window:
			neigh = []
			##Deal with the pixel at the same column
			##Up pixel

			if sum(visit[tmp_row-1-window:tmp_row-1,tmp_col])==0:
				for tmp_r in range(tmp_row-window,tmp_row):
					if preprocess_img[tmp_r][tmp_col]!=0:
						neigh.append([(preprocess_img[tmp_r][tmp_col]-preprocess_img[tmp_r-1][tmp_col])-\
							np.average(preprocess_img[tmp_r-devide_window-1:tmp_r-1,tmp_col]),tmp_r-tmp_row,0])
			##Down pixel
			if sum(visit[tmp_row+1:tmp_row+1+window,tmp_col])==0:
				for tmp_r in range(tmp_row+1,tmp_row+1+window):
					if preprocess_img[tmp_r][tmp_col]!=0:
						neigh.append([(preprocess_img[tmp_r][tmp_col]-preprocess_img[tmp_r-1][tmp_col])-\
							np.average(preprocess_img[tmp_r-devide_window-1:tmp_r-1,tmp_col]),tmp_r-tmp_row,0])

			##Right column pixel
			for tmp_r in range(tmp_row-window,tmp_row+window+1):
				for tmp_c in range(tmp_col+1,tmp_col+window+1):
					if preprocess_img[tmp_r][tmp_c]!=0:
						neigh.append([(preprocess_img[tmp_r][tmp_c]-preprocess_img[tmp_r-1][tmp_c])-\
							np.average(preprocess_img[tmp_r-devide_window-1:tmp_r-1,tmp_c]),tmp_r-tmp_row,tmp_c-tmp_col])

			neigh = sorted(neigh,key=lambda tup: tup[0],reverse=True)
			if len(neigh)==0:
				found_flag = False
				break
			tmp_row = tmp_row+neigh[0][1]
			tmp_col = tmp_col+neigh[0][2]
			visit[tmp_row][tmp_col] = 1
			up_bd.append([tmp_row,tmp_col])
	bd_img = np.zeros(preprocess_img.shape,dtype=np.uint8)
	for pos in up_bd:
		bd_img[pos[0]][pos[1]] = 255
	return [found_flag,up_bd,preprocess_img]

def FindUpperBoundary(path,mode,origin_img=False):
	# flag,up_bd,preprocess_img = UpperBoundary(path,mode)
	flag,up_bd,preprocess_img = UpperBoundaryUpdate(path,mode,origin_img=origin_img)
	crop_margin = 50
	up_bd_flip = []
	if flag==False:
		# flag_flip,up_bd_flip,preprocess_img_flip = UpperBoundary(path,mode,flip=True)
		flag_flip,up_bd_flip,preprocess_img_flip = UpperBoundaryUpdate(path,mode,flip=True,origin_img=origin_img)
		if up_bd_flip==False:
			return

	row,col = preprocess_img.shape
	bd_img = np.zeros((row,col),dtype=np.uint8)
	for pos in up_bd_flip:
		pos[1] = col-1-pos[1]
	final_up_bd = LineUpPixels(up_bd,up_bd_flip,col)
	# print(final_up_bd)
	for i in range(len(final_up_bd)):
		bd_img[int(final_up_bd[i])][i] = 255

	
	return final_up_bd
def LineUpPixels(l1,l2,end_col):
	flag_arr = np.zeros((end_col,))
	for pos in l1:
		# print(flag_arr)
		# print(pos)
		if flag_arr[pos[1]]==0:
			flag_arr[pos[1]] = pos[0]
		else:
			flag_arr[pos[1]] = min(flag_arr[pos[1]],pos[0])

	for pos in l2:
		if flag_arr[pos[1]]==0:
			flag_arr[pos[1]] = pos[0]
		else:
			flag_arr[pos[1]] = min(flag_arr[pos[1]],pos[0])

	# print(flag_arr)
	seg_begin = -1
	idx = 0
	while idx<end_col:
		if flag_arr[idx]==0:
			idx += 1
		else:
			if seg_begin==idx-1:
				seg_begin = idx
				idx += 1
			else:
				lope = float(flag_arr[idx]-flag_arr[seg_begin])/(idx-seg_begin)
				for i in range(seg_begin+1,idx):
					flag_arr[i] = int(flag_arr[seg_begin]+lope*(i-seg_begin))
				seg_begin = idx
				idx += 1
	if seg_begin!=end_col-1:
		for i in range(seg_begin+1,end_col):
			flag_arr[i] = flag_arr[seg_begin]

	return flag_arr

def CropImage(path,mode):
	up_bd = FindUpperBoundary(path,mode,origin_img=True).astype(np.int)
	lw_bd = int(FindLowerBoundary(path,mode))
	crop_margin = 50
	img = ToGrayImage(path)
	#Crop margin image
	cm_img = img[:,crop_margin:-crop_margin]

	min_up_bd = min(up_bd)
	row,col = cm_img.shape
	new_img = np.zeros(cm_img.shape)
	if lw_bd==-1:
		lw_bd = row

	for tmp_r in range(min_up_bd,lw_bd):
		for tmp_c in range(col):
			if tmp_r>=up_bd[tmp_c]:
				new_img[tmp_r][tmp_c] = cm_img[tmp_r][tmp_c]
	crop_img = new_img[min_up_bd:lw_bd+1,:]

	bd_img = np.zeros(cm_img.shape)
	for i in range(len(up_bd)):
		bd_img[up_bd[i]][i] = 255
		bd_img[lw_bd][i] = 255
		##Add margin
		for j in range(10):
			bd_img[-j][i] = 255
			img[-j][i] = 255


	os.system("mkdir "+mode)
	cv2.imwrite(mode+"/"+re.split(r'/|\.',path)[-2]+".jpg",crop_img)

	file = open(mode+"/"+re.split(r'/|\.',path)[-2]+"_upper_boundary.txt","w+")
	for i in range(len(up_bd)):
		file.write(str(up_bd[i]-min_up_bd)+" ")
	file.close()

	return






if __name__=="__main__":

	mode = sys.argv[2]
	path = sys.argv[1]

	if mode == "one_image":
		CropImage(path,mode)
	
	elif mode == "image_dir":
		image_list = os.listdir(path)
		# image_list = random.sample(os.listdir(path),200)
		for i in range(len(image_list)):
			print(image_list[i])
			if "jpg" not in image_list[i]:
				continue
			CropImage(path+"/"+image_list[i],mode)
