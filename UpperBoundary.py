import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
import os
from MyDenoise import MyDenoise
import re
import random

# GrayScale Image Convertor
# https://extr3metech.wordpress.com
def ToGrayImage(path):
	image = cv2.imread(path)
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# cv2.imwrite('gray_image.jpg',gray_image)
	# cv2.imshow('color_image',image)
	# cv2.imshow('gray_image',gray_image)
	# cv2.waitKey(0)                 # Waits forever for user to press any key
	# cv2.destroyAllWindows()        # Closes displayed windows
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
	# cv2.imwrite("haha.jpg",blur)

	# blur = ToGrayImage(path).astype(np.int)

	# _,threshold_img = cv2.threshold(blur,100,255,cv2.THRESH_TOZERO)
	# # threshold_img = threshold_img.astype(np.int)
	# blur2 = cv2.GaussianBlur(threshold_img,(5,5),0)

	row,col = blur.shape

	_,threshold_img = cv2.threshold(blur,40,255,cv2.THRESH_TOZERO)

	# cv2.imwrite("haha.jpg",blur)
	# plt.subplot(1,2,1),plt.imshow(blur,'gray')
	# plt.xticks([]),plt.yticks([])
	# plt.subplot(1,2,2),plt.imshow(threshold_img,'gray')
	# plt.xticks([]),plt.yticks([])
	# plt.show()

	preprocess_img = threshold_img[:,crop_margin:col-crop_margin].astype(np.float)
	# cv2.imwrite("Boundary/haha_upper_boundary.jpg",preprocess_img)


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
			# print(visit[tmp_row-1][tmp_col])

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
			# print(visit[tmp_row-1][tmp_col])

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
			# if (tmp_row==196 and tmp_col==217):
			# 	print(neigh)
			# 	##194,218
			# 	print(preprocess_img[194][218])
			# 	print(np.average(preprocess_img[194-devide_window-1:194-1,218]))
			# 	print(np.mean(preprocess_img[194-devide_window-1:194-1,218]))
			# 	print(preprocess_img[200][221])
			# 	print(np.average(preprocess_img[200-devide_window-1:200-1,221]))
			# 	print(np.mean(preprocess_img[200-devide_window-1:200-1,221]))
			if len(neigh)==0:
				found_flag = False
				break
			tmp_row = tmp_row+neigh[0][1]
			tmp_col = tmp_col+neigh[0][2]
			visit[tmp_row][tmp_col] = 1
			up_bd.append([tmp_row,tmp_col])
			# print([tmp_row,tmp_col])

		# up_bd =[begin_row]
		# flag =TrackBoundary(preprocess_img,up_bd,col)

	# if found_flag==False:
	# 	print("No complete upper boundary!")
	# 	return
	bd_img = np.zeros(preprocess_img.shape,dtype=np.uint8)
	# print(up_bd)
	for pos in up_bd:
		bd_img[pos[0]][pos[1]] = 255
	# origin_img = ToGrayImage(path)

	
	# print(preprocess_img.shape)
	# print(bd_img.shape)
	# print(origin_img[:,50:col].shape)
	# print(origin_img.shape)
	# cv2.imwrite("haha.jpg",bd_img)
	# cv2.imwrite("Boundary/"+re.split(r'/|\.',path)[-2]+"_upper_boundary_"+mode+"_.jpg",\
	# 	np.vstack((np.hstack((preprocess_img,bd_img)),np.hstack((bd_img,origin_img[:,50:origin_img.shape[1]])))))

	# os.system("mkdir Boundary")
	# cv2.imwrite("Boundary/"+re.split(r'/|\.',path)[-2]+"_upper_boundary_"+mode+"_.jpg",\
	# 	np.hstack((preprocess_img,bd_img)))
	return [found_flag,up_bd,preprocess_img]