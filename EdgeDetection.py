import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
import os
from MyDenoise import MyDenoise
import re
import random
from UpperBoundary import UpperBoundaryUpdate, ToGrayImage, MyDenoiseSobely

def FindLowerBoundary(path,mode):
	img_gray = ToGrayImage(path)
	_,img_bi = cv2.threshold(img_gray,10,255,cv2.THRESH_BINARY)
	# print(img_bi)
	# os.system("mkdir "+mode)
	# cv2.imwrite(mode+"/"+re.split(r'/|\.',path)[-2]+"_lower_boundary"+mode+"_.jpg",img_bi)
	# plt.imshow(img_bi,cmap = 'gray')
	# plt.show()
	threshold_rate = 0.8
	threshold_row = -1
	row,col = img_bi.shape
	for tmp_r in range(row-1,-1,-1):
		tmp_sum = sum(img_bi[tmp_r])
		rate = float(tmp_sum)/255/col
		if rate>threshold_rate:
			threshold_row = tmp_r
			break
	# if threshold_row==-1:
	# 	print("Lower boundary not found.")

	# else:
	# 	img_gray = img_gray[0:threshold_row+1,:]
	# 	plt.imshow(img_gray,cmap = "gray")
	# 	plt.show()
	# return
	return threshold_row



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

def EdgeDetection(img):
	# img = cv2.medianBlur(img,5)
	img = cv2.fastNlMeansDenoising(img,None,3,7,21)
	_,img = cv2.threshold(img,30,255,cv2.THRESH_TOZERO)
	denoise_img = img
	# print(img)
	# cv2.imwrite("Denoise.jpg",img)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	# convolute with proper kernels
	laplacian = cv2.Laplacian(img,cv2.CV_64F)
	sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)  # x
	sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)  # y
	# sobel2y = cv2.Sobel(sobely,cv2.CV_64F,0,1,ksize=3)
	# sobelxy = cv2.Sobel(img,cv2.CV_64F,1,1,ksize=5)  # y
	canny = cv2.Canny(img,100,200)
	contour_image, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	# print(canny)
	# cv2.imwrite('laplacian.jpg',laplacian)
	# cv2.imwrite('sobelx.jpg',sobelx)
	# cv2.imwrite('sobely.jpg',sobely)
	# cv2.imwrite('sobelxy.jpg',sobelxy)
	# cv2.imwrite('canny.jpg',canny)

	# plt.subplot(3,2,1),plt.imshow(img,cmap = 'gray')
	# plt.title('Original'), plt.xticks([]), plt.yticks([])

	# plt.subplot(3,2,2),plt.imshow(laplacian,cmap = 'gray')
	# plt.title('Laplacian'), plt.xticks([]), plt.yticks([])

	# plt.subplot(3,2,3),plt.imshow(sobelx,cmap = 'gray')
	# plt.title('Sobel X'), plt.xticks([]), plt.yticks([])

	# plt.subplot(3,2,4),plt.imshow(sobely,cmap = 'gray')
	# plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

	# plt.subplot(3,2,4),plt.imshow(sobelxy,cmap = 'gray')
	# plt.title('Sobel XY'), plt.xticks([]), plt.yticks([])

	# plt.subplot(3,2,5),plt.imshow(canny,cmap = 'gray')
	# plt.title('Canny'), plt.xticks([]), plt.yticks([])

	# plt.show()
	# return {"denoise":img}
	return {"denoise":denoise_img,"laplacian":laplacian,"canny":canny,"sobely":sobely,"sobelx":sobelx,"contour":contour_image}

def UpperBoundary(path,mode,flip=False,crop_margin=50):
	# img = MyDenoiseSobely(path)
	# print("Finish MyDenoise.")
	img = None
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
	detect_range = 50
	begin_row = -1
	for i in range(2,row-1):
		##Find the begin point of the boundary.
		if preprocess_img[i][0]<50:
			continue

		tmp_row = i
		tmp_col = 0
		found = True
		for j in range(detect_range):
			u_r = preprocess_img[tmp_row-1][tmp_col+1]-preprocess_img[tmp_row-2][tmp_col+1]
			r_r = preprocess_img[tmp_row][tmp_col+1]-preprocess_img[tmp_row-1][tmp_col+1]
			d_r = preprocess_img[tmp_row+1][tmp_col+1]-preprocess_img[tmp_row][tmp_col+1]
			tmp_max = max(u_r,r_r,d_r)
			# if tmp_max==0:
			if preprocess_img[tmp_row][tmp_col+1]==0 and preprocess_img[tmp_row-1][tmp_col+1]==0 \
			and preprocess_img[tmp_row+1][tmp_col+1]==0:
				found = False
				break
			elif tmp_max==u_r:
				tmp_row -= 1
				tmp_col += 1
			elif tmp_max==r_r:
				tmp_col += 1
			elif tmp_max==d_r:
				tmp_row += 1
				tmp_col += 1
		if found:
			begin_row = i
			break

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


	# tmp_bd_img = np.zeros((row,col),dtype=np.uint8)
	# for pos in up_bd:
	# 	tmp_bd_img[pos[0]][pos[1]] = 255

	# if flag==False:
	# 	for pos in up_bd_flip:
	# 		tmp_bd_img[pos[0]][col-1-pos[1]] = 255


	bd_img = np.zeros((row,col),dtype=np.uint8)
	for pos in up_bd_flip:
		pos[1] = col-1-pos[1]
	final_up_bd = LineUpPixels(up_bd,up_bd_flip,col)
	# print(final_up_bd)
	for i in range(len(final_up_bd)):
		bd_img[int(final_up_bd[i])][i] = 255

	if mode!="crop_image_dir" and mode!="crop_image":
		os.system("mkdir "+mode)
		cv2.imwrite(mode+"/"+re.split(r'/|\.',path)[-2]+"_upper_boundary_"+mode+"_.jpg",\
			np.hstack((preprocess_img,bd_img)))
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

	for tmp_r in range(min_up_bd,lw_bd+1):
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
	cv2.imwrite(mode+"/"+re.split(r'/|\.',path)[-2]+"_"+mode+"_.jpg",\
		np.vstack((np.hstack((cm_img,bd_img)),np.hstack((crop_img,crop_img)))))
	return






if __name__=="__main__":

	mode = sys.argv[2]
	path = sys.argv[1]
	if mode == "mydenoise_sobely":
		os.system("mkdir "+mode)
		img = MyDenoiseSobely(path)
		cv2.imwrite(mode+"/"+re.split(r'/|\.',path)[-2]+"_mydenoise_sobely.jpg",img)

	elif mode == "upper_boundary":
		FindUpperBoundary(path,mode)

	elif mode == "upper_boundary_origin_image":
		FindUpperBoundary(path,mode,origin_img=True)

	elif mode == "crop_image":
		CropImage(path,mode)
	
	elif mode == "crop_image_dir":
		image_list = os.listdir(path)
		# image_list = random.sample(os.listdir(path),200)
		for i in range(len(image_list)):
			print(image_list[i])
			if "jpg" not in image_list[i]:
				continue
			CropImage(path+"/"+image_list[i],mode)

	elif mode == "upper_boundary_dir":
		image_list = os.listdir(path)
		# image_list = random.sample(os.listdir(path),200)
		for i in range(len(image_list)):
			print(image_list[i])
			if "jpg" not in image_list[i]:
				continue
			FindUpperBoundary(path+"/"+image_list[i],mode)

	elif mode == "mydenoise_sobely_dir":
		os.system("mkdir "+mode)
		image_list = os.listdir(path)
		for image_name in image_list:
			print(image_name)
			if "jpg" not in image_name:
				continue
			tmp_img = MyDenoiseSobely(path+"/"+image_name)
			cv2.imwrite(mode+"/"+image_name,tmp_img)
	
	elif mode == "lower_boundary":
		FindLowerBoundary(path,mode)

	elif mode == "process_dir":
		image_dir = path
		# image_dir = "/Users/weidong/GoogleDrive/CMU/Summer Project/conv"
		image_list = os.listdir(image_dir)
		# result_dir = "Edge_Detection_IMG"
		result_dir = "sample_result"
		os.system("mkdir "+result_dir)
		count = 0
		for image_name in image_list:
			if "jpg" not in image_name:
				continue
			# if image_name !="27478_R_DRY_OTHER.jpg":
			# 	continue
			print(image_name)
			path = image_dir+"/"+image_name
			img_gray = ToGrayImage(path)
			print("Convert to grayscale: done.")

			img_denoise = MyDenoise(img_gray,5)
			print("Implemented denoise: done.")

			mydenoise_edge_det = EdgeDetection(img_denoise)
			print("MyDenoise edge detection: done.")

			nodenoise_edge_det = EdgeDetection(img_gray)
			print("NoDenoise edge detection: done.")

			mydenoise_combine_img = np.vstack((np.hstack((img_gray,mydenoise_edge_det["denoise"],mydenoise_edge_det["laplacian"])),\
				np.hstack((mydenoise_edge_det["sobely"],mydenoise_edge_det["sobelx"],mydenoise_edge_det["canny"]))))
			nodenoise_combine_img = np.vstack((np.hstack((img_gray,nodenoise_edge_det["denoise"],nodenoise_edge_det["laplacian"])),\
				np.hstack((nodenoise_edge_det["sobely"],nodenoise_edge_det["sobelx"],nodenoise_edge_det["canny"]))))
			# combine_img = mydenoise_edge_det["denoise"]
			# combine_img = np.vstack((np.hstack((img_gray,nodenoise_edge_det["sobely"],nodenoise_edge_det["contour"])),\
			# 	np.hstack((img_denoise,mydenoise_edge_det["sobely"],mydenoise_edge_det["contour"]))))
			# combine_img = mydenoise_edge_det["sobely"]
			# combine_img = np.hstack((img_gray,nodenoise_edge_det["sobely"]))
			cv2.imwrite(result_dir+"/"+image_name.split(".")[0]+"_mydenoise.jpg",mydenoise_combine_img)
			cv2.imwrite(result_dir+"/"+image_name.split(".")[0]+"_nodenoise.jpg",nodenoise_combine_img)
			count+=1
			print(count)

			print("")
	
