import cv2
import numpy as np
from matplotlib import pyplot as plt

def CropLowerBoundary(img):
	# img_gray = ToGrayImage(path)
	_,img_bi = cv2.threshold(img,60,255,cv2.THRESH_BINARY)
	threshold_rate = 0.95
	threshold_row = -1
	row,col = img_bi.shape
	for tmp_r in range(row-1,-1,-1):
		tmp_sum = sum(img_bi[tmp_r])
		rate = float(tmp_sum)/255/col
		# print(rate)
		if rate>threshold_rate:
			threshold_row = tmp_r
			break
	img = img[0:threshold_row,:]
	# plt.imshow(img,"gray")
	# plt.show()
	return img

def BlackRate(img,boundary):
	black_threshold = 60
	row,col = img.shape
	black_count = 0
	for r in range(row):
		for c in range(col):
			if r >= boundary[c]:
				if img[r][c]<black_threshold:
					black_count += 1
	return [float(black_count)/(row*col)]


def MinMax(boundary):
	return [min(boundary),max(boundary)]

def MinGridBlackRate(img,boundary):
	grid_r = 5
	grid_c = 8
	row,col = img.shape
	feature = []
	black_threshold = 60
	for i in range(grid_r):
		for j in range(grid_c):
			tmp_r = [i*row/grid_r,(i+1)*row/grid_r]
			tmp_c = [j*col/grid_c,(j+1)*col/grid_c]
			black_count = 0
			for r in range(tmp_r[0],tmp_r[1]):
				for c in range(tmp_c[0],tmp_c[1]):
					if r >= boundary[c]:
						if img[r][c]<black_threshold:
							black_count += 1
			feature.append(float(black_count)/(row*col/(grid_r*grid_c)))
	return [max(feature)]

def ImageShape(img):
	row,col = img.shape
	return [row,col]

def CountHill(boundary,img):
##Feature: left_most_pixel_gradiant, hill_number, average_hill_peak, average_hill_valley

	##Left most and right most
	feature = []
	row,col = img.shape

	feature.append(float(boundary[0])/row)
	feature.append(float(boundary[-1])/row)

	sensitive_height = 5
	sensitive_length = 10
	bd = boundary
	##Remove horizontal pixel
	bd_noh = [[bd[0],0]]
	for i in range(1,len(bd)):
		if bd[i]!=bd_noh[-1][0]:
			bd_noh.append([bd[i],i])

	##Find every peak and valley
	bd_hill = [bd_noh[0]]
	for i in range(1,len(bd_noh)-1):
		if (bd_noh[i][0]-bd_noh[i-1][0])*(bd_noh[i][0]-bd_noh[i+1][0])>0:
			bd_hill.append(bd_noh[i])
	bd_hill.append(bd_noh[-1])
	# print(bd_hill)

	##Smooth
	bd_hill_smooth = [bd_hill[0]]
	for i in range(1,len(bd_hill)):
		if (np.abs(bd_hill[i][0]-bd_hill[i-1][0])>sensitive_height and np.abs(bd_hill[i][1]-bd_hill[i-1][1])>sensitive_length) \
		or \
		(np.abs(bd_hill[i][0]-bd_hill_smooth[-1][0])>sensitive_height and np.abs(bd_hill[i][1]-bd_hill_smooth[-1][1])>sensitive_length):
			bd_hill_smooth.append(bd_hill[i])

	##Find every peak and valley
	bd_hill = [bd_hill_smooth[0]]
	for i in range(1,len(bd_hill_smooth)-1):
		if (bd_hill_smooth[i][0]-bd_hill_smooth[i-1][0])*(bd_hill_smooth[i][0]-bd_hill_smooth[i+1][0])>0:
			bd_hill.append(bd_hill_smooth[i])
	bd_hill.append(bd_hill_smooth[-1])
	# print(bd_hill)

	##Count the hill:
	count = 0
	hill_peak = []
	hill_valley = []
	if bd_hill[0][0]-bd_hill[1][0]<0:
		# feature.append(1)
		count = (len(bd_hill)+1)/2
		hill_peak.append(bd_hill[0][0])
	else:
		# feature.append(-1)
		count = len(bd_hill)/2
		hill_valley.append(bd_hill[0][0])
	feature.append(count)
	# print(bd_hill)
	# print(count)

	#Get the average of the peak pixel
	for i in range(1,len(bd_hill)):
		if bd_hill[i][0]-bd_hill[i-1][0]<0:
			hill_peak.append(bd_hill[i][0])
		else:
			hill_valley.append(bd_hill[i][0])
	# print(hill_peak)
	if len(hill_peak)==0 or len(hill_valley)==0:
		return [False,feature]
	# feature.append(float(np.average(hill_peak))/row)
	# feature.append(float(np.average(hill_valley))/row)

	feature.append(float(np.average(hill_peak))/row-float(np.average(hill_peak))/row)

	return [True,feature]

def GetFeature(image_path):
	#MinBlackRate, left_most_pixel_gradiant,  hill_number, average_hill_peak, average_hill_valley, BlackRate
	boundary_path = image_path.split(".")[0]+"_upper_boundary.txt"
	file = open(boundary_path)
	tmp_str = file.readline().strip()
	tmp_arr = tmp_str.split(" ")
	boundary = []
	for i in range(len(tmp_arr)):
		if tmp_arr[i]!="":
			boundary.append(int(tmp_arr[i]))
	boundary = np.array(boundary)
	file.close()
	image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
	image = CropLowerBoundary(image)

	feature = MinGridBlackRate(image,boundary)+BlackRate(image,boundary)
	flag,tmp_feature = CountHill(boundary,image)
	if flag==False:
		return [False,feature]
	feature += tmp_feature
	return [True,feature]


if __name__=="__main__":
	path = "/Users/weidong/GoogleDrive/CMU/Summer Project/MyCode/BoundaryExtraction/image_dir/27473_L ONLY_WET_FOVEA_1.jpg"
	print(GetFeature(path))



