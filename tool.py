import os
from matplotlib import pyplot as plt
import cv2
import numpy as np
from MyDenoise import MyDenoise
import sys
import re

if __name__=="__main__":
	os.system("mkdir Test_Image")
	mode = sys.argv[2]
	path = sys.argv[1]
	img = cv2.imread(path)
	if mode=="gaussian_blur":
		blur = cv2.GaussianBlur(img,(5,5),0)
		cv2.imwrite("./Test_Image/"+re.split(r'/|\.',path)[-2]+"_"+mode+".jpg",blur)
	elif mode=="threshold":
		thre = sys.argv[3]
		_,threshold_img = cv2.threshold(img,int(thre),255,cv2.THRESH_TOZERO)
		cv2.imwrite("./Test_Image/"+re.split(r'/|\.',path)[-2]+"_"+mode+".jpg",threshold_img)
	elif mode=="gaussian_blur_threshold":
		thre = sys.argv[3]
		blur = cv2.GaussianBlur(img,(5,5),0)
		_,threshold_img = cv2.threshold(blur,int(thre),255,cv2.THRESH_TOZERO)
		cv2.imwrite("./Test_Image/"+re.split(r'/|\.',path)[-2]+"_"+mode+".jpg",threshold_img)


