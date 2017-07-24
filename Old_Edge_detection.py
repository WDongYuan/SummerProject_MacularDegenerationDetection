import cv2
from matplotlib import pyplot as plt
import os
#
image_path = "./"
images = []
for i in (os.listdir(image_path)):
# i = '0012.jpg'
    if i.endswith('.jpg'):
        img = cv2.imread(image_path+i,0)
        ret,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        edges = cv2.Canny(img,100,200)
        images.append(img)
        images.append(edges)

titles = ['Original Image', 'Edge Image']
# images = [img] # , edges]
for i in range(len(images)):
    plt.subplot(6,7,i+1),plt.imshow(images[i],'gray')
    plt.title('Orig_' if i&1==0 else 'Edges_'+ str(i))
    plt.xticks([]),plt.yticks([])


plt.show()