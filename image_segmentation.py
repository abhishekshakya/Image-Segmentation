import numpy as np 
import matplotlib.pyplot as plt 
import cv2
from sklearn.cluster import KMeans
img = cv2.imread('find.jpg')#BGR format
cv2.imshow('image',img)
cv2.waitKey(0)

print(img.shape)
width = img.shape[0]
height = img.shape[1]

#we are creating segmentation of K colors
#firstly, we will find K dominating colors in an image using K means clustering 

#we have RGB image(3 channels) so we will use 3 channels values as features 
img = img.reshape(-1,3)
print(img.shape)

k = 4;#number of dominating colors you want

classifier = KMeans(n_clusters=k)
kmeans = classifier.fit(img)

labels = kmeans.labels_

dominating_colors = np.array(kmeans.cluster_centers_,dtype='uint8')

print(dominating_colors)
colors = []

for col in dominating_colors:
	b = col[0]#as cv2 read as BGR
	g = col[1]
	r = col[2]
	colors.append([r,g,b])
	#-----------------colors that are dominating-------------
	# swatch = np.zeros((100,100,3),dtype='uint8')
	# swatch[:,:,:] = [r,g,b]
	# plt.imshow(swatch)
	# plt.show()


segmented_img = np.zeros((img.shape[0],3),dtype='uint8');

for pix in range(segmented_img.shape[0]):
	r_g_b = colors[labels[pix]]
	segmented_img[pix] = r_g_b

segmented_img = segmented_img.reshape((width,height,3))
plt.imshow(segmented_img)
plt.show()


