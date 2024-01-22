#!/usr/bin/env python
# coding: utf-8

# ### 4. Eigenfaces and simple face recognition [25 points].This question is a simplified illustration of using PCA for face recognition. We will use a subset of data from the famous Yale Face dataset.

# In[1]:


import numpy as np
import pandas as pd
from skimage import io, color, transform
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# ### (a) (10 points) Perform analysis on the Yale face dataset for Subject 1 and Subject 2, respectively, using all the images EXCEPT for the two pictures named subject01-test.gif and subject02-test.gif. Plot the first 6 eigenfaces for each subject. When visualizing, please reshape the eigenvectors into proper images. Please explain can you see any patterns in the top 6 eigenfaces?
# 

# In[2]:


p_1 = ["subject01.leftlight.gif","subject01.noglasses.gif",
                 "subject01.normal.gif","subject01.rightlight.gif",
                 "subject01.sad.gif","subject01.sleepy.gif",
                 "subject01.surprised.gif","subject01.wink.gif"]
lst_1 = []
downsampling_factor = (4, 4)
for ip in p_1:
    i = io.imread(ip)
    i_downsampled = transform.downscale_local_mean(i, downsampling_factor)
    lst_1.append(i_downsampled.ravel())
s01 = np.array(lst_1)
m,n = s01.shape
pca_1 = PCA(n_components= 6)
pca_1.fit(s01)
ef = pca_1.components_[:6]
plt.figure()  
fig, axes = plt.subplots(2,3,sharex=True,sharey=True)
axes[0][0].imshow(ef[0].reshape(61,80),cmap='gray')
axes[0][0].set_title('eigenfaces: 1', fontsize=10)

axes[0][1].imshow(ef[1].reshape(61,80),cmap='gray')
axes[0][1].set_title('eigenfaces: 2', fontsize=10)

axes[0][2].imshow(ef[2].reshape(61,80),cmap='gray')
axes[0][2].set_title('eigenfaces: 3', fontsize=10)

axes[1][0].imshow(ef[3].reshape(61,80),cmap='gray')
axes[1][0].set_title('eigenfaces: 4', fontsize=10)

axes[1][1].imshow(ef[4].reshape(61,80),cmap='gray')
axes[1][1].set_title('eigenfaces: 5', fontsize=10)

axes[1][2].imshow(ef[5].reshape(61,80),cmap='gray')
axes[1][2].set_title('eigenfaces: 6', fontsize=10)

plt.show()
plt.savefig('Q4-ef-s1.pdf')


# In[3]:


p_2 = ["subject02.glasses.gif","subject02.happy.gif",
                 "subject02.leftlight.gif","subject02.noglasses.gif",
                 "subject02.normal.gif","subject02.rightlight.gif",
                 "subject02.sad.gif","subject02.sleepy.gif",
                 "subject02.wink.gif"]
lst_2 = []
downsampling_factor = (4, 4)
for ip in p_2:
    i = io.imread(ip)
    i_downsampled = transform.downscale_local_mean(i, downsampling_factor)
    lst_2.append(i_downsampled.ravel())
s02 = np.array(lst_2)
pca_2 = PCA(n_components= 6)
pca_2.fit(s02)
ef2 = pca_2.components_[:6]
plt.figure()
fig, axes = plt.subplots(2,3,sharex=True,sharey=True)
axes[0][0].imshow(ef2[0].reshape(61,80),cmap='gray')
axes[0][0].set_title('eigenfaces: 1', fontsize=10)

axes[0][1].imshow(ef2[1].reshape(61,80),cmap='gray')
axes[0][1].set_title('eigenfaces: 2', fontsize=10)

axes[0][2].imshow(ef2[2].reshape(61,80),cmap='gray')
axes[0][2].set_title('eigenfaces: 3', fontsize=10)

axes[1][0].imshow(ef2[3].reshape(61,80),cmap='gray')
axes[1][0].set_title('eigenfaces: 4', fontsize=10)

axes[1][1].imshow(ef2[4].reshape(61,80),cmap='gray')
axes[1][1].set_title('eigenfaces: 5', fontsize=10)

axes[1][2].imshow(ef2[5].reshape(61,80),cmap='gray')
axes[1][2].set_title('eigenfaces: 6', fontsize=10)

plt.show()
plt.savefig('Q4-ef-s2.pdf')


# ### (b) (10 points)  Face recognition.

# In[4]:


t1_list = ["subject01-test.gif"]
lst_2 = []
downsampling_factor = (4, 4)
for ip in t1_list:
    i = io.imread(ip)
    i_downsampled = transform.downscale_local_mean(i, downsampling_factor)
    lst_2.append(i_downsampled.ravel())
t1 = np.array(lst_2)

t2_list = ["subject02-test.gif"]
lst_2 = []
downsampling_factor = (4, 4)
for ip in t2_list:
    i = io.imread(ip)
    i_downsampled = transform.downscale_local_mean(i, downsampling_factor)
    lst_2.append(i_downsampled.ravel())
t2 = np.array(lst_2)
test = [t1, t2]

eg_face_1 = ef[0].reshape(-1, 1)
eg_face_2 = ef2[0].reshape(-1, 1)
eg_face = [eg_face_1, eg_face_2]

W = np.zeros((2, 2))
for r in range(2):
    for e in range(2):
        a = test[e]
        b = np.outer(eg_face[r], eg_face[r])
        c = b * a
        W[r][e] = np.linalg.norm(a - c) ** 2
s_df = pd.DataFrame(W, index=['eg_face_1', 'eg_face_2'], columns=['test 1', 'test 2'])
s_df


# In[5]:


pd.set_option('display.float_format', lambda x: '%.3f' % x)
s_df


# In[8]:


plt.figure()  

fig, axes = plt.subplots(2,2,sharex=True,sharey=True)
axes[0][0].imshow(ef[0].reshape(61,80),cmap='gray')
axes[0][0].set_title('Subject01_eigenfaces: 1', fontsize=10)

axes[0][1].imshow(ef2[0].reshape(61,80),cmap='gray')
axes[0][1].set_title('Subject02_eigenfaces: 1', fontsize=10)

axes[1][0].imshow(t1.reshape(61,80),cmap='gray')
axes[1][0].set_title('Subject01_test', fontsize=10)

axes[1][1].imshow(t2.reshape(61,80),cmap='gray')
axes[1][1].set_title('Subject02_test', fontsize=10)
plt.show()
plt.savefig('Q4-eigenfaces-test_1_2.pdf')

