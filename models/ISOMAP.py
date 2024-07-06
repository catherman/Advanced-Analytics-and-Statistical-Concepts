#!/usr/bin/env python
# coding: utf-8

# Order of faces using ISOMAP 
# 
# This analysis aims to reproduce the ISOMAP algorithm results in the original paper for ISOMAP, J.B.
# Tenenbaum, V. de Silva, and J.C. Langford, Science 290 (2000) 2319-2323.
# 
# The file isomap.mat (or isomap.dat) contains 698 images, corresponding to different poses of the same
# face. Each image is given as a 64 × 64 luminosity map, hence represented as a vector in R4096. This vector
# is stored as a row in the file. (This is one of the datasets used in the original paper.) For this analysis, we
# will implement the ISOMAP algorithm by scratch. 
# 
# Using Euclidean distance (i.e., in this case, a distance in R4096) to construct the ϵ-ISOMAP.
# We will tune the ϵ parameter to achieve the most reasonable performance. (Note: this is different from K-ISOMAP, 
where each node has exactly K nearest neighbors.)
# 
# In the code that folllows, we will: 
#     (a) Visualize the nearest neighbor graph (with the adjacency matrix (e.g., as an image), 
# 
#     (b) Implement the ISOMAP algorithm (from scratch) to obtain a two-dimensional low-dimensiona embedding.
#         - visualize the embeddings using a scatter plot, 
#         - identify a few images in the embedding space (mark w/ red x) 
#         - show what these images look like and specify the face locations on the scatter plot. 
#         - observe whether there are any visual similarity among the images and their arrangement, 
#           similar to what you seen in the paper.
# 
#     (c) Perform PCA on the images and project them into the top 2 principal components. 
#         - visualize on a scatter plot. 
#         - Compare quality of projection using ISOMAP vs. PCA.  

import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.io import loadmat
import scipy.sparse.csgraph as csgraph
import matplotlib.gridspec as gridspec
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

data = loadmat('isomap.mat')['images']

def distance_processor (data, n):
    lp = n
    (d, n) = data.shape
    distance_matrix = np.zeros([n,n])
    if lp == 1:
        distance_matrix = pairwise_distances(data.T, data.T, metric='cityblock')
    else:
        distance_matrix = pairwise_distances(data.T, data.T, metric='euclidean')
    distance_matrix_list = np.sort(distance_matrix.reshape(-1))  
    ids = n + 40000
    epsilon = distance_matrix_list[ids]
    q = distance_matrix < epsilon
    q = q | q.T
    q = q.astype('float')
    G = distance_matrix * q + 99999.9 * (1 - q)
    return distance_matrix, d, n, G 
distance_matrix, d, n, G = distance_processor (data,1)


fig_graph = plt.figure(constrained_layout=True)  
gs_graph = gridspec.GridSpec(ncols=4, nrows=3, figure=fig_graph)
ax_graph = fig_graph.add_subplot(gs_graph[:, :3])  
ax_graph.imshow(G, cmap=plt.get_cmap('gray'), extent=[0, 698, 0, 698])
ax_graph.set_aspect('equal')
selected_random_img = np.array([200, 400, 500])  
ax_graph.scatter(selected_random_img, 698 - selected_random_img, marker='x', c='r')
img_graph1 = np.reshape(data[:, selected_random_img[0]], [64, -1]).T
img_graph2 = np.reshape(data[:, selected_random_img[1]], [64, -1]).T
img_graph3 = np.reshape(data[:, selected_random_img[2]], [64, -1]).T
ax_graph1 = fig_graph.add_subplot(gs_graph[0, 3])
ax_graph2 = fig_graph.add_subplot(gs_graph[1, 3])
ax_graph3 = fig_graph.add_subplot(gs_graph[2, 3])
ax_graph1.imshow(img_graph1, cmap=plt.get_cmap('gray'))
ax_graph1.axis('off')
ax_graph2.imshow(img_graph2, cmap=plt.get_cmap('gray'))
ax_graph2.axis('off')
ax_graph3.imshow(img_graph3, cmap=plt.get_cmap('gray'))
ax_graph3.axis('off')
lp = 1  #
fig_graph.suptitle('Adjacency Matrix Weighted, Lp{}'.format(lp))
fig_graph.savefig('Q3_Adjacency-Matrix-Weighted-Lp{}.pdf'.format(lp))
plt.show()  

def pairs_distances_matrix_C_processor ():
    d = csgraph.shortest_path(G)
    d = (d + d.T)/2
    ones = np.ones([n, 1])
    H = np.eye(n) - 1/n * ones.dot(ones.T)
    C = -H.dot(d**2).dot(H) / (2 * n)  
    C = (C + C.T) / 2
    eig_val, eig_vec = eigh(C)
    eig_val = eig_val.real  
    eig_vec = eig_vec.real
    eig_index = np.argsort(-eig_val)  
    Z = eig_vec[:,eig_index[0:2]].dot(np.diag(np.sqrt(eig_val[eig_index[0:2]])))
    return d, Z, C 
d, Z, C  = pairs_distances_matrix_C_processor ()

fig = plt.figure(constrained_layout=True)
gs = gridspec.GridSpec(ncols=4, nrows=3, figure=fig)

ax0 = fig.add_subplot(gs[:,:3])
index = np.argsort(Z[:,1])[4:7]
ax0.scatter(Z[:,0], Z[:,1], s = 5)
ax0.scatter(Z[index,0],Z[index,1], marker='x', c='red', label='selected image location')
ax0.set_title('LP{}: 2-d embedding'.format(lp))
ax0.legend(bbox_to_anchor=(0.5, -0.2), loc='center', borderaxespad=0.1)
ax0.set_aspect('equal')

img_1 = np.reshape(data[:,index[0]], [64, -1]).T
img_2 = np.reshape(data[:,index[1]], [64, -1]).T
img_3 = np.reshape(data[:, index[2]], [64, -1]).T
ax1 = fig.add_subplot(gs[0,3])
ax2 = fig.add_subplot(gs[1,3])
ax3 = fig.add_subplot(gs[2,3])
ax1.imshow(eval('img_'+str(1)), cmap=plt.get_cmap('gray'))
ax1.axis('off')
ax2.imshow(eval('img_'+str(2)), cmap=plt.get_cmap('gray'))
ax2.axis('off')
ax3.imshow(eval('img_'+str(3)), cmap=plt.get_cmap('gray'))
ax3.axis('off')
fig.savefig('Q3_lp{}-2-d embedding.pdf'.format(lp))

fig, ax = plt.subplots()
ax.scatter(Z[:, 0], Z[:, 1], s=5)
ax.set_aspect('equal')

for ff in range(n):
    img_data = data[:, ff].reshape(64, 64).T
    im = Image.fromarray(np.uint8(img_data * 255), mode='L')
    im.thumbnail((6.4, 6.4))  # Resize the image to fit the plot
    imbox = OffsetImage(im, zoom=1.0, cmap='gray')
    ab = AnnotationBbox(imbox, (Z[ff, 0], Z[ff, 1]), frameon=False)
    ax.add_artist(ab)

ax.set_title('LP{}: 2-d manifold, All images'.format(lp))
plt.savefig('Q3_2-d manifold, All images-lp{}.pdf'.format(lp), dpi=300)
plt.show()

def PCA_processor (data):
    mu = np.mean(data, axis=1, keepdims=True)
    d_mean = data - mu
    covariance_matrix = np.cov(d_mean)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]
    eigenvalues = eigenvalues[sorted_indices]
    top_eigenvectors = eigenvectors[:, :2]
    Zpca = np.dot(d_mean.T, top_eigenvectors)
    return Zpca

Zpca = PCA_processor (data)
Zpca2=(Zpca*-1)

figpca, axpca = plt.subplots()
axpca.scatter(Zpca2[:, 0], Zpca2[:, 1], s=8)
axpca.set_aspect('equal')

for ff in range(n):
    img = Image.fromarray((data[:, ff].reshape(64, 64) * 255).astype('uint8'), 'L')
    img = OffsetImage(img, cmap='gray', zoom=0.1)
    axpca.add_artist(AnnotationBbox(img, (Zpca2[ff, 0], Zpca2[ff, 1]), frameon=False))
axpca.set_title('PCA: First 2 principal components, all images')
figpca.savefig('Q3_all-images-pca.pdf', dpi=300)

