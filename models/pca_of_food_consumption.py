#!/usr/bin/env python
# coding: utf-8

# 2. PCA: Food consumption in European countries [20 points].
# 
# The data food-consumption.csv contains 16 countries in Europe and their consumption for 20 food items, such as tea, jam, coffee, yogurt, and others. We will perform principal component analysis to explore the data. In this question, please implement PCA by writing your own code (you can use any basic packages,
# such as numerical linear algebra, reading data, in your file).
# 
# First, we will perform PCA analysis on the data by treating each country’s food consumption as their
# “feature” vectors. In other words, we will find weight vectors to combine 20 food-item consumptions for
# each country.
# 
#     (a) (10 points) For this problem of performing PCA on countries by treating each country’s food consumption
#     as their “feature” vectors, explain how the data matrix is set-up in this case (e.g., the columns and
#     the rows of the matrix correspond to what). Now extract the first two principal components for each
#     data point (thus, this means we will represent each data point using a two-dimensional vector). Draw a
#     scatter plot of two-dimensional representations of the countries using their two principal components.
#     Mark the countries on the plot (you can do this by hand if you want). Please explain any pattern you
#     observe in the scatter plot.
#     
# 
#     (b) (10 points) Now, we will perform PCA analysis on the data by treating country consumptions as “feature” vectors for each food item. In other words, we will now find weight vectors to combine country consumptions for each food item to perform PCA another way. Project data to obtain their two principle components (thus, again each data point – for each food item – can be represented using a two-dimensional vector). Draw a scatter plot of food items. Mark the food items on the plot (you can do this by hand if you want). Please explain any pattern you observe in the scatter plot.
#     
#     

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('food-consumption.csv')
countries = df['Country']
foods = df.columns[1:]
data = df.iloc[:, 1:].values
m, n = np.shape(data)
k = 2
mu = np.mean(data, axis=0)
d_mean = data - mu
C = np.dot(d_mean.T, d_mean) / m
eigenvalues, eigenvectors = np.linalg.eigh(C)
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]
selected_eigenvectors = eigenvectors[:, :k]
pc = np.dot(d_mean, selected_eigenvectors)
norms = np.linalg.norm(pc, axis=0)
pc /= norms
pc[:, 1] *= -1
plt.figure()
plt.scatter(pc[:, 0], pc[:, 1], s=20, facecolors='none', edgecolors='r')
plt.axis('equal')
for i in range(len(countries)):
    plt.annotate(countries[i], (pc[i, 0], pc[i, 1]), fontsize=8)
plt.grid()
plt.title('PC plot for each country')
plt.show()
plt.savefig('Q2-country-pca.pdf')



# In[3]:


# Transpose the data 
data1 = data.T
mu1 = np.mean(data1, axis=0)
d_mean1 = data1 - mu1
C1 = np.dot(d_mean1.T, d_mean1) / n
eigenvalues1, eigenvectors1 = np.linalg.eigh(C1)
sorted_indices1 = np.argsort(eigenvalues1)[::-1]
eigenvalues1 = eigenvalues1[sorted_indices1]
eigenvectors1 = eigenvectors1[:, sorted_indices1]
selected_eigenvectors1 = eigenvectors1[:, :k]
pc1 = np.dot(d_mean1, selected_eigenvectors1)
norms = np.linalg.norm(pc1, axis=0) 
pc1 /= norms
pc1b=(pc1*-1)
plt.figure()
plt.scatter(pc1b[:, 0], pc1b[:, 1], s=20, facecolors='none', edgecolors='r')
plt.axis('equal')
for i in range(len(foods)):
    plt.annotate(foods[i], (pc1b[i, 0], pc1b[i, 1]), fontsize=8)
plt.grid()
plt.title('PC plot for each food')
plt.show()
plt.savefig('Q2-food-pca.pdf')

