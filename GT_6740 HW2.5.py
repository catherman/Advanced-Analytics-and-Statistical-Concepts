#!/usr/bin/env python
# coding: utf-8

# ### 5. To subtract or not to subtract, that is the question [10 points].
# 
# We proved in 1.1, above, that the first principle component direction v corresponds to the largest eigenvector of the the sample covariance matrix, $w^{1}$, written as $z^{i}$:
# 
# (a)
# 
# $$z^{i} = 𝑤^{1^{T}}(𝑥^{i}-\mu)/\sqrt{\lambda_{1}}$$
# Or:
# $$z^{1} = 𝑤^{1^{T}}(𝑥^{1}-\mu)/\sqrt{\lambda_{1}}$$
# 
# $$z^{1} = 𝑤^{T}(𝑥^{1}-\mu)/\sqrt{\lambda_{1}}=5$$
# 
# 
# Now suppose Prof. X insists not subtracting the mean.  Following the steps illustrated in the answer to 1.1, above, the result to finding the largest eigenvector of the the sample covariance matrix, $w^{1}$, written as $z^{i}$: would be similar to (a), but without subtracting the mean. This would result in:
# 
# (b)
# $$z^{i} = 𝑤^{1^{T}}𝑥^{i}/\sqrt{\lambda_{1}}$$
# 
# $$z^{1} = 𝑤^{T}𝑥^{1}/\sqrt{\lambda_{1}}$$
# 
# Clearly, these two equations are not equal 
# 
# $$𝑤^{T}𝑥^{1}/\sqrt{\lambda_{1}} != 𝑤^{T}(𝑥^{1}-\mu)/\sqrt{\lambda_{1}}$$
# 
# Answer:  No, with and without subtracting the mean are not will result in equal eigenvectors.
# 
# Source: pca.pdf, Xie, Yao, Ph.D., Associate Professor, Georgia Institute of Technology 
# 
