from scipy import linalg, spatial
from functools import reduce
import numpy as np
import math

# given term x doc matrix
a = np.array([[2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [1, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 2, 1, 2, 1, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 2, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [0, 0, 0, 0, 1, 1, 2, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 2, 1, 1, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 1, 3, 2, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 2, 2, 3, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1, 0],
    [0, 0, 1, 0, 0, 0, 2, 2, 3, 0, 1, 0],
    [1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 2]]) # 11 x 12

# transposing a to easily calculate column l2 norm
a_trans = a.transpose()

# a_normalised = [[x/] for row in a] for ]

a_trans_norm = np.array([[element/math.sqrt(row[0]**2 + row[1]**2 + row[2]**2 + row[3]**2 + row[4]**2 + row[5]**2 + row[6]**2 + row[7]**2 + row[8]**2 + row[9]**2 + row[10]**2) for element in row] for row in a_trans])

a_norm = a_trans_norm.transpose() # 11 x 12

(u, s, vh) = np.linalg.svd(a_norm)

print("Singular values matrix sigma = ")
print(s)

singular_values_square_sum = reduce(lambda x,y: x + y**2, s)
print("Sum of squeres of singular values = ", singular_values_square_sum)
print("90 % Sum of squeres of singular values = ", 0.9*singular_values_square_sum)

sum = 0
k = 0
for index, val in enumerate(s):
    sum = sum + val**2
    if (sum <= 0.9 * singular_values_square_sum):
        k = index + 1
    else:
        break

print("Singular values to retain (k) = ", k)

# based on the values of s, we can set k = 2 since only the first 2 values are big enough to have significant influence
u_trunc = np.array([row[0:k] for row in u])
u_trunc_trans = u_trunc.transpose() # 5 x 11

q = np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0])
q_trans = q.transpose() # 11 x 1

q_prime = np.matmul(u_trunc_trans, q_trans)

doc_similarities = []
for index, doc in enumerate(a_trans):
    doc_trans = np.array(doc).transpose()
    doc_prime = np.matmul(u_trunc_trans, doc_trans)
    doc_similarities.append((index, 1 - spatial.distance.cosine(doc_prime, q_prime)))

doc_similarities_sorted = sorted(doc_similarities, key = lambda tup: tup[1], reverse=True)[:3]
print("Document rankings (document, similarity score) = ")
print(doc_similarities_sorted)

# Rank    Doc
# 1       5
# 2       4
# 3       7

# gene = [1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0]
# gen_norm = np.array([element/math.sqrt(gene[0]**2 + gene[2]**2 + gene[3]**2 + gene[4]**2 + gene[5]**2 + gene[6]**2 + gene[7]**2 + gene[8]**2 + gene[9]**2 + gene[10]**2 + gene[11]**2) for element in gene])

# gene_prime = np.matmul(u_trunc_trans, gene_norm)
