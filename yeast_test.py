import numpy as np
from mypca import mypcp
from numpy.linalg import norm, svd, matrix_rank
import matplotlib
matplotlib.use('AGG')

from matplotlib import pyplot as plt


M = np.loadtxt('expression_SC_superfinal.tsv', skiprows=1,
               usecols=range(1, 2578))
L, S, _, _ = mypcp(M)

error = 100*(norm(M - L, 'fro')/norm(M, 'fro'))
plt.semilogy(svd(M, compute_uv=False), label='Data')
plt.semilogy(svd(M, compute_uv=False), label='Low Rank')
print('Error = %f' % error)
print('Rank Data = %d' % matrix_rank(M))
print('Rank L = %d' % matrix_rank(L))

plt.save('plot.png')
