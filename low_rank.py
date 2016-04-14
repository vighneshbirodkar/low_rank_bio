import numpy as np
from numpy.linalg import norm
from matplotlib import pyplot as plt
from mypca import mypcp


M = np.loadtxt('homo_sapiens.csv', delimiter=',')
L, S, _, _ = mypcp(M)

s_data = np.linalg.svd(M, compute_uv=False)
s_l = np.linalg.svd(L, compute_uv=False)

rank_data = np.linalg.matrix_rank(M)
rank_l = np.linalg.matrix_rank(L)

plt.semilogy(s_data,
             label='Singular Values of Data (rank=%d)' % rank_data)
plt.semilogy(s_l,
             label='Singular Values of Low rank component (rank=%d)' % rank_l)
plt.legend()
plt.grid(True)
print('Percentage Error = %f', 100*norm(M - L, 'fro')/norm(M))

plt.show()
