import numpy as np


#dse_jk = DSEJK(d_ao=np.eye(2))
#dse_jk.C_left_add(np.array([[1.0], [0.0]]))
#dse_jk.C_right_add(np.array([[0.0], [1.0]]))
d = np.eye(2)
C_L = np.array([[1.0], [0.0]])
C_R = np.array([[0.0], [1.0]])

# |L> = 1|0> + 0|1>
# |R> = 0|0> + 1|1>

# d = (1 0)
#     (0 1)

# J = oe.contract("pq,rs,rs->pq", self.d_ao, self.d_ao, D, optimize="optimal")
# K = oe.contract("pr,qs,rs->pq", self.d_ao, self.d_ao, D, optimize="optimal")
#D = np.ascontiguousarray(C_left @ C_right_i.T)
D = C_L @ C_R.T

J_l = np.zeros((2,2))
K_l = np.zeros((2,2))
for mu in range(2):
    for nu in range(2):
        sum_J = 0
        sum_K = 0
        for lam in range(2):
            for sig in range(2):
                sum_J += d[mu, nu] * d[lam, sig] * D[lam, sig]
                sum_K += d[mu, sig] * d[lam, nu] * D[lam, sig]
        J_l[mu, nu] = sum_J
        K_l[mu, nu] = sum_K

J = np.einsum("mn,ls,ls->mn", d, d, D)
K = np.einsum("ms,ln,ls->mn", d, d, D)

print(F"D is \n {D}")
print(F"d is \n {d}")
print(F"J is \n {J}")
print(F"K is \n {K}")
print(F"J_l is \n {J_l}")
print(F"K_l is \n {K_l}")