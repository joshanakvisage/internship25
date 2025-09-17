import numpy as np

#define a multivariate function 
def gfunc(x, y, v):
    f1 = x * y
    f2 = x + y

    f = np.vstack((f1, f2))  
    g = f + v  
    return g, f

#x, y, v
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([0, 1, 2, 3, 4, 5])
v = 0.17

#function outputs
g, f = gfunc(x, y, v)
u = np.vstack((x, y))  


data = np.vstack((f, u))  

#covariance matrix over all variables
cov_matrix = np.cov(data)

#extract block matrices
Σ_ff = cov_matrix[0:2, 0:2]     
Σ_fu = cov_matrix[0:2, 2:4]      
Σ_uf = Σ_fu.T                  
Σ_uu = cov_matrix[2:4, 2:4]    

#invert Σ_uu 
try:
    Σ_uu_inv = np.linalg.inv(Σ_uu)
except np.linalg.LinAlgError:
    Σ_uu_inv = np.linalg.pinv(Σ_uu)

#conditional covariance
Σ_cond = Σ_ff - Σ_fu @ Σ_uu_inv @ Σ_uf  

#weight matrix W (in this case an indentityy matrix)
W = np.eye(2)

M = np.sqrt(np.trace(W @ Σ_cond))
Mnom = M / np.sqrt(np.var(g.flatten()))

print("M:", M)
print("Mnom:", Mnom)
