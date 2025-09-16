import numpy as np

# Define the function
def gfunc(x, y, v):
    func = x * y
    gfunc = func + v
    return gfunc, func

# Define inputs
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([0, 1, 2, 3, 4, 5])
v = 0.17

# Call the function and capture outputs
g, f = gfunc(x, y, v)

# Stack f, x, y into one 2D array (rows=variables, cols=samples)
data = np.vstack((f, x, y))

# Calculate covariance matrix of all three variables
cov_matrix = np.cov(data)

# Extract components from covariance matrix
Eff = cov_matrix[0, 0]          # variance of f
Efu = cov_matrix[0, 1:].reshape(1, 2)   # covariance of f with [x, y], shape (1,2)
Euf = Efu.T                    # transpose, shape (2,1)
Euu = cov_matrix[1:, 1:]       # covariance matrix of [x, y], shape (2,2)

# Invert Euu safely (use pseudo-inverse if singular)
try:
    Euu_inv = np.linalg.inv(Euu)
except np.linalg.LinAlgError:
    Euu_inv = np.linalg.pinv(Euu)

# Calculate the quantity inside the sqrt
inside = Eff - (Efu @ Euu_inv @ Euf).item()

# Make sure inside sqrt is non-negative
if inside < 0:
    inside = 0

M = np.sqrt(inside)

# Normalize by sqrt of variance of g
Mnom = M / np.sqrt(np.var(g))

print("Mnom:", Mnom)


