import numpy as np

#RSME for one dimension
def rmse(truevalue, estimatedvalue):
    truevalue = np.array(truevalue)
    estimatedvalue = np.array(estimatedvalue)
    mse = np.mean((truevalue - estimatedvalue)**2)
    return np.sqrt(mse)

#RSME for two dimensions
def rmsexandy(truex, truey, estix, estiy):
    gt = np.vstack((truex, truey)).T
    est = np.vstack((estix, estiy)).T
    mse = np.mean((gt - est)**2)
    return np.sqrt(mse)

#Error at each step
def error_step(truex, turey, estix, estiy):
    gtx = np.array(truex)
    gty = np.array(turey)
    estx = np.array(estix)
    esty = np.array(estiy)

    errorx = gtx - estx
    abserrorx = np.abs(errorx)
    errory = gty - esty
    abserrory = np.abs(errory)

    return errorx, abserrorx, errory, abserrory

