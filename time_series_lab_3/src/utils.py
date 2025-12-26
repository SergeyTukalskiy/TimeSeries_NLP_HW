import numpy as np

def train_val_test_split(y, train_size=0.7, val_size=0.15):
    n = len(y)
    t1 = int(n * train_size)
    t2 = int(n * (train_size + val_size))
    return y.iloc[:t1], y.iloc[t1:t2], y.iloc[t2:]

def invert_boxcox(y, lambda_bc):
    if lambda_bc == 0:
        return np.exp(y)
    return np.exp(np.log(lambda_bc * y + 1) / lambda_bc)
