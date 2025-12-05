def dim_1d_check(y):
    """Reshape (n,1) 2D array to (n) 1D array else do nothing"""
    if not isinstance(y, type(None)):
        if not isinstance(y, tuple):
            if len(y.shape) == 2:
                if y.shape[1] == 1:
                    return y.reshape(-1)
        else:
            return (dim_1d_check(y[0]), dim_1d_check(y[1]))
    return y
