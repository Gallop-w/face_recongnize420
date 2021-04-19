import numpy as np

def drop_out(x, level):
    if level < 0. or level >= 1:
        raise ValueError('Dropout level must be in interval [0, 1]')
    retain_prob = 1. - level

    random_tensor = np.random.binomial(n=1, p=retain_prob, size=x.shape)
    print(random_tensor)

    x *= random_tensor
    print(x)
    x /= retain_prob

    return x

x = np.asarray([1,2,3,4,5,6,7,8,9,10], dtype=np.float32)
drop_out(x, 0.1)