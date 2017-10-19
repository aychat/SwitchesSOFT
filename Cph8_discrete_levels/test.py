import numpy as np

def f(w1, g1, w2, g2):
    return w1 * w2

molecules = [
    dict(
        w1=np.random.rand(),
        g1=np.random.rand(),
        w2=np.random.rand(),
        g2=np.random.rand(),
    ) for _ in range(3)
]