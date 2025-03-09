import numpy as np

def integrateTrapezoid(start, end, N, func):

    dt = (end - start) / N
    ans = 0

    for i in range(N):
        ans += ((func(start + i * dt) + func(start + (i + 1) * dt)) * dt) / 2

    return ans

def loss(start, end, N, f):
    dt = (end -start) / N
    ans = 0

    for i in range(N):
        ans += np.log(f(start + (i + 1) * dt))

    ans -= integrateTrapezoid(start, end, N, f)

    return ans