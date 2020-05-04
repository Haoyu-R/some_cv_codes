import numpy as np

X = np.zeros([2, 1]).reshape((2, 1))
P = np.array([[1000, 0], [0, 1000]])
U = np.zeros([2, 1]).reshape((2, 1))
F = np.array([[1, 1], [0, 1]])
H = np.array([1, 0]).reshape((1, 2))
R = np.array([1])
I = np.identity(2)


measurements = [1, 2, 3]

for i in measurements:
    Z = np.array([i])
    y = Z - (H @ X)
    S = H @ P @ H.T + R
    K = P @ H.T * np.linalg.inv(S)
    # K = K.reshape((2, 1))
    X = X + (K * y)

    P = (I - (K @ H)) @ P

    # print("X: ", X)
    # print()
    # print("P: ", P)
    # print()

    X = (F @ X) + U
    P = F @ P @ F.T
    print("X: ", X)
    print()
    print("P: ", P)
    print()



