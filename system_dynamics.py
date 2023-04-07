import numpy as np
from matrices import skew
from matrices import quaternion_product, vect2quat, cross_product

def system(state, t, Q, params):
    m, g = params['m'], params['g']

    rc, v, q, w = state[:3], state[3:6], state[6:10], state[10:13]    

    # equation 1
    dr = v

    # equation 2
    Q_v = Q[:3]
    dv = m ** (-1) * Q_v - np.array([0, 0, -g])

    # equation 3
    dq = -1/2 * quaternion_product(vect2quat(w), q)

    # equation 4
    Im = np.array(params['I'])
    Im_inv = np.linalg.inv(Im)

    tau = Q[3:]
    tau_prime = tau - cross_product(w, Im @ w)
    dw = Im_inv @ tau_prime

    dstate = np.concatenate([dr, dv, dq, dw])
    return dstate


def L(w):
    L1 = np.diag(w)
    L2 = np.array([[w[1], w[2], 0],
                   [w[0], 0, w[2]],
                   [0, w[0], w[1]]
                  ])
    return np.hstack([L1, L2])


# def regressor(w, dw, dv, sys_params):
#     # Y(q, q_dot, qs_dot, qs_ddot)
#     g = np.array([0, 0, (-1) * sys_params['g']])
#     Y = np.zeros((6,7))
#     Y[:3, 0] = dv + g
#     Y[3:, 1:] = L(dw) + skew(w) @ L(w)
#     return Y

def regressor(q, q_dot, q_ddot, sys_params):
    # Y(q, q_dot, qs_dot, qs_ddot)
    g = np.array([0, 0, (-1) * sys_params['g']])
    Y = np.zeros((6,7))
    dv = q_ddot[:3]
    dw = q_ddot[3:]
    w = q_dot[:3]
    Y[:3, 0] = dv + g
    Y[3:, 1:] = L(dw) + skew(w) @ L(w)
    return Y


def p(params):
    m = params['m']
    I = params['I']
    return np.array([m, I[0][0], I[1][1], I[2][2], I[0][1], I[0][2], I[1][2]])
