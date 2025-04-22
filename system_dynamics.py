import numpy as np
from matrices import skew
from matrices import quaternion_product, vect2quat, cross_product
import random
def disturbance(t):
    return 70* np.sin(10 * t), 0* 10 * np.cos(5 * t)
    # return np.random.normal(0, 0.005), np.random.normal(0, 0.005)
    # return 10*random.gauss(0.5, 0.01), 10*random.gauss(0.05, 0.01)

def system(state, t, Q, params):
    m, g = params['m'], params['g']

    rc, v, q, w = state[:3], state[3:6], state[6:10], state[10:13]    

    # equation 1
    dr = v
    f_disturbance, tau_disturbance = disturbance(t)
    # equation 2
    Q_v = Q[:3]
    dv = m ** (-1) * (Q_v + f_disturbance) - np.array([0, 0, -g])
    # equation 3
    dq = -1/2 * quaternion_product(vect2quat(w), q)

    # equation 4
    Im = np.array(params['I'])
    Im_inv = np.linalg.inv(Im)

    tau = Q[3:]
    tau_prime = tau + tau_disturbance - cross_product(w, Im @ w)
    # print(tau_disturbance)
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



# print(np.random.normal(0, 7.05), np.random.normal(0, 10.05))
# t = 1/2
# print(7* np.sin(10 * t), 10 * np.cos(5 * t))