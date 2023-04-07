import numpy as np
from matrices import quaternion_product, conjugate
from system_dynamics import regressor, p
from scipy.optimize import linprog

def attitude_pos_control(state, t, state_d, sys_params, ctrl_params):
    Kp_w = ctrl_params['K1']
    Kd_w = ctrl_params['K2']
    
    Kp_v = ctrl_params['K3']
    Kd_v = ctrl_params['K4']

    r_actual, v_actual, quat_actual, w_actual = state[:3], state[3:6], state[6:10], state[10:13]
    r_des, v_des, quat_des, w_des = state_d[:3], state_d[3:6], state_d[6:10], state_d[10:13]
    dr, dv = r_des - r_actual, v_des - v_actual
    dw = w_des - w_actual
    
    q_err = quaternion_product(quat_des, conjugate(quat_actual))
    if q_err[0] < 0:
        q_err_axis = np.array(q_err[1:])
    else:
        q_err_axis = -np.array(q_err[1:])

    a_w = Kp_w @ q_err_axis + Kd_w @ dw
    a_v = Kp_v @ dr + Kd_v @ dv
    q_ddot = np.hstack([a_v, a_w])
    Y = regressor(0, w_actual, q_ddot, sys_params)
    param_vector = p(sys_params)

    return Y @ param_vector


def robust_sliding_mode(state, t, state_d, sys_params, ctrl_params):
    Kp_w = ctrl_params['K1']
    Kd_w = ctrl_params['K2']
    
    Kp_v = ctrl_params['K3']
    Kd_v = ctrl_params['K4']
    Lambda = ctrl_params['Lambda']

    # s, dqs, ddqs = sliding_variables_slotine(state,state_d, Lambda)
    s, dqs, ddqs = sliding_variables_prop(state,state_d, Lambda)
    Y = regressor(0, dqs, ddqs, sys_params)

    K = np.block([[Kp_v, np.zeros((3,3))], 
                  [np.zeros((3,3)), Kp_w]])

    param_vector = p(sys_params)*.9
    delta_p = delta_p_classic(Y.T @ s)
    # delta_p = delta_p_lp(Y.T @ q_star)
    # print('delta norm ', np.linalg.norm(delta_p))

    return Y @ (param_vector+delta_p) + K @ s


def sgn(z):
    if z >=0:
        r = 1
    else:
        r = -1
    return r


def delta_p_classic(z):
    eps = 0.9 # reduces chattering by enlarging the value
    rho = 0.03 # uncertainty bound
    if np.linalg.norm(z) > eps:
        delta_p = rho * z / np.linalg.norm(z)
    else:
        delta_p = rho * z / eps
    return delta_p


def delta_p_lp(z):
    p1_min = -0.01
    p1_max = 0.01

    p2_min = -0.007
    p2_max = 0.007

    p1_bounds = [p1_min, p1_max]
    p2_bounds = [p2_min, p2_max]

    zp = -linprog(z,
                 bounds=[p1_bounds, p2_bounds,
                         p1_bounds, p2_bounds,
                         p1_bounds, p2_bounds,
                         p1_bounds]
                 ).x  # minimizes
    return zp


def sliding_variables_slotine(state, state_d, Lambda):
    r_actual, v_actual, quat_actual, w_actual = state[:3], state[3:6], state[6:10], state[10:13]
    r_des, v_des, quat_des, w_des = state_d[:3], state_d[3:6], state_d[6:10], state_d[10:13]
    
    r_wave, v_wave = r_des - r_actual, v_des - v_actual
    w_wave = w_des - w_actual

    # quaternion error = quaternion desires [*] conjugate(quaternion actual)
    quat_wave = quaternion_product(quat_des, conjugate(quat_actual))

    # derivative of quaternion error = 0.5 * quaternion error [*] [0, dw]^T
    dquat_wave = 0.5 * quaternion_product(quat_wave, np.hstack([[0], w_wave]))

    q_wave = np.hstack([r_wave, -sgn(quat_wave[0])*quat_wave[1:]])
    dq_wave = np.hstack([v_wave, sgn(quat_wave[0])*dquat_wave[1:]])
    dqs = np.hstack([v_des, w_des]) + Lambda @ q_wave
    ddqs = np.zeros(6) + Lambda @ dq_wave
    s = dq_wave + Lambda @ q_wave 
    return s, dqs, ddqs


def sliding_variables_prop(state, state_d, Lambda):
    r_actual, v_actual, quat_actual, w_actual = state[:3], state[3:6], state[6:10], state[10:13]
    r_des, v_des, quat_des, w_des = state_d[:3], state_d[3:6], state_d[6:10], state_d[10:13]
    
    r_wave, v_wave = r_des - r_actual, v_des - v_actual
    w_wave = w_des - w_actual

    # quaternion error = quaternion desires [*] conjugate(quaternion actual)
    quat_wave = quaternion_product(quat_des, conjugate(quat_actual))
    if quat_wave[0] < 0:
        quat_wave_axis = np.array(quat_wave[1:])
    else:
        quat_wave_axis = -np.array(quat_wave[1:])

    # derivative of quaternion error = 0.5 * quaternion error [*] [0, dw]^T
    dquat_wave = 0.5 * quaternion_product(quat_wave, np.hstack([[0], w_wave]))

    q_wave = np.hstack([r_wave, quat_wave_axis])
    dq_wave = np.hstack([v_wave, dquat_wave[1:]])
    dqs = np.hstack([v_des, w_des]) + Lambda @ q_wave
    ddqs = np.zeros(6) + Lambda @ dq_wave
    s = dq_wave + Lambda @ q_wave 
    return s, dqs, ddqs

