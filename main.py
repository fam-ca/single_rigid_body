from matplotlib.pyplot import *
from scipy.integrate import odeint
import numpy as np
from matrices import quaternion_product, vect2quat, cross_product
from control import attitude_pos_control, robust_sliding_mode
from system_dynamics import system
from scipy.spatial.transform import Rotation

def desired_trajectory(t, traj_params):
    return traj_params['state_d']

def simulate_system(state_init, frequency, t_0, t_final, sys_params, ctrl_params, traj_params):
    dT = 1/frequency
    t = np.arange(t_0, t_final, dT)
    t_star = np.linspace(0, dT, 3)
    state_prev = state_init
    states, quats_d, angles = [], [], []
    poss_d = []
    U_pos = []
    for i in range(len(t)):
        t_curr = t[i]
        state_des = desired_trajectory(t_curr, traj_params)

        u_real = robust_sliding_mode(state_prev, t_curr, state_des, sys_params, ctrl_params)

        state = odeint(system, state_prev, t_star, args=(u_real, sys_params))

        state_prev = state[-1]

        quats_d.append(state_des[6:10])
        poss_d.append(state_des[:3])

        states.append(state_prev)

        quat_cur = state_prev[6:10]
        quat_ = np.array([quat_cur[3], quat_cur[0], quat_cur[1], quat_cur[2]])
        angle = Rotation.from_quat(quat_)
        angles.append(angle.as_euler('xyz'))

        U_pos.append(u_real)

    quats_d = np.array(quats_d)
    poss_d = np.array(poss_d)
    states = np.array(states)
    angles = np.array(angles)
    U_pos = np.array(U_pos)

    figure()
    text = ['$q_0$', '$q_1$', '$q_2$', '$q_3$']
    colors = ['r', 'g', 'b', 'k']
    styles = [':', '--', '-', '-.']
    for i in range(1,4):
        plot(t, states[:,i+6], color=str(colors[i]), linewidth=2.0, linestyle=str(styles[i]), label=str(text[i]))
        plot(t, quats_d[:, i], color=str(colors[i]), linewidth=1.0, linestyle=':')
    grid(color='black', linestyle='--', linewidth=0.7, alpha=0.7)
    xlim([t_0, t_final])
    ylabel(r'Quaternion ${q}$')
    xlabel(r'Time $t$ (s)')
    legend(loc='lower right')
    title('Quaternion')
    savefig('quaternion.png')

    figure()
    text = [r'$\phi$', r'$\theta$', r'$\psi$']
    for i in range(3):
        plot(t, angles[:,i], linewidth=2.0, label=str(text[i]))
    grid(color='black', linestyle='--', linewidth=0.7, alpha=0.7)
    xlim([t_0, t_final])
    ylabel(r'Angle $rad$')
    xlabel(r'Time $t$ (s)')
    legend(loc='lower right')
    title('Orientation')
    savefig('orientation.png')

    figure()
    text = ['$x_{real}$', '$y_{real}$', '$z_{real}$']
    text1 = ['$x_{with uncert}$', '$y_{with uncert}$', '$z_{with uncert}$']
    text2 = ['$x_{des}$', '$y_{des}$', '$z_{des}$']
    colors1 = ['orange', 'y', 'k']
    styles = ['-.', '--', '-']
    for i in range(1):
        i = 2
        plot(t, states[:, i], color=str(colors[i]), linewidth=2.0, linestyle=str(styles[i]), label=str(text[i]))
        plot(t, poss_d[:, i], color=str(colors[i]), linewidth=1.5, linestyle=':', label=str(text2[i]))
    grid(color='black', linestyle='--', linewidth=0.7, alpha=0.7)
    xlim([t_0, t_final])
    ylabel(r'Position ${p}$')
    xlabel(r'Time $t$ (s)')
    legend(loc='lower right')
    title('Position')
    savefig('position.png')

    figure()
    text = ['$v_x$', '$v_y$', '$v_z$']
    for i in range(3):
        plot(t, states[:, i+3], linewidth=2.0, label=str(text[i]))
    grid(color='black', linestyle='--', linewidth=0.7, alpha=0.7)
    xlim([t_0, t_final])
    ylabel(r'Velocity ${p}$')
    xlabel(r'Time $t$ (s)')
    legend(loc='lower right')
    title('Velocity')
    savefig('velocity.png')


    figure()
    text = ['$u_x$', '$u_y$', '$u_z$']
    plot(t, U_pos, linewidth=2.0, label=str(text[i]))
    grid(color='black', linestyle='--', linewidth=0.7, alpha=0.7)
    xlim([t_0, t_final])
    ylabel(r'Position control ${U}$')
    xlabel(r'Time $t$ (s)')
    legend(loc='lower right')
    title('Control')
    savefig('control.png')



t0 = 0
tf = 10.0
freq = 50 # 500
dT = 1/freq
g = 9.81

# Brick parameters
length = 0.5
width = 0.5
height = 0.2

m_real = 12

mat = np.array([[(width/2) ** 2 + (height/2) ** 2, 0.001, 0.001],
                    [0.001, (height/2) ** 2 + (length/2) ** 2, 0.001],
                    [0.001, 0.001, (length/2) ** 2 + (width/2) ** 2]]) / 3

I_real = m_real * mat

system_params = {'length': length,
                 'width': width,
                 'height': height,
                 'm': m_real,
                 'I': I_real,
                 'g': g,
                 }


m_modified = m_real * 0.8
I_modified = m_modified * mat

u_min = -500.0
u_max = 500.0

# gains for orientational components: Kp_w, Kd_w
K1 = np.diag([20, 20, 20])
K2 = np.diag([10, 10, 10])

# gains for positional components: Kp_v, Kd_v
K3 = np.diag([30, 30, 30])
K4 = np.diag([25, 25, 25])

K_pos = np.block([K1, K1])
K_att = np.block([K2, K2])

control_params = {'K1': K1, 'K2': K2,
                  'K3': K3, 'K4': K4,
                  'K_pos': K_pos, 'K_att': K_att,
                  'u_min': u_min, 'u_max': u_max,
                  'Lambda': np.eye(6)*25
                  }


# rot = Rotation.from_euler('xyz', [90, 45, 30], degrees=True)
# rot_quat = rot.as_quat()
# print(rot_quat)

# print(rot.as_euler('xyz'))

# rot = Rotation.from_quat(rot_quat)

# # Convert the rotation to Euler angles given the axes of rotation
# print(rot.as_euler('xyz'))

quat_0 = Rotation.from_euler('xyz', [15, 10, 5], degrees=True).as_quat()
quat_d = Rotation.from_euler('xyz', [0, 0, 0], degrees=True).as_quat()
print(quat_0, quat_d)


# initial position, velocity, quaternion, angular velocity
x0 = np.array([0., 1.0, 0.0,
      0, 0, 0,
      quat_0[3], quat_0[0], quat_0[1], quat_0[2], 
      0., 0., 0.])


trajectory_params = {'state_d': np.array([0.3, 0.7, 0.5,
                                 0., 0., 0.,
                                 1,0,0,0, 
                                 0., 0., 0.])}

simulate_system(state_init=x0, frequency=freq, t_0=t0, t_final=tf, 
                sys_params=system_params, ctrl_params=control_params, traj_params=trajectory_params)