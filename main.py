from matplotlib.pyplot import *
from scipy.integrate import odeint
import numpy as np
from matrices import quaternion_product, vect2quat, cross_product
from control import attitude_pos_control, robust_sliding_mode
from system_dynamics import system
from scipy.spatial.transform import Rotation
from visualize import visualize_rotation
import time

def quat2rot(q):
    q0, q1, q2, q3 = q
    # q = q0 + q1*i + q2*j + q3*k
    rotation_matrix = np.array(
        [[1 - 2 * q1 ** 2 - 2 * q2 ** 2, 2 * q0 * q1 - 2 * q2 * q3, 2 * q0 * q2 + 2 * q1 * q3],
         [2*q0*q1+2*q2*q3, 1 - 2 * q0 ** 2 - 2 *q2 ** 2, 2 * q1 * q2 - 2 * q0 * q3],
         [2 * q0 * q2 - 2 * q1 * q3, 2 * q2 * q1 + 2 * q0 * q3, 1 - 2 * q0 ** 2 - 2 * q1 ** 2]]
    )
    return rotation_matrix

def desired_trajectory(t, traj_params):
    return traj_params['state_d']

t1 = time.time()
def simulate_system(state_init, frequency, t_0, t_final, sys_params, ctrl_params, traj_params):
    dT = 1/frequency
    t = np.arange(t_0, t_final, dT)
    t_star = np.linspace(0, dT, 3)
    state_prev = state_init
    states, quats_d, angles = [], [], []
    poss_d = []
    U_pos = []
    rm = np.zeros((len(t),3,3))
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
        quat_ = np.array([quat_cur[0], quat_cur[1], quat_cur[2], quat_cur[3]]) #?
        # print(np.linalg.norm(quat_cur))
        angle = Rotation.from_quat(quat_)
        
        angles.append(angle.as_euler('xyz'))

        U_pos.append(u_real)

        rm[i,:,:] = quat2rot(quat_cur)

    quats_d = np.array(quats_d)
    poss_d = np.array(poss_d)
    states = np.array(states)
    angles = np.array(angles)
    U_pos = np.array(U_pos)

    figure()
    time_label = r'$t$, с'
    text = ['$q_0$', '$q_1$', '$q_2$', '$q_3$']
    colors = ['r', 'g', 'b', 'k']
    styles = [':', '--', '-', '-.']
    for i in range(0,4):
        plot(t, states[:,i+6], color=str(colors[i]), linewidth=2.0, linestyle=str(styles[i]), label=str(text[i]))
        plot(t, quats_d[:, i], color=str(colors[i]), linewidth=1.0, linestyle=':')
    grid(color='black', linestyle='--', linewidth=0.7, alpha=0.7)
    xlim([t_0, t_final])
    ylabel(r'Кватерион ${q}$')
    xlabel(time_label)
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
    for i in range(3):
        plot(t, states[:, i], color=str(colors[i]), linewidth=2.0, linestyle=str(styles[i]), label=str(text[i]))
        plot(t, poss_d[:, i], color=str(colors[i]), linewidth=1.5, linestyle=':', label=str(text2[i]))
    grid(color='black', linestyle='--', linewidth=0.7, alpha=0.7)
    xlim([t_0, t_final])
    ylabel(r'${p}$, м')
    xlabel(time_label)
    legend(loc='lower right')
    title('Position')
    savefig('position.png')

    figure()
    text = ['$v_x$', '$v_y$', '$v_z$']
    for i in range(3):
        plot(t, states[:, i+3], color=str(colors[i]), linewidth=2.0, linestyle=str(styles[i]), label=str(text[i]))
    grid(color='black', linestyle='--', linewidth=0.7, alpha=0.7)
    xlim([t_0, t_final])
    ylabel(r'${v}$, м/с')
    xlabel(time_label)
    legend(loc='lower right')
    title('Velocity')
    savefig('velocity.png')


    figure()
    text = ['$u_x$', '$u_y$', '$u_z$']
    # plot(t, U_pos, linewidth=2.0, label=str(text[i]))
    for i in range(3):   
        plot(t, U_pos[:,i], color=str(colors[i]), linewidth=2.0, linestyle=str(styles[i]), label=str(text[i]))
    grid(color='black', linestyle='--', linewidth=0.7, alpha=0.7)
    xlim([t_0, t_final])
    ylabel(r'${U}$')
    xlabel(time_label)
    legend(loc='lower right')
    title('Control')
    savefig('control.png')


    figure()
    text = ['$\delta x$', '$\delta y$', '$\delta z$']
    colors1 = ['orange', 'y', 'k']
    styles = ['-.', '--', '-']
    for i in range(3):
        err_i = states[:, i]-poss_d[:, i]
        plot(t, err_i, color=str(colors[i]), linewidth=2.0, linestyle=str(styles[i]), label=str(text[i]))
    grid(color='black', linestyle='--', linewidth=0.7, alpha=0.7)
    xlim([t_0, t_final])
    ylabel(r'$\delta r$, м')
    xlabel(time_label)
    legend(loc='lower right')
    title('Error position')
    savefig('error_position.png')


    fig2, axis2 = subplots(2, 3)
    text = ['$u_0$', '$u_1$', '$u_2$', '$u_3$', '$u_4$', '$u_5$']
    
    for k in range(3):
        i = 0
        axis2[i, k].plot(t, U_pos[:,i+k],color = 'b', label=text[i+k]+', Н*м')    
        axis2[i, k].grid()
        axis2[i, k].set_xlim([t0, 3])
        axis2[i, k].legend()
        
        i=1
        axis2[i, k].plot(t, U_pos[:,i+k+2], color='g', label=text[i+k+2]+', Н*м')    
        axis2[i, k].grid()
        axis2[i, k].set_xlim([t0, 3])
        axis2[i, k].legend()
        
    savefig('control_detailed.png')

    # axis2[1].plot(t,Us1[:,1],label='gamma='+str(control_params['gamma']))
    # axis2[1].plot(t,Us2[:,1],label='gamma='+str(control_params2['gamma']))
    # axis2[1].plot(t,Us3[:,1],label='gamma='+str(control_params3['gamma']))
    # axis2[1].plot(t,Us4[:,1],label='gamma='+str(control_params4['gamma']))
    # axis2[1].grid()
    # axis2[1].set_xlim([t0, tf])
    # axis2[1].legend()
    # axis2[1].set_title(r'$U_1$')
    # savefig('pics/exp2_us01.png')


    # fig3, axis3 = subplots(1, 2)
    # axis3[0].plot(t,Us1[:,2],label='gamma='+str(control_params['gamma']))
    # axis3[0].plot(t,Us2[:,2],label='gamma='+str(control_params2['gamma']))
    # axis3[0].plot(t,Us3[:,2],label='gamma='+str(control_params3['gamma']))
    # axis3[0].plot(t,Us4[:,2],label='gamma='+str(control_params4['gamma']))
    # axis3[0].grid()
    # axis3[0].set_xlim([t0, tf])
    # axis3[0].legend()
    # axis3[0].set_title(r'$U_2$')

    # axis3[1].plot(t,Us1[:,3],label='gamma='+str(control_params['gamma']))
    # axis3[1].plot(t,Us2[:,3],label='gamma='+str(control_params2['gamma']))
    # axis3[1].plot(t,Us3[:,3],label='gamma='+str(control_params3['gamma']))
    # axis3[1].plot(t,Us4[:,3],label='gamma='+str(control_params4['gamma']))
    # axis3[1].grid()
    # axis3[1].set_xlim([t0, tf])
    # axis3[1].legend()
    # axis3[1].set_title(r'$U_3$')
    # savefig('pics/exp2_us23.png')


    # fig4, axis4 = subplots(1, 2)
    # axis4[0].plot(t,Us1[:,4],label='gamma='+str(control_params['gamma']))
    # axis4[0].plot(t,Us2[:,4],label='gamma='+str(control_params2['gamma']))
    # axis4[0].plot(t,Us3[:,4],label='gamma='+str(control_params3['gamma']))
    # axis4[0].plot(t,Us4[:,4],label='gamma='+str(control_params4['gamma']))
    # axis4[0].grid()
    # axis4[0].set_xlim([t0, tf])
    # axis4[0].legend()
    # axis4[0].set_title(r'$U_4$')

    # axis4[1].plot(t,Us1[:,5],label='gamma='+str(control_params['gamma']))
    # axis4[1].plot(t,Us2[:,5],label='gamma='+str(control_params2['gamma']))
    # axis4[1].plot(t,Us3[:,5],label='gamma='+str(control_params3['gamma']))
    # axis4[1].plot(t,Us4[:,5],label='gamma='+str(control_params4['gamma']))
    # axis4[1].grid()
    # axis4[1].set_xlim([t0, tf])
    # axis4[1].legend()
    # axis4[1].set_title(r'$U_5$')
    # savefig('pics/exp2_us45.png')

    close("all")

    # visualize_rotation(rm)
    return states, angles

t2 = time.time()

print(t2-t1)

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

m_modified = m_real * 0.8
# print(m_real, m_modified)
I_modified = m_modified * mat


system_params = {'length': length,
                 'width': width,
                 'height': height,
                 'm': m_modified,
                 'I': I_modified,
                 'g': g,
                 }




u_min = -500.0
u_max = 500.0

# gains for orientational components: Kp_w, Kd_w
K1 = np.diag([20, 20, 20])
K2 = np.diag([10, 10, 10])

# gains for positional components: Kp_v, Kd_v
K3 = np.diag([20, 20, 20])
K4 = np.diag([25, 25, 25])

K_pos = np.block([K1, K1])
K_att = np.block([K2, K2])

control_params = {'K1': K1, 'K2': K2,
                  'K3': K3, 'K4': K4,
                  'K_pos': K_pos, 'K_att': K_att,
                  'u_min': u_min, 'u_max': u_max,
                  'Lambda': np.eye(6)*25
                  }


quat_0 = Rotation.from_euler('xyz', [0, 0, 90], degrees=True).as_quat()
quat_d = Rotation.from_euler('xyz', [0, 0, 0], degrees=True).as_quat()
print(quat_0, quat_d)

# initial position, velocity, quaternion, angular velocity
x0 = np.array([0., 1.0, 0.0,
      0, 0, 0,
      quat_0[3], quat_0[0], quat_0[1], quat_0[2], 
      0., 0., 0.])

# desired position, velocity, quaternion, angular velocity, linear acceleration, angular acceleration
trajectory_params = {'state_d': np.array([
                                0.7, 0.7, 0.,
                                0., 0., 0.,
                                1, 0, 0, 0, 
                                0., 0., 0.,
                                0., 0., 0.,
                                0., 0., 0.])}

states, angles = simulate_system(state_init=x0, frequency=freq, t_0=t0, t_final=tf, 
                sys_params=system_params, ctrl_params=control_params, traj_params=trajectory_params)
close("all") #this is the line to be added



# from matplotlib import pyplot as plt, animation
# import numpy as np
# from matplotlib.pyplot import *
# from scipy.integrate import odeint
# import numpy as np
# from scipy.optimize import linprog
# from qpsolvers import solve_qp
# import mpl_toolkits.mplot3d.axes3d as p3
# import matplotlib.animation as animation
# from matrices import *
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

# def get_position(states):
#     pos = np.vstack([states[:, 0], states[:, 1], states[:, 2]])
#     print(pos)
#     # wf - wireframe of a brick
#     pos_c = pos[:, 0] # x, y, z coord of CoM
#     # pos_1 =
#     # pos_2 =
#     # pos_3 =
#     # pos_4 =
    
#     return pos

# def get_current_orientation(i):
#     return angles[i, 0], angles[i, 1], angles[i, 2]

# def animate(num):
#     ax.clear()
#     xc, yc, zc = data[0][num], data[1][num], data[2][num]
#     roll, pitch, yaw = get_current_orientation(num)
#     Hc = Tx(xc) @ Ty(yc) @ Tz(zc)  @ Rx(roll) @ Ry(pitch) @ Rz(yaw)
#     H = Hc @ Tx(length/2) @ Ty(-width/2)
#     p1 = H[:3, 3]

#     H = Hc @ Tx(-length / 2) @ Ty(-width / 2)
#     p2 = H[:3, 3]

#     H = Hc @ Tx(-length / 2) @ Ty(width / 2)
#     p3 = H[:3, 3]

#     H = Hc @ Tx(length / 2) @ Ty(width / 2)
#     p4 = H[:3, 3]

#     points = np.vstack([p1, p2, p3, p4])
#     # points = np.array([[-p1[0], -p1[1], -p1[2]],
#     #                    [p1[0], -p1[1], -p1[2]],
#     #                    [p1[0], p1[1], -p1[2]],
#     #                    [-p1[0], p1[1], -p1[2]]
#     #                    ,[-1, -1, 1],
#     #                    [1, -1, 1],
#     #                    [1, 1, 1],
#     #                    [-1, 1, 1]
#     #                    ])
#     # points = np.array([[-1, -1, -1],
#     #                    [1, -1, -1],
#     #                    [1, 1, -1],
#     #                    [-1, 1, -1]
#     #                    ,[-1, -1, 1],
#     #                    [1, -1, 1],
#     #                    [1, 1, 1],
#     #                    [-1, 1, 1]
#     #                    ])
#     Z = points
#     verts = [[Z[0], Z[1], Z[2], Z[3]]
#              # ,[Z[4], Z[5], Z[6], Z[7]],
#              # [Z[0], Z[1], Z[5], Z[4]],
#              # [Z[2], Z[3], Z[7], Z[6]],
#              # [Z[1], Z[2], Z[6], Z[5]],
#              # [Z[4], Z[7], Z[3], Z[0]]
#              ]

#     # ax.scatter3D(data[0][:], data[1][:], data[2][:])
#     ax.scatter3D(data[0][0], data[1][0], data[2][0], color='b')
#     ax.scatter3D(data[0][-1], data[1][-1], data[2][-1], color='r')

#     ax.scatter3D(xc, yc, zc, color='b', alpha=.50)
#     ax.scatter3D(Z[:, 0], Z[:, 1], Z[:, 2], color='b', alpha=.50)
#     collection = Poly3DCollection(verts, facecolors='b', linewidths=1, edgecolors='r', alpha=.10)
#     ax.add_collection3d(collection)
#     ax.set_xlim3d([0.0, 2.])
#     ax.set_ylim3d([0.0, 3.0])
#     ax.set_zlim3d([-1.0, 1.0])

#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')

#     # return trj  # collection

# # Attach 3D axis to the figure
# fig = figure()
# ax = p3.Axes3D(fig)

# data = get_position(states)

# ani = animation.FuncAnimation(fig, animate, frames=np.shape(data)[1],
#                                   interval=dT * 1e4) #repeat=False,



# show()