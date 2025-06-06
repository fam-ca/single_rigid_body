from matplotlib import pyplot as plt, animation
import numpy as np



def visualize_rotation(rotation_matrices,
                        output_file=None,
                       stats=None,
                       verbose=True,
                       show=True):

    axes_t = np.array([rotation_matrices[:, :, 0],
                       rotation_matrices[:, :, 1],
                       rotation_matrices[:, :, 2]])

    N = len(axes_t[0])
    frames = N
    interval = 1/N

    # ////////////////////
    # Animate the solution
    # ////////////////////

    # Set up figure & 3D axis for animation
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d') # for older version write fig.add_axes([0, 0, 1, 1], projection='3d')
    # ax.axis('off')

    # choose a different color for each trajectory
    colors = ['r', 'g', 'b']

    # set up lines and points
    lines = sum([ax.plot([], [], [], '--', c=c, alpha=0.3)
                for c in colors], [])
    axes = sum([ax.plot([], [], [], '-', c=c, lw=3)
                for c in colors], [])
    pts = sum([ax.plot([], [], [], 'o', c=c, lw=5)
               for c in colors], [])

    corner_text = ax.text(0.02, 0.90, 1.1, '')

    ax.set_xlim((-1.1, 1.1))
    ax.set_ylim((-1.1, 1.1))
    ax.set_zlim((-1.1, 1.1))

    ax.view_init(14, 0)

    def init():
        for line, pt, axs, xi in zip(lines, pts, axes, axes_t):
            x, y, z = xi[:0].T
            line.set_data(x[-lag:], y[-lag:])
            line.set_3d_properties(z[-lag:])
            axs.set_data(np.hstack((0, x[-1:])), np.hstack((0, y[-1:])))
            axs.set_3d_properties(np.hstack((0, z[-1:])))


            pt.set_data(x[-1:], y[-1:])
            pt.set_3d_properties(z[-1:])
        return lines + pts + axes

    lag = 35
    
    

    
    def animate(i):
        rate = 1
        j = (rate * i) % axes_t.shape[1]

        stat_text = ''
        
        for line, pt, axs, xi in zip(lines, pts, axes, axes_t):
            x, y, z = xi[:j].T
            line.set_data(x[-lag:], y[-lag:])
            line.set_3d_properties(z[-lag:])
            axs.set_data(np.hstack((0, x[-1:])), np.hstack((0, y[-1:])))
            axs.set_3d_properties(np.hstack((0, z[-1:])))


            pt.set_data(x[-1:], y[-1:])
            pt.set_3d_properties(z[-1:])
            
        if stats is not None:
            for stat in stats:
                stat_text += rf'{stat[0]}:  {round(stat[1][j],3)}'
                stat_text +='\n'
            corner_text.set_text(stat_text)

        # ax.view_init(30, 0.3)
        fig.canvas.draw()
        return lines + pts + axes
    
    if verbose:
        print('Animation begin...')
        print('Hit CTRL+W to exit')

    # instantiate the animator.
    anim = animation.FuncAnimation(fig, animate,
                                   init_func=init,
                                   frames=frames,
                                   interval=interval,
                                   blit=True)
    if show:
        plt.show()
    else:
        anim.save('animation.gif', writer='pillow', fps=60)
    return anim