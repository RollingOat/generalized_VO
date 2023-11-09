import numpy as np
from scipy.optimize import minimize
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
import time

class generalVO:
    def __init__(self, v_max, w_max, v_step, w_step, t_max, t_step, robot_radius, obstacle_radius):
        self.v_max = v_max
        self.w_max = w_max
        self.v_step = v_step
        self.w_step = w_step
        self.t_max = t_max
        self.t_step = t_step
        self.robot_radius = robot_radius
        self.obstacle_radius = obstacle_radius

    def compute_optimal_u(self, u_desired, u_feasible):
        diff_norm = np.linalg.norm(u_desired - u_feasible, axis=1)
        u_optimal = u_feasible[np.argmin(diff_norm)]
        return u_optimal, np.argmin(diff_norm).item()

    def Dt(self, dt, u, obstacle_state, robot_state):
        # u = [v, w]
        # obstacle_state = [x, y, vx, vy]
        # robot_state = [x, y, theta]
        robot_x, robot_y, _ = self.predict_robot_state(robot_state, dt, u)
        obs_x, obs_y = self.predict_obstacle_pos(obstacle_state, dt)
        d = np.sqrt((robot_x - obs_x)**2 + (robot_y - obs_y)**2)
        return d

    def find_t_min(self, init_guess, u, obstacle_state, robot_state):
        res = minimize(self.Dt, init_guess, args=(u, obstacle_state, robot_state),
                       method='nelder-mead', options={'xtol': 1e-8, 'disp': False}, bounds=[(0, self.t_max)])
        t_min = res.x
        # print("t_min: ", t_min)
        d_min = res.fun.item()
        # d_min = self.distance(u, t_min, obstacle_state, robot_state)
        # print("t_min: ", t_min)
        # print("d_min: ", d_min)
        return t_min, d_min

    def find_prefered_control(self, robot_state, goal):
        """
        Find the prefered control to the goal
        :param robot_state: current robot state, [x, y, theta]
        :param goal: goal position, [x, y]
        :return: prefered control
        """
        x = robot_state[0]
        y = robot_state[1]
        theta = robot_state[2]
        x_g = goal[0]
        y_g = goal[1]
        v = self.v_max
        goal_angle = np.arctan2(y_g - y, x_g - x)
        w = (goal_angle - theta) / self.t_step
        u = np.array([v, w])
        if(((x_g - x)**2+(y_g - y)**2)**(1/2) < 0.5):
            u = np.array([0, 0])
        return u

    def find_best_feasible_control(self, robot_state, obstacle_state, goal):
        v = np.arange(-self.v_max-self.v_step, self.v_max+self.v_step, self.v_step)
        w = np.arange(-self.w_max-self.w_step, self.w_max+self.w_step, self.w_step)
        feasible_u = np.array([0, 0])
        feasible_t = np.array([0])
        feasible_d = np.array([0])
        for i in range(len(v)):
            for j in range(len(w)):
                d_mins = []
                t_mins = []
                u = np.array([v[i], w[j]])
                for obs in obstacle_state:
                    start = time.time()
                    t_min, d_min = self.find_t_min(0,
                        u, obs, robot_state)
                    end = time.time()
                    # print("runtime to find minium time to collision: ", end - start)
                    d_mins.append(d_min)
                    t_mins.append(t_min)
                    # print("t_min: ", t_min)
                    # print("d_min: ", d_min)
                min_in_d_mins = np.min(d_mins)
                argmin_in_d_mins = np.argmin(d_mins).item()
                t_min = t_mins[argmin_in_d_mins]
                if min_in_d_mins > (self.robot_radius + self.obstacle_radius) and (t_min >= self.t_step or t_min == 0):
                    feasible_u = np.vstack((feasible_u, u))
                    feasible_t = np.vstack((feasible_t, t_min))
                    feasible_d = np.vstack((feasible_d, d_min))
        if feasible_u.shape[0] == 1:
            print("no feasible control")
            return np.array([0, 0])
        prefered_u = self.find_prefered_control(robot_state, goal)
        optimal_u, optimal_idx = self.compute_optimal_u(prefered_u, feasible_u)
        print("optimal_u: ", optimal_u)
        print("min distance to obstacle: ", feasible_d[optimal_idx])
        print("t_min: ", feasible_t[optimal_idx])
        print("prefered_u: ", prefered_u)
        print("\n")
        return optimal_u

    def predict_obstacle_pos(self, cur_obs_state, t):
        pos = cur_obs_state[0:2]
        vel = cur_obs_state[2:4]
        new_pos = pos + vel * t
        return new_pos[0], new_pos[1]

    def predict_robot_state(self, cur_state, dt, u):
        # robot state = [x, y, theta]
        # u = [v, w]
        x = cur_state[0]
        y = cur_state[1]
        v = u[0]
        w = u[1]
        theta = cur_state[2]
        # dx = v*(dt - w**2)*dt**3/6
        # dy = v*(w*dt**2/2-w**3*dt**4 / 24)
        # dtheta = w * dt
        # new_x = x + dx*np.cos(theta) - dy*np.sin(theta)
        # new_y = y + dx*np.sin(theta) + dy*np.cos(theta)
        # new_theta = theta + dtheta
        new_x = x + v*np.cos(theta)*dt
        new_y = y + v*np.sin(theta)*dt
        new_theta = theta + w*dt
        return new_x, new_y, new_theta

    def distance(self, u, dt, obs_state, robot_state):
        # u = [v, w]
        # obs_state = [x, y, vx, vy]
        # robot_state = [x, y, theta]
        robot_x, robot_y, _ = self.predict_robot_state(robot_state, dt, u)
        obs_x, obs_y = self.predict_obstacle_pos(obs_state, dt)
        d = np.sqrt((robot_x - obs_x)**2 + (robot_y - obs_y)**2)
        return d


def simulate_obs_state(init_pos, vel, t):
    cur_pos = init_pos + vel*t
    cur_state = np.hstack((cur_pos, vel))
    return cur_state


def simulate_robot_state(last_pos, cur_u, dt):
    last_x = last_pos[0]
    last_y = last_pos[1]
    last_theta = last_pos[2]
    v = cur_u[0]
    w = cur_u[1]
    # dx = v*(dt - w**2)*dt**3/6
    # dy = v*(w*dt**2/2-w**3*dt**4 / 24)
    # dtheta = w * dt
    # new_x = last_x + dx*np.cos(last_theta) - dy*np.sin(last_theta)
    # new_y = last_y + dx*np.sin(last_theta) + dy*np.cos(last_theta)
    # new_theta = last_theta + dtheta
    new_x = last_x + v*np.cos(last_theta)*dt
    new_y = last_y + v*np.sin(last_theta)*dt
    new_theta = last_theta + w*dt
    new_state = np.array([new_x, new_y, new_theta])
    return new_state

def update(frame, robot_hist_traj, obs_hist_traj, robot_hist_control):
    plt.cla()
    cur_robot_pos = robot_hist_traj[:, frame]
    cur_obs_poss = obs_hist_traj[:, :, frame]
    cur_v = robot_hist_control[0, frame]
    cur_theta = robot_hist_traj[2, frame]
    dx = np.cos(cur_theta)
    dy = np.sin(cur_theta)

    # plot the robot position
    robot_circle = Circle((cur_robot_pos[0], cur_robot_pos[1]), 0.2, color='b')
    arrow = plt.arrow(cur_robot_pos[0], cur_robot_pos[1], dx,dy, width=0.05, color='b')
    ax.add_patch(robot_circle)
    for i in range(len(cur_obs_poss)):
        # plot the obstacle position
        obs_circle = Circle((cur_obs_poss[i][0], cur_obs_poss[i][1]), 0.2, color='r')
        ax.add_patch(obs_circle)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
   


if __name__ == "__main__":

    obs_speed = 1
    obs_vels = np.array([[-obs_speed, -obs_speed], [-obs_speed, obs_speed], [obs_speed, -obs_speed], [-obs_speed,0], [obs_speed, 0]])
    
    obs_init_positions = np.array([[5, 5], [5, 1], [1, 5], [6, 3], [1, 4]])
    robot_init_state = np.array([0, 0, 0])
    robot_radius = 0.2
    obstacle_radius = 0.2
    v_max = 1
    w_max = 2
    v_step = 0.1
    w_step = 0.1
    t_max = 2
    t_step = 0.1
    t = np.arange(0, 10, t_step)
    goal = np.array([3, 6])
    vo = generalVO(v_max, w_max, v_step, w_step, t_max,
                   t_step, robot_radius, obstacle_radius)
    last_robot_state = robot_init_state

    obs_hist_traj = np.zeros((len(obs_vels), 4, len(t)))
    robot_hist_traj = np.zeros((3, len(t)))
    robot_hist_control = np.zeros((2, len(t)))

    reached_goal = False
    for i in range(len(t)):
        if reached_goal:
            robot_hist_control[:, i] = np.array([0, 0])
            robot_hist_traj[:, i] = last_robot_state
            continue
        cur_t = t[i]
        obs_states = simulate_obs_state(obs_init_positions, obs_vels, cur_t)
        obs_hist_traj[:, :, i] = obs_states
        # measure the runtime of the find_best_feasible_control function
        start = time.time()
        u_best = vo.find_best_feasible_control(
            last_robot_state, obs_states, goal)
        end = time.time()
        print("runtime: ", end - start)
        robot_hist_control[:, i] = u_best
        robot_state = simulate_robot_state(last_robot_state, u_best, t_step)
        print("robot_state: ", robot_state)
        robot_hist_traj[:, i] = last_robot_state
        last_robot_state = robot_state
        if np.linalg.norm(robot_state[0:2] - goal) < 0.5:
            print("reach the goal")
            reached_goal = True
            
    
    fig, ax = plt.subplots()
    
    num_frames = len(t)
    interval = t_step * 1000
    ani = FuncAnimation(fig, update, fargs=(robot_hist_traj, obs_hist_traj, robot_hist_control),frames=num_frames,
                        interval=interval, repeat=False)
    ani.save('vo.gif', writer='imagemagick', fps=10)
    plt.show()
