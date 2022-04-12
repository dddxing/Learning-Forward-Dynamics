import numpy as np
from arm_dynamics_teacher import ArmDynamicsTeacher
from robot import Robot
import argparse
import os
from mpc import MPC
import time
import tqdm
import ray

np.set_printoptions(suppress=True)


def simulate():
    # Controller
    X = []
    Y = []
    controller = MPC()
    arm.reset()
    action = np.zeros((arm.dynamics.get_action_dim(), 1))
    goal = np.zeros((2, 1))

    xgoal = np.random.uniform(-0.7*dynamics.num_links*dynamics.link_length, 0.7*dynamics.num_links*dynamics.link_length)
    ygoal = np.random.uniform(-dynamics.link_length, -0.7*dynamics.num_links*dynamics.link_length)
    goal[0, 0] = xgoal
    goal[1, 0] = ygoal
    arm.goal = goal
    print(goal)

    dt = args.time_step
    timer = 3

    while arm.get_t() < timer:

        t = time.time()
        
        time.sleep(max(0, dt - (time.time() - t)))
        state = arm.get_state()
        # print(state)
        action = controller.compute_action(dynamics, state, goal, action)
        x = np.concatenate((state, action), axis=None)
        X.append(x)
        # print(action)
        # arm.advance()
        arm.set_action(action)
        arm.advance()
        k = 0
        next_state = arm.get_state()
        Y.append(next_state)

    return X, Y

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_links', type=int, default=3)
    parser.add_argument('--link_mass', type=float, default=0.1)
    parser.add_argument('--link_length', type=float, default=1)
    parser.add_argument('--friction', type=float, default=0.1)
    parser.add_argument('--time_step', type=float, default=0.01)
    parser.add_argument('--time_limit', type=float, default=5)
    parser.add_argument('--save_dir', type=str, default='dataset')
    return parser.parse_args()


if __name__ == '__main__':
    np.random.seed(10)
    start_time = time.time()
    args = get_args()
    ray.init()
    # Teacher arm
    dynamics_teacher = ArmDynamicsTeacher(
        num_links=args.num_links,
        link_mass=args.link_mass,
        link_length=args.link_length,
        joint_viscous_friction=args.friction,
        dt=args.time_step
    )
    arm_teacher = Robot(dynamics_teacher)
    arm = arm_teacher

    dynamics = dynamics_teacher
    # ---
    # You code goes here. Replace the X, and Y by your collected data
    # Control the arm to collect a dataset for training the forward dynamics.
    X = np.zeros((arm_teacher.dynamics.get_state_dim() + arm_teacher.dynamics.get_action_dim(), 0))
    Y = np.zeros((arm_teacher.dynamics.get_state_dim(), 0))

    # ---
    
    simulate_remote = ray.remote(simulate)
    # futures = [simulate_remote.remote() for i in range(100)]
    futures = []
    for i in tqdm.tqdm(range(100)):
        futures.append(simulate_remote.remote())
    res = ray.get(futures)

    X = []
    Y = []

    for item in res:
        X.extend(item[0])
        Y.extend(item[1])

    X = np.array(X).T
    Y = np.array(Y) 

    print('X shape:',X.shape,'Y shape:',Y.shape) 
    # X = np.hstack(X)
    Y = np.hstack(Y) 

    print('X shape:',X.shape,'Y shape:',Y.shape)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    np.save(os.path.join(args.save_dir, 'X.npy'), X)
    np.save(os.path.join(args.save_dir, 'Y.npy'), Y)
    elapsed = time.time() - start_time
    elapsed = time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed))
    print("it took ", (elapsed))
    