import sys
from turtle import pos
import numpy as np
from arm_dynamics_teacher import ArmDynamicsTeacher
from arm_dynamics_student import ArmDynamicsStudent
from robot import Robot
from arm_gui import ArmGUI, Renderer
import argparse
import time
import math
import torch
import random
import copy


np.set_printoptions(suppress=True)

class MPC:

    def __init__(self,):
        self.control_horizon = 20 # H
        self.planning_horizon = 85 # N
        self.learning_rate = 0.4
        # Define other parameters here

    def cost_function(self, dynamics, traj, goal, action):
        cost =[]

        for i in range(len(traj)):
            pos_ee = dynamics.compute_fk(traj[i])
            loss = np.linalg.norm(goal - pos_ee)
            # print(f"loss, {loss}")
            cost.append(loss)
        # print(cost) 
        # return np.linalg.norm(goal - dynamics.compute_fk(traj[0]))
        return sum(cost) #+ 0.5 * np.sum(action)
    
    def roll_out_seq(self, dynamics, init_state, action):
        X_k = [init_state]
        N = self.planning_horizon
        for i in range(N-1):
            next_state = dynamics.advance(X_k[i], action)
            X_k.append(next_state)
        return X_k


    def compute_action(self, dynamics, state, goal, action):
        # Put your code here. You must return an array of shape (num_links, 1)
        # Initialization
        du = self.learning_rate
        dU = [du, -du]
              

        N = self.planning_horizon
        H = self.control_horizon
        num_links = dynamics.get_num_links()
        u_k = action.copy()
        x_k = state.copy()
        u_star = action.copy()
        
        # Roll out the trajectory 
        X_k = self.roll_out_seq(dynamics, x_k, u_k)

        for n in range(num_links):
            cost_record = np.zeros(2)
            init_cost = self.cost_function(dynamics, X_k, goal, u_k)
            while 1:
                for i in range(len(dU)):
                    u_k[n] = u_star[n] + dU[i]
                    X_k = self.roll_out_seq(dynamics, x_k, u_k)
                    new_cost = self.cost_function(dynamics, X_k, goal, u_k)
                    cost_record[i] = new_cost

                best_cost = np.min(cost_record)
                best_idx = np.argmin(cost_record)
                
                if best_cost < init_cost:
                    init_cost = best_cost
                    u_star[n] = u_star[n] + dU[best_idx]
                else:
                    # print(2)
                    break

        return u_star


def main(args):

    # Arm
    arm = Robot(
        ArmDynamicsTeacher(
            num_links=args.num_links,
            link_mass=args.link_mass,
            link_length=args.link_length,
            joint_viscous_friction=args.friction,
            dt=args.time_step,
        )
    )

    # Dynamics model used for control
    if args.model_path is not None:
        dynamics = ArmDynamicsStudent(
            num_links=args.num_links,
            link_mass=args.link_mass,
            link_length=args.link_length,
            joint_viscous_friction=args.friction,
            dt=args.time_step,
        )
        dynamics.init_model(args.model_path, args.num_links, device=torch.device("cpu"))
    else:
        # Perfectly accurate model dynamics
        dynamics = ArmDynamicsTeacher(
            num_links=args.num_links,
            link_mass=args.link_mass,
            link_length=args.link_length,
            joint_viscous_friction=args.friction,
            dt=args.time_step,
        )


    # Controller
    controller = MPC()

    # Control loop
    arm.reset()
    action = np.zeros((arm.dynamics.get_action_dim(), 1))
    goal = np.zeros((2, 1))
    goal[0, 0] = args.xgoal
    goal[1, 0] = args.ygoal
    arm.goal = goal

    if args.gui:
        renderer = Renderer()
        time.sleep(0.25)

    dt = args.time_step
    k = 0
    while True:
        t = time.time()
        arm.advance()
        if args.gui:
            renderer.plot([(arm, "tab:blue")])
        k += 1
        time.sleep(max(0, dt - (time.time() - t)))
        if k == controller.control_horizon:
            state = arm.get_state()
            action = controller.compute_action(dynamics, state, goal, action)
            arm.set_action(action)
            k = 0


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_links", type=int, default=3)
    parser.add_argument("--link_mass", type=float, default=0.1)
    parser.add_argument("--link_length", type=float, default=1)
    parser.add_argument("--friction", type=float, default=0.1)
    parser.add_argument("--time_step", type=float, default=0.01)
    parser.add_argument("--time_limit", type=float, default=5)
    parser.add_argument("--gui", action="store_const", const=True, default=False)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--xgoal", type=float, default=-1.4)
    parser.add_argument("--ygoal", type=float, default=-1.4)
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())

