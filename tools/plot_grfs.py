import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
from torch.autograd import Variable
import time, os, sys
import math

from cassie import CassieEnv, CassiePlayground, CassieStandingEnv, CassieEnv_noaccel_footdist_omniscient, CassieEnv_footdist, CassieEnv_noaccel_footdist
from cassie.cassiemujoco import CassieSim

import argparse
import pickle

def collect_data(policy, args, run_args):
    wait_steps = 100
    num_cycles = 10
    speed = 1.0

    env = CassieEnv(traj=run_args.traj, state_est=run_args.state_est, no_delta=run_args.no_delta, dynamics_randomization=run_args.dyn_random, phase_based=run_args.phase_based, clock_based=run_args.clock_based, reward=args.reward, history=run_args.history)

    state = torch.Tensor(env.reset_for_test())
    env.update_speed(speed)
    env.render()

    data = []

    # print("iros: ", iros_env.simrate, iros_env.phaselen)
    # print("aslip: ", aslip_env.simrate, aslip_env.phaselen)

    with torch.no_grad():

        # Run few cycles to stabilize (do separate incase two envs have diff phaselens)
        for i in range(wait_steps):
            action = policy.forward(torch.Tensor(state), deterministic=True).detach().numpy()
            state, reward, done, _ = env.step(action)
            env.render()
            print(f"act spd: {env.sim.qvel()[0]:.2f}\t cmd speed: {env.speed:.2f}")
            # curr_qpos = aslip_env.sim.qpos()
            # print("curr height: ", curr_qpos[2])

        # Collect actual data
        print("Start actual data")
        for i in range(num_cycles):
            for j in range(env.phaselen):
                action = policy.forward(torch.Tensor(state), deterministic=True).detach().numpy()
                for k in range(env.simrate):
                    env.step_simulation(action)
                    data.append(env.sim.get_foot_forces())
                
                env.time += 1
                env.phase += env.phase_add
                if env.phase > env.phaselen:
                    env.phase = 0
                    env.counter += 1
                state = env.get_full_state()
                env.render()
        
        print(len(data))

    np.save("grf_run_data.npy".format(speed), data)
        