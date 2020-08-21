import os, pickle
import numpy as np
import matplotlib.pyplot as plt

def load_reward_clock_funcs(path):
    with open(path, "rb") as f:
        clock_funcs = pickle.load(f)
    return clock_funcs

dirname = os.path.dirname(__file__)
reward_clock_func = load_reward_clock_funcs(os.path.join(dirname, "no_incentive_aslip_clock_strict0.3.pkl"))
left_clock = reward_clock_func["left"][-1]
right_clock = reward_clock_func["right"][-1]

l_force_clock = left_clock[0]
l_vel_clock = left_clock[1]
r_force_clock = right_clock[0]
r_vel_clock = right_clock[1]

phaselen = 90
phases = np.arange(-30, phaselen)

fig, axs = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(10,5), constrained_layout=True)
axs[0].plot(phases, l_force_clock(phases))
axs[1].plot(phases, l_vel_clock(phases))
axs[2].plot(phases, r_force_clock(phases))
axs[3].plot(phases, r_vel_clock(phases))

plt.show()

