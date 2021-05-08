# The source for most of this code is here:
# https://deeplizard.com/learn/video/PyQNfsGUnQA
# Their code is itself a modification of pytorch's tutorial code for deep Q networks.

import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython: from IPython import display

import threading
from subprocess import PIPE, Popen, run
import time
import subprocess

class DQN(nn.Module):
    def __init__(self, img_height, img_width):
        super().__init__()

        self.fc1 = nn.Linear(in_features=img_height*img_width, out_features=24)   
        self.fc2 = nn.Linear(in_features=24, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=8)

def forward(self, t):
    t = t.flatten(start_dim=1)
    t = F.relu(self.fc1(t))
    t = F.relu(self.fc2(t))
    t = self.out(t)
    return t

Experience = namedtuple(
    'Experience',
    ('state', 'action', 'next_state', 'reward')
)

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

        def push(self, experience):
            if len(self.memory) < self.capacity:
                self.memory.append(experience)
            else:
                self.memory[self.push_count % self.capacity] = experience
                self.push_count += 1

        def sample(self, batch_size):
            return random.sample(self.memory, batch_size)

        def can_provide_sample(self, batch_size):
            return len(self.memory) >= batch_size

class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

        def get_exploration_rate(self, current_step):
            return self.end + (self.start - self.end) * \
                math.exp(-1. * current_step * self.decay)

class Agent():
    def __init__(self, strategy, device):
        self.current_step = 0
        self.strategy = strategy
        self.device = device

    def select_actions(self, state, policy_net):
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        if rate > random.random():
            randActions = []
            for i in range(8):
                randActions.append(random.random() > 0.5)
            return randActions # explore
        else:
            with torch.no_grad():
                rawOuts = policy_net(state).numpy()
                outs = []
                for elem in rawOuts:
                    outs.append(elem > 0.5)
                return outs # exploit

# This section is naturally almost entirely written by me. 
class DustforceEnv():
    def __init__(self):
        cmd = "/home/caleb/DF/dustmod.linux.steam.nofocus.bin.x86_64"
        proc = Popen(["unbuffer", cmd], stdout=PIPE, stderr=PIPE, universal_newlines=True)
        threading.Thread(target=lambda:self.process_input(proc)).start()
        out = subprocess.run(["unbuffer", "xdotool", "search", "dustforce"], capture_output=True)
        print("Window ID is " + out.stdout.decode())
        self.winID = out.stdout.decode()
        self.lastActions = []
        for i in range(8):
            self.lastActions.append(False)
        self.inputList = ["w","a","s","d","u","i","o","p"]

    def process_input(self, proc):
        while (line := proc.stdout.readline()) != b"":
            print(line, end="")
        
    def sendInput(self, cmd, key, delay=17):
        print("sendInput " + cmd + " " + key)
        subprocess.run(["xdotool",
                        cmd,
                        "--window", self.winID,
                        "--delay", str(delay),
                        key])

    def sendKeypress(self, key, delay=17):
        print("sendKeypress " + key)
        subprocess.run(["xdotool",
                        "keydown",
                        "--window", self.winID,
                        "--delay", str(delay),
                        key,
                        "keyup",
                        "--window", self.winID,
                        "--delay", str(delay),
                        "key"])

    def reset(self):
        for i in range(8):
            self.sendKeypress("r")
        # sent so many times because this is the one place dropping the input is intolerable
        # and also it doesn't really matter if it's sent too many times; the game interprets it fine
        
    def close(self):
        self.sendInput("Escape")
        self.sendInput("w")
        self.sendInput("Return")
        self.sendInput("a")
        self.sendInput("u")

    def step(self, actions):
        for (action, lastAction, key) in zip(actions, lastActions, inputList):
            if action and not lastAction:
                sendInput("keydown", key)
            if lastAction and not action:
                sendInput("keyup", key)
        # reward, etc.
        return (0, 0, False, 0)
    

class EnvManager():
    def __init__(self, device):
        self.device = device
        self.env = DustforceEnv()
        self.env.reset()
        self.current_screen = None
        self.done = False

    def reset(self):
        self.env.reset()
        self.current_screen = None

    def close(self):
        self.env.close()

    def take_actions(self, actions):
        _, reward, self.done, _ = self.env.step(actions)
        return torch.tensor([reward], device=self.device)

    def just_starting(self):
        return self.current_screen is None

    def get_state(self):
        if self.just_starting() or self.done:
            black_screen = torch.zeros_like(self.current_screen)
            return black_screen
        else:
            s1 = self.current_screen
            s2 = refresh_screen()
            self.current_screen = s2
            return s2 - s1

    def get_screen_height(self):
        return 10
    def get_screen_width(self):
        return 10

def plot(values, moving_avg_period):
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(values)
    plt.plot(get_moving_average(moving_avg_period, values))
    plt.pause(0.001)
    if is_ipython: display.clear_output(wait=True)

def get_moving_average(period, values):
    values = torch.tensor(values, dtype=torch.float)
    if len(values) >= period:
        moving_avg = values.unfold(dimension=0, size=period, step=1) \
                           .mean(dim=1).flatten(start_dim=0)
        moving_avg = torch.cat((torch.zeros(period-1), moving_avg))
        return moving_avg.numpy()
    else:
        moving_avg = torch.zeros(len(values))
        return moving_avg.numpy()

def extract_tensors(experiences):
    # Convert batch of Experiences to Experience of batches
    batch = Experience(*zip(*experiences))

    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)

    return (t1,t2,t3,t4)

batch_size = 256
gamma = 0.999
eps_start = 1
eps_end = 0.01
eps_decay = 0.001
target_update = 10
memory_size = 100000
lr = 0.001
num_episodes = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
em = EnvManager(device)
strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
agent = Agent(strategy, device)
memory = ReplayMemory(memory_size)
policy_net = DQN(em.get_screen_height(), em.get_screen_width()).to(device)
target_net = DQN(em.get_screen_height(), em.get_screen_width()).to(device)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)

episode_scores = []

for episode in range(num_episodes):
    em.reset()
    state = em.get_state()
    for timestep in count():
        action = agent.select_actions(state, policy_net)
        reward = em.take_action(action)
        next_state = em.get_state()
        memory.push(Experience(state, action, next_state, reward))
        state = next_state
        if memory.can_provide_sample(batch_size):
            experiences = memory.sample(batch_size)
            states, actions, rewards, next_states = extract_tensors(experiences)

            current_q_values = QValues.get_current(policy_net, states, actions)
            next_q_values = QValues.get_next(target_net, next_states)
            target_q_values = (next_q_values * gamma) + rewards

            loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if em.done:
            if episode % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())
                episode_durations.append(timestep)
                plot(episode_durations, 100)
            break

em.close()
