# The source for much of this code is from this series of lessons:
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

        self.conv1 = nn.Conv2d(in_channels=4, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=18, kernel_size=5)

        self.fc1 = nn.Linear(in_features=18*5*3, out_features=200)
        self.fc2 = nn.Linear(in_features=200, out_features=100)
        self.fc3 = nn.Linear(in_features=100, out_features=40)        
        self.out = nn.Linear(in_features=40, out_features=8)

    def forward(self, t):
        weight = torch.tensor([1], dtype=torch.float)
        
        t = self.conv1(t)
        t = F.prelu(t, weight)
        t = F.max_pool2d(t, kernel_size=2, stride=2)        
        
        t = self.conv2(t)
        t = F.prelu(t, weight)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        
        t = t.reshape(-1, 18*5*3)
        t = self.fc1(t)
        t = F.prelu(t, weight)

        t = self.fc2(t)
        t = F.prelu(t, weight)

        t = self.fc3(t)
        t = F.prelu(t, weight)        

        t = self.out(t)
        
        return t

class DumbDQN(nn.Module):
    def __init__(self, img_height, img_width):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=6, kernel_size=5)
        self.fc1 = nn.Linear(in_features = 6 * img_height * img_width, out_features = 8)

    def forward(self, t):
        weight = torch.tensor([1], dtype=torch.float)
        
        t = self.conv1(t)
        t = F.prelu(t, weight)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = t.reshape(-1, 6*10*14)
        t = self.fc1(t)
        
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

    def select_action(self, state, policy_net):
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1
        if rate > random.random():
            return torch.tensor([random.randrange(8)]).to(self.device) # explore
        else:
            with torch.no_grad():
                outs = policy_net(state)
                return outs.argmax(dim=1).to(self.device) # exploit

# This section is naturally almost entirely written by me.
# Note from later in the project: at this point, *most* sections are heavily modified.
class DustforceEnv():
    def __init__(self):
        cmd = "/home/caleb/DF/dustmod.linux.steam.nofocus.bin.x86_64"
        proc = Popen(["unbuffer", cmd], stdout=PIPE, stderr=PIPE, universal_newlines=True)
        threading.Thread(target=lambda:self.process_input(proc)).start()
        time.sleep(2)
        out = subprocess.run(["unbuffer", "xdotool",
                              "search", "--name", "dustforce"],
                             capture_output=True)
        print("Window ID is " + out.stdout.decode())
        self.winID = out.stdout.decode()
        self.gameFrame = 0
        self.currentReward = 0
        self.inputList = ["a","d","w","s","u","i","o","p"]
        self.frame = np.zeros([4, self.get_height(), self.get_width()])
        self.framePos = 0
        self.navigate()
        self.done = False

    def navigate(self):
        time.sleep(0.3)
        self.sendKeypress("Return")
        # time.sleep(0.3)
        # self.sendKeypress("w")
        # time.sleep(0.3)
        # self.sendKeypress("Return")
        # time.sleep(1)
        # self.sendKeypress("Tab")
        
    def process_input(self, proc):
        while (line := proc.stdout.readline().strip()) != b"":
            if len(line) != 0:
                signalChar = line[0]
                line = line[1:None]
                if signalChar == "}":
                    self.framePos += 1
                    if self.framePos < self.get_height():
                        for i in range(self.get_width()):
                            tile_type = int(line[i])
                            # gotta basically do manual one-hot encoding
                            # thanks a ton to Alexspeedy for, well, telling me about one-hot encoding
                            # and also for this code since they screwed with this once I showed them
                            if tile_type == 0:
                                self.frame[0][self.framePos][i] = 0
                                self.frame[1][self.framePos][i] = 0
                                self.frame[2][self.framePos][i] = 0
                                self.frame[3][self.framePos][i] = 0
                            elif tile_type == 1:
                                self.frame[0][self.framePos][i] = 1
                                self.frame[1][self.framePos][i] = 0
                                self.frame[2][self.framePos][i] = 0
                                self.frame[3][self.framePos][i] = 0
                            elif tile_type == 2:
                                self.frame[0][self.framePos][i] = 0
                                self.frame[1][self.framePos][i] = 1
                                self.frame[2][self.framePos][i] = 0
                                self.frame[3][self.framePos][i] = 0
                            elif tile_type == 3:
                                self.frame[0][self.framePos][i] = 0
                                self.frame[1][self.framePos][i] = 0
                                self.frame[2][self.framePos][i] = 1
                                self.frame[3][self.framePos][i] = 0
                            elif tile_type == 4:
                                self.frame[0][self.framePos][i] = 0
                                self.frame[1][self.framePos][i] = 0
                                self.frame[2][self.framePos][i] = 0
                                self.frame[3][self.framePos][i] = 1
                elif signalChar == ">":
                    if line[0:2] == "~>":
                        self.framePos = 0
                elif signalChar == ")":
                    self.currentReward = int(line)
                elif signalChar == "]":
                    self.gameFrame = int(line)
                elif signalChar == "`":
                    self.done = True
                else:
                    print(signalChar + line)

    def sendInput(self, cmd, key, delay=17):
        subprocess.run(["xdotool",
                        cmd,
                        "--window", self.winID,
                        "--delay", str(delay),
                        key])

    def sendKeypress(self, key, delay=17):
        subprocess.run(["xdotool",
                        "keydown",
                        "--window", self.winID,
                        "--delay", str(delay),
                        key,
                        "keyup",
                        "--window", self.winID,
                        "--delay", str(delay),
                        key])

    def reset(self):
        for i in range(8):
            self.sendKeypress("r")
        # sent so many times because this is the one place dropping the input is intolerable
        # and also it doesn't really matter if it's sent too many times; the game interprets it fine
        self.done = False

    def close(self):
        self.sendKeypress("Escape")
        self.sendKeypress("w")
        self.sendKeypress("Return")
        self.sendKeypress("a")
        self.sendKeypress("u")

    def step(self, action):
        self.sendInput("keydown", self.inputList[action])
        time.sleep(wait_time)
        self.sendInput("keyup", self.inputList[action])        
        # Don't worry - the reward in an interval of time gets figured out later.
        return (0, self.currentReward, self.done, 0)

    def get_height(self):
        return 24

    def get_width(self):
        return 32

class EnvManager():
    def __init__(self, device):
        self.device = device
        self.env = DustforceEnv()
        # not currently necessary
        # self.env.reset()
        self.current_screen = None
        self.done = False

    def reset(self):
        self.env.reset()
        self.current_screen = None

    def close(self):
        self.env.close()

    def take_action(self, action):
        _, reward, self.done, _ = self.env.step(action)
        return torch.tensor([reward], device=self.device)

    def just_starting(self):
        return self.current_screen is None

    def refresh_screen(self):
        screen = self.env.frame
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 4
        screen = torch.from_numpy(screen)
        return screen.unsqueeze(0).to(self.device)

    def get_state(self):
        if self.just_starting() or self.done:
            self.current_screen = self.refresh_screen()
            black_screen = torch.zeros_like(self.current_screen)
            return black_screen
        else:
            self.current_screen = self.refresh_screen()
            return self.current_screen

    def get_screen_height(self):
        return self.env.get_height()
    def get_screen_width(self):
        return self.env.get_width()

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

def tensorize(actions):
    tensorActs = []
    for action in actions:
        if action:
            tensorActs.append(1)
        else:
            tensorActs.append(0)
    tensorActs = torch.tensor(tensorActs)
    return tensorActs

class QValues():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @staticmethod
    def get_current(policy_net, states, actions):
        return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))
    
    @staticmethod        
    def get_next(target_net, next_states):
        final_state_locations = next_states.flatten(start_dim=1) \
                                   .max(dim=1)[0].eq(0).type(torch.bool)
        non_final_state_locations = (final_state_locations == False)
        non_final_states = next_states[non_final_state_locations]
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(QValues.device)
        values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach()
        return values

wait_time = 0.15
batch_size = 256
gamma = 0.99
eps_start = 0.99
eps_end = 0.1
decay = 0.0001
target_update = 10
memory_size = 10000
lr = 0.001
num_episodes = 10000

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

time.sleep(3)
input()

k = 0
prev_reward = 0
for episode in range(num_episodes):
    k += 1
    em.reset()
    state = em.get_state()
    print("Starting episode ", end="")
    print(k)
    print("Exploration rate: ", end="")
    print(agent.strategy.get_exploration_rate(agent.current_step))
    for timestep in count():
        action = agent.select_action(state, policy_net)
        reward = em.take_action(action)
        next_state = em.get_state()
        action = tensorize(action)
        memory.push(Experience(state, action, next_state, reward - prev_reward));
        state = next_state
        prev_reward = reward
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
            print("Reward was: " + str(em.env.currentReward))
            episode_scores.append(em.env.currentReward)
            plot(episode_scores, 100)            
            if episode % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())
            break
em.close()
