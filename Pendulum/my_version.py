import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# matplotlib 설정
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

env = gym.make('Pendulum-v1', g=9.81)
env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)

# GPU를 사용할 경우
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """transition 저장"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, n_actions)

    # 최적화 중에 다음 행동을 결정하기 위해서 하나의 요소 또는 배치를 이용해 호촐됩니다.
    # ([[left0exp,right0exp]...]) 를 반환합니다.
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
def select_action(state):
    #select_action but
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done = steps_done + 1

    if sample > eps_threshold:
        with torch.no_grad():
            # print("action : ", policy_net(state))
            return policy_net(state)
    else:
        # print("random : ", torch.tensor([env.action_space.sample()], device=device, dtype = torch.float32))
        return torch.tensor([env.action_space.sample()], device=device, dtype = torch.float32)
    
def plot_durations(show_result=False):
    plt.figure(1)
    rewards_t = torch.tensor(episode_rewards, dtype = torch.float)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('reward')
    plt.plot(rewards_t.numpy())
    # 100개의 에피소드 평균을 가져 와서 도표 그리기
    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # 도표가 업데이트되도록 잠시 멈춤
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def optimize_model():
    if len(memory) < BATCH_SIZE:
        print("memory's len needs more size")
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). 이것은 batch-array의 Transitions을 Transition의 batch-arrays로
    # 전환합니다.
    batch = Transition(*zip(*transitions))

    # 최종이 아닌 상태의 마스크를 계산하고 배치 요소를 연결합니다
    # (최종 상태는 시뮬레이션이 종료 된 이후의 상태)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Q(s_t, a) 계산 - 모델이 Q(s_t)를 계산하고, 취한 행동의 열을 선택합니다.
    # 이들은 policy_net에 따라 각 배치 상태에 대해 선택된 행동입니다.
    # print(policy_net(state_batch))
    # print("policy_net: ", policy_net(state_batch))
    # print("gather: ", policy_net(state_batch).gather(1, action_batch))
    state_action_values = policy_net(state_batch)

    # print(state_action_values)
    # 모든 다음 상태를 위한 V(s_{t+1}) 계산
    # non_final_next_states의 행동들에 대한 기대값은 "이전" target_net을 기반으로 계산됩니다.
    # max(1).values로 최고의 보상을 선택하십시오.
    # 이것은 마스크를 기반으로 병합되어 기대 상태 값을 갖거나 상태가 최종인 경우 0을 갖습니다.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # 기대 Q 값 계산
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    # Huber 손실 계산
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # 모델 최적화
    optimizer.zero_grad()
    loss.backward()
    # 변화도 클리핑 바꿔치기
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

def learning():
    for i_episode in range(num_episodes):
        # 환경과 상태 초기화
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            action = select_action(state)
            observation, reward, terminated, truncated, _ = env.step([action.item()])
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated
            cumulative_reward += reward.item()
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            memory.push(state, action, next_state, reward)

            state = next_state
            optimize_model()

            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_rewards.append(cumulative_reward)
                cumulative_reward = 0
                episode_durations.append(t + 1)
                plot_durations()
                break
    torch.save(policy_net.state_dict(), 'final_pendulum_policy_net.pth')
    print('Complete')
    plot_durations(show_result=True)
    plt.ioff()
    plt.show()
    
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 0.005
LR = 1e-4
n_actions = env.action_space.shape[0]
n_observations = env.observation_space.shape[0]


policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(100000)

step_done = 0
EPS_END = 0.05
EPS_START = 0.9
EPS_DECAY = 10000
steps_done = 0

num_episodes = 2000
episode_durations = []
cumulative_reward = 0
episode_rewards = []


def control():
    env = gym.make('Pendulum-v1', g=9.81, render_mode = 'rgb_array')
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
    env = RecordVideo(env, 'pendulum_video', episode_trigger=lambda episode_number: True)
    n_actions = env.action_space.shape[0]
    n_observations = env.observation_space.shape[0]

    model = DQN(n_observations, n_actions).to(device)
    model.load_state_dict(torch.load('final_pendulum_policy_net.pth'))
    model.eval()

    done = False
    env.reset()
    state, info = env.reset()
    state = torch.tensor(state, dtype = torch.float32, device = device).unsqueeze(0)
    print(state)
    while not done:
        env.render()
        # 모델을 사용하여 행동 선택
        with torch.no_grad():
            q_values = model(state)
            action = q_values  # 가장 높은 Q 값을 가진 행동 선택
        # 환경에서 행동 수행
        next_state, reward, terminated, truncated, _ = env.step([action.item()])
        done = terminated or truncated

        # 상태 업데이트
        state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
    env.close()

if __name__ == "__main__":
    learning()
    # control()