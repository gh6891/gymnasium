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

# env = gym.make('Pendulum-v1', g=9.81)
# env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)

# # GPU를 사용할 경우
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


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

class PendulumAgent():
    def __init__(self, env, n_observations, n_actions):
        self.env = env
        self.n_observations = n_observations
        self.n_actions = n_actions

        # 하이퍼파라미터
        self.BATCH_SIZE = 200 #128
        self.GAMMA = 0.98 #0.99
        self.TAU = 0.01 #0.005
        self.LR = 0.01 #1e-4

        # 네트워크 및 메모리 초기화
        self.policy_net = DQN(n_observations, n_actions).to(device)
        self.target_net = DQN(n_observations, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.memory = ReplayMemory(100000)

        self.steps_done = 0
        self.EPS_END = 0.05 #0.05
        self.EPS_START = 1.0 #0.9
        self.EPS_DECAY = 10000 

        # 학습 기록
        self.episode_durations = []
        self.episode_rewards = []
        self.cumulative_reward = 0

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        # if eps_threshold < self.EPS_END + self.EPS_END / 2:
        #     print(eps_threshold)
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state)
        else:
            return torch.tensor([self.env.action_space.sample()], device=device, dtype=torch.float32)
        
    def learning(self, num_episodes):
        for i_episode in range(num_episodes):
            state, info = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            for t in count():
                action = self.select_action(state)
                observation, reward, terminated, truncated, _ = self.env.step([action.item()])
                done = terminated or truncated
                done = 0.0 if done else 1.0
                reward = torch.tensor([reward], device=device)
                done = torch.tensor([done], device =device)
                self.cumulative_reward += reward.item()

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

                self.memory.push(state, action, next_state, reward, done)
                state = next_state
                self.optimize_model()

                # Soft update for target network
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * self.TAU + target_net_state_dict[key] * (1 - self.TAU)
                self.target_net.load_state_dict(target_net_state_dict)
                if done:
                    self.episode_rewards.append(self.cumulative_reward)
                    if len(self.episode_rewards) >= 100:
                        means = sum(self.episode_rewards[-100:]) / 100  # 최근 100개의 보상 평균
                    else:
                        means = sum(self.episode_rewards) / len(self.episode_rewards)  # 현재까지의 평균

                    print("episode : ", i_episode, "reward : ", self.cumulative_reward, "mean : ", means)
                    self.cumulative_reward = 0
                    self.episode_durations.append(t + 1)
                    break
        torch.save(self.policy_net.state_dict(), 'final_pendulum_policy_net.pth')
        print('Complete')

    
    def plot_durations(self, show_result=False):
        plt.figure(1)
        rewards_t = torch.tensor(self.episode_rewards, dtype = torch.float)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
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

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            print("memory's len needs more size")
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        # non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        next_state_batch = torch.cat(batch.next_state)
        state_batch = torch.cat(batch.state)
        # action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        done_batch = torch.cat(batch.done)
        state_action_values = self.policy_net(state_batch)
        # print(state_action_values.shape)
        # next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values = self.target_net(next_state_batch)
            # next_state_values[non_fiklnal_mask] = self.target_net(non_final_next_states).max(1).values
        # 기대 Q 값 계산
        
        expected_state_action_values = next_state_values * self.GAMMA * done_batch + reward_batch
        # Huber 손실 계산
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        #print(loss.item())
        # 모델 최적화
        self.optimizer.zero_grad()
        loss.backward()
        # 변화도 클리핑 바꿔치기
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

def control():
    env = gym.make('Pendulum-v1', g=9.81, render_mode = 'rgb_array')
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
    # env = RecordVideo(env, 'pendulum_video', episode_trigger=lambda episode_number: True)
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
    env = gym.make('Pendulum-v1', g=9.81)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)

    # GPU를 사용할 경우
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    n_actions = env.action_space.shape[0]
    n_observations = env.observation_space.shape[0]

    agent = PendulumAgent(env, n_observations, n_actions)
    agent.learning(num_episodes=1000)