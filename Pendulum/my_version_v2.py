import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
torch.set_printoptions(linewidth=400, sci_mode=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        """transition 저장"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def sample_tensor(self, batch_size):
        sample_mini_batch = random.sample(self.memory, batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = [], [], [], [], []

        for t in sample_mini_batch:
            s, a, r, s_prime, d = t
            state_batch.append(s)
            action_batch.append([a]) # have to add bracket to pad the extra dimension
            reward_batch.append([r]) # have to add bracket to pad the extra dimension
            next_state_batch.append(s_prime)
            done_batch.append([d]) # have to add bracket to pad the extra dimension

        s_batch = torch.tensor(state_batch, dtype=torch.float32, device=device)
        a_batch = torch.tensor(action_batch, dtype=torch.int64, device=device)
        r_batch = torch.tensor(reward_batch, dtype=torch.float32, device=device)
        s_prime_batch = torch.tensor(next_state_batch, dtype=torch.float32, device=device)
        done_batch = torch.tensor(done_batch, dtype=torch.float32, device=device)

        # reverse done because of math equation
        reversedone = 1 - done_batch #.unsqueeze(1)

        return s_batch, a_batch, r_batch, s_prime_batch, reversedone

    def __len__(self):
        return len(self.memory)
    
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class PendulumAgent():
    def __init__(self, env):
        self.env = env
        self.n_observations = 3
        self.n_actions = 9

        # 하이퍼파라미터
        self.BATCH_SIZE = 3 #128
        self.GAMMA = 0.98 #0.99
        self.TAU = 0.01 #0.005
        self.LR = 0.0001 #1e-4

        # 네트워크 및 메모리 초기화
        self.policy_net = DQN(self.n_observations, self.n_actions).to(device)
        self.target_net = DQN(self.n_observations, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.memory = ReplayMemory(100000)

        self.steps_done = 0
        self.EPS_END = 0.001 #0.05
        self.EPS_START = 1.0 #0.9
        self.EPS_DECAY = 100000

        # 학습 기록
        self.episode_durations = []
        self.episode_rewards = []
        self.cumulative_reward = 0

        self.actionvaluelist = np.array([-2, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])

    def select_action(self, state):
        rand = np.random.uniform(0.0, 1.0)
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * np.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1

        if rand > eps_threshold:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
                policy_out = self.policy_net(state_tensor)
                action_idx = torch.argmax(policy_out)
                action_value = self.actionvaluelist[action_idx]
        else:
            action_idx = np.random.randint(low=0, high=9)
            action_value = self.actionvaluelist[action_idx]

        return action_idx, action_value, eps_threshold

    def optimize_model_q_policy(self):
        s_batch, a_batch, r_batch, s_prime_batch, reversedone = self.memory.sample_tensor(self.BATCH_SIZE)

        # Get state action value
        Q_a_dist = agent.policy_net(s_batch)
        Q_a = Q_a_dist.gather(1, a_batch)
        
        # Get target (next state action value)
        with torch.no_grad():
            next_state_values_dist = agent.target_net(s_prime_batch)
            next_state_values = next_state_values_dist.max(1)[0].unsqueeze(1)

        expected_state_action_values = next_state_values * agent.GAMMA * reversedone + r_batch # torch.Size([200, 1])
        q_loss = F.smooth_l1_loss(Q_a, expected_state_action_values)
        loss_output = q_loss.item()
        self.optimizer.zero_grad()
        q_loss.mean().backward()
        self.optimizer.step()

        return loss_output

    def optimize_model_q_target(self):

         for param_target, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.TAU) + param.data * self.TAU)
        # target_net_state_dict = self.target_net.state_dict()
        # policy_net_state_dict = self.policy_net.state_dict()
        
        # for key in policy_net_state_dict:
        #     target_net_state_dict[key] = policy_net_state_dict[key] * self.TAU + target_net_state_dict[key] * (1.0 - self.TAU)
        # self.target_net.load_state_dict(target_net_state_dict)

    def learning(self, num_episodes):
        for epi in range(num_episodes):
            # print(f"In eps: [{epi}]")

            state, info = self.env.reset()
            acc_reward = 0
            done = False

            while not done:
                action, real_action, eps = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step([real_action])
                done = terminated or truncated

                self.memory.push(state, action, reward, next_state, done)

                acc_reward += reward
                state = next_state

                # Q update
                if self.memory.__len__() > 1000:
                    loss = self.optimize_model_q_policy()
                    self.optimize_model_q_target()

            if self.memory.__len__() > 1000 and epi % 10 == 0:
                print(f"In eps: [{epi}]")
                print(f"loss: [{loss:.3f}], epsilon: [{eps:.3f}]")

        torch.save(self.policy_net.state_dict(), 'final_pendulum_policy_net.pth')
        print('Complete')


def control():
    env = gym.make('Pendulum-v1', g=9.81, render_mode = 'rgb_array')
    # env = gym.wrappers.TimeLimit(env, max_episode_steps=20000)
    env = RecordVideo(env, 'pendulum_video', episode_trigger=lambda episode_number: True)
    n_actions = 9
    n_observations = env.observation_space.shape[0]

    model = DQN(n_observations, n_actions).to(device)
    model.load_state_dict(torch.load('final_pendulum_policy_net.pth', weights_only=True))
    model.eval()

    done = False
    env.reset()
    state, info = env.reset()

    while not done:
        env.render()
        # 모델을 사용하여 행동 선택
        with torch.no_grad():

            state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
            policy_out = agent.policy_net(state_tensor)
            action_idx = torch.argmax(policy_out)
            real_action = agent.actionvaluelist[action_idx]

        # 환경에서 행동 수행
        next_state, reward, terminated, truncated, _ = env.step([real_action])
        done = terminated or truncated

        # 상태 업데이트
        state = next_state
    env.close()


if __name__ == "__main__":
    env = gym.make('Pendulum-v1', g=9.81)
    agent = PendulumAgent(env)
    agent.learning(1500)

    control()












    # ----------------------------------------------test code -------------------------------------------------------------

    # state, _ = env.reset()
    # action_idx, action_value, eps_threshold = agent.select_action(state)
    # print(f"==>> action_idx: {action_idx}")
    # print(f"==>> action_value: {action_value}")
    # print(f"==>> eps_threshold: {eps_threshold}")

    # for i in range(1000):
    #     state = np.random.uniform(0,1.0, size=(3,))
    #     # action = np.random.randint(9, size=1)
    #     action, _, _ = agent.select_action(state)
    #     print(f"==>> action: {action}")
    #     reward = np.random.uniform(size=(1,))
    #     next_state = np.random.uniform(0,1.0, size=(3,))
    #     done = False

    #     agent.memory.push(state, action, reward, next_state, done)

    # sample_mini_batch = agent.memory.sample(5)
    # state_batch, action_batch, reward_batch, next_state_batch, done_batch = [], [], [], [], []

    # for t in sample_mini_batch:
    #     s, a, r, s_prime, d = t
    #     state_batch.append(s)
    #     action_batch.append([a])
    #     reward_batch.append(r)
    #     next_state_batch.append(s_prime)
    #     done_batch.append(d)

    # s_batch = torch.tensor(state_batch, dtype=torch.float32, device=device)
    # print(f"==>> s_batch.shape: {s_batch.shape}")
    # a_batch = torch.tensor(action_batch, dtype=torch.int64, device=device)
    # r_batch = torch.tensor(reward_batch, dtype=torch.float32, device=device)
    # s_prime_batch = torch.tensor(next_state_batch, dtype=torch.float32, device=device)
    # done_batch = torch.tensor(done_batch, dtype=torch.float32, device=device)
    # print(f"==>> a_batch.shape: {a_batch.shape}")
    # print(f"==>> r_batch.shape: {r_batch.shape}")
    # print(f"==>> s_prime_batch.shape: {s_prime_batch.shape}")
    # print(f"==>> done_batch.shape: {done_batch.shape}")


    # # Get state action value
    # Q_a_dist = agent.policy_net(s_batch)
    # Q_a = Q_a_dist.gather(1, a_batch)
    
    # # Get target (next state action value)
    # with torch.no_grad():
    #     next_state_values_dist = agent.target_net(s_prime_batch)
    #     print(f"==>> next_state_values_dist: {next_state_values_dist}")
    #     next_state_values = next_state_values_dist.max(1)[0].unsqueeze(1)
    #     print(f"==>> next_state_values.shape: {next_state_values.shape}")
    #     print(f"==>> next_state_values: {next_state_values}")

    #     reversedone = 1 - done_batch.unsqueeze(1)
    #     print(f"==>> reversedone.shape: {reversedone.shape}")

    #     expected_state_action_values = next_state_values * agent.GAMMA * reversedone + r_batch # torch.Size([200, 1])
    #     print(f"==>> expected_state_action_values.shape: {expected_state_action_values.shape}")
    #     print(f"==>> expected_state_action_values: {expected_state_action_values}")

    #     q_loss = F.smooth_l1_loss(Q_a, expected_state_action_values)
    #     print(f"==>> q_loss: {q_loss}")
