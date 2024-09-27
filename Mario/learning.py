import mario
import utils
import gym
from mario import Mario, MarioNet, LazyMemmapStorage
from utils import SkipFrame, MetricLogger, ResizeObservation, GrayScaleObservation
import torch
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from pathlib import Path
import random, datetime, os

def main():
    if gym.__version__ < '0.26':
        env = gym_super_mario_bros.make("SuperMarioBros-1-1-v3", new_step_api=True)
    else:
        env = gym_super_mario_bros.make("SuperMarioBros-1-1-v3", render_mode='rgb', apply_api_compatibility=True)

    # 상태 공간을 2가지로 제한하기
    #   0. 오른쪽으로 걷기
    #   1. 오른쪽으로 점프하기
    env = JoypadSpace(env, [["right"], ["right", "A"]])

    env.reset()
    next_state, reward, done, trunc, info = env.step(action=0)
    print(f"{next_state.shape},\n {reward},\n {done},\n {info}")

    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")
    print()


    save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True)

    mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)
    logger = MetricLogger(save_dir)

    
    episodes = 40000
    for e in range(episodes):

        state = env.reset()

        # 게임을 실행시켜봅시다!
        while True:
            # 현재 상태에서 에이전트 실행하기
            action = mario.act(state)
            # 에이전트가 액션 수행하기
            next_state, reward, done, trunc, info = env.step(action)
            # 기억하기
            mario.cache(state, next_state, action, reward, done)

            # 배우기
            q, loss = mario.learn()

            # 기록하기
            logger.log_step(reward, loss, q)

            # 상태 업데이트하기
            state = next_state

            # 게임이 끝났는지 확인하기
            if done or info["flag_get"]:
                break

        logger.log_episode()

        if (e % 20 == 0) or (e == episodes - 1):
            logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)

if __name__ == "__main__":
    main()