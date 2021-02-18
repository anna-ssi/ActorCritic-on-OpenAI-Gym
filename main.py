import argparse

import gym
import numpy as np
import torch

from agent import Agent


def main():
    parser = argparse.ArgumentParser(
        description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('--env-name', default="LunarLander-v2",
                        help='Mujoco Gym environment (default: HalfCheetah-v2)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--num_steps', type=int, default=1000000, metavar='N',
                        help='maximum number of steps (default: 1000000)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                        help='steps sampling random actions (default: 10000)')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--eval_every', type=int, default=50,
                        help='see the gameplay every n episodes.')
    parser.add_argument('--cuda', action="store_true",
                        help='run on CUDA (default: False)')
    args = parser.parse_args()

    # Environment
    env = gym.make(args.env_name)
    env.seed(args.seed)
    env.action_space.seed(args.seed)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    agent = Agent(env, args)
    agent.run()


if __name__ == '__main__':
    main()
