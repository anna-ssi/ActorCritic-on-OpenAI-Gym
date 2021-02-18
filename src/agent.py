import itertools

import torch
import torch.nn.functional as F
from torch.optim import Adam

from .memory import ReplayMemory
from .models import Actor, Critic


class Agent:
    def __init__(self, env, args):
        self.env = env
        self.action_space = self.env.action_space
        self.args = args
        self.gamma = self.args.gamma
        self.batch_size = self.args.batch_size
        self.device = torch.device("cuda" if self.args.cuda else "cpu")
        self.num_inputs = self.env.observation_space.shape[0]

        self.critic = Critic(
            self.num_inputs, self.args.hidden_size).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.args.lr)

        self.actor = Actor(self.num_inputs, self.args.hidden_size,
                           self.action_space.n).to(self.device)
        self.actor_optim = Adam(self.actor.parameters(), lr=self.args.lr)

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action, _ = self.actor.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, R):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(
            batch_size=self.batch_size, device=self.device)
        R = reward_batch + self.gamma * R

        with torch.no_grad():
            next_state_actions, _ = self.actor.sample(next_state_batch)
            next_rewards = self.critic(
                next_state_batch, next_state_actions.unsqueeze(1))

            next_q_value = reward_batch + self.gamma * mask_batch * next_rewards

        # Value loss
        q_value = self.critic(state_batch, action_batch.unsqueeze(1))
        value_loss = F.mse_loss(q_value, next_q_value)

        # Policy loss
        actions, log_actions = self.actor.sample(state_batch)
        actions_value = self.critic(state_batch, actions.unsqueeze(1))
        advantage = R - actions_value

        policy_loss = (log_actions * advantage).mean()

        self.critic_optim.zero_grad()
        self.actor_optim.zero_grad()

        value_loss.backward()
        policy_loss.backward()

        self.critic_optim.step()
        self.actor_optim.step()

        return value_loss.item(), policy_loss.item(), R

    def evaluate(self):
        avg_reward = 0.
        episodes = 2
        for _ in range(episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = self.select_action(state)
                self.env.render()

                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward

                state = next_state
            avg_reward += episode_reward
        avg_reward /= episodes

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(
            episodes, round(avg_reward, 2)))
        print("----------------------------------------")

    def run(self):
        memory = ReplayMemory(self.args.replay_size, self.args.seed)

        # Training Loop
        total_steps = 0
        updates = 0

        for i_episode in itertools.count(1):
            episode_reward = 0
            episode_steps = 0
            done = False
            state = self.env.reset()
            losses = []

            while not done:
                reward = 0
                if self.args.start_steps > total_steps:
                    action = self.env.action_space.sample()  # Sample random action
                else:
                    # Sample action from policy
                    action = self.select_action(state)

                if len(memory) > self.args.batch_size:
                    # Number of updates per step in environment
                    for i in range(self.args.updates_per_step):
                        # Update parameters of all the networks
                        value_loss, policy_loss, R = self.update_parameters(
                            memory, reward)
                        losses.append((value_loss, policy_loss))
                        updates += 1
                        reward += R

                next_state, reward, done, _ = self.env.step(action)  # Step
                episode_steps += 1
                total_steps += 1
                episode_reward += reward

                mask = 1 if episode_steps == self.env._max_episode_steps else float(
                    not done)

                memory.push(state, action, reward, next_state, mask)

                state = next_state

            if total_steps > self.args.num_steps:
                break

            print("Episode: {}, episode steps: {}, reward: {}".format(
                i_episode, episode_steps, episode_reward))

            if i_episode % self.args.eval_every == 0:
                self.evaluate()

        self.env.close()
