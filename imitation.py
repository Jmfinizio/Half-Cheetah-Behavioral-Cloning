import gymnasium as gym
from typing import Optional, Tuple, Union
from gymnasium import logger, spaces

import math
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Ensure dependencies are installed before running:
# pip install torch gymnasium[mujoco] numpy matplotlib

# Load expert data
file_path = "HalfCheetah_expert_data.pkl"
with open(file_path, "rb") as f:
    expert_data = pickle.load(f)[0]

print(expert_data.keys())
print("Number of data points:", len(expert_data['observation']))

# Convert expert data to tensors
states = torch.tensor(expert_data["observation"], dtype=torch.float32)
actions = torch.tensor(expert_data["action"], dtype=torch.float32)

# Define policy network
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Function to evaluate policy
def evaluate_policy(policy, env, episodes=10):
    total_rewards = []
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        for _ in range(1000):
            env.render()  # Add this line to show the simulation
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action = policy(state_tensor).detach().numpy()[0]
            state, reward, done, _, _ = env.step(action)
            episode_reward += reward
            if done:
                break
        total_rewards.append(episode_reward)
    env.close()  # Close environment after evaluation
    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    print(f"Evaluation Results: Mean Reward = {mean_reward:.2f}, Std Reward = {std_reward:.2f}")
    return mean_reward, std_reward

if __name__ == "__main__":
    # Initialize environment and policy
    env = gym.make("HalfCheetah-v5", render_mode = "human")
    state_dim = states.shape[1]
    action_dim = actions.shape[1]
    policy_bc = PolicyNet(state_dim, action_dim)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(policy_bc.parameters(), lr=1e-3)

    # Create dataset and dataloader
    dataset = TensorDataset(states, actions)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Train using Behavior Cloning
    num_epochs = 1000
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_states, batch_actions in dataloader:
            optimizer.zero_grad()
            predicted_actions = policy_bc(batch_states)
            loss = criterion(predicted_actions, batch_actions)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_states.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # Evaluate and save BC policy
    evaluate_policy(policy_bc, env)
    torch.save(policy_bc.state_dict(), "bc_policy.pth")
    print("Trained policy saved as bc_policy.pth")

    # DAgger Algorithm
    policy_dagger = PolicyNet(state_dim, action_dim)
    optimizer = torch.optim.Adam(policy_dagger.parameters(), lr=1e-3)

    dagger_iterations = 5
    epochs_per_iteration = 100
    rollout_episodes = 10
    max_steps = 2000

    aggregated_states = states.clone()
    aggregated_actions = actions.clone()

    for dagger_iter in range(dagger_iterations):
        print(f"\nDAgger Iteration {dagger_iter+1}/{dagger_iterations}")

        dataset = TensorDataset(aggregated_states, aggregated_actions)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        for epoch in range(epochs_per_iteration):
            total_loss = 0.0
            for batch_states, batch_actions in dataloader:
                optimizer.zero_grad()
                pred_actions = policy_dagger(batch_states)
                loss = criterion(pred_actions, batch_actions)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch_states.size(0)

            if (epoch + 1) % 5 == 0:
                avg_loss = total_loss / len(dataset)
                print(f"Epoch {epoch+1}/{epochs_per_iteration} | Loss: {avg_loss:.4f}")

        print("Collecting rollouts...")
        new_states, new_expert_actions = [], []
        for _ in range(rollout_episodes):
            state, _ = env.reset()
            done = False
            steps = 0
            while not done and steps < max_steps:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    action_tensor = policy_dagger(state_tensor)
                    action = action_tensor.squeeze(0).numpy()

                new_states.append(state)
                expert_idx = torch.randint(0, len(actions), (1,))
                expert_action = actions[expert_idx].squeeze(0).numpy()
                new_expert_actions.append(expert_action)

                state, _, done, _, _ = env.step(action)
                steps += 1

        new_states_tensor = torch.tensor(np.array(new_states), dtype=torch.float32)
        new_actions_tensor = torch.tensor(np.array(new_expert_actions), dtype=torch.float32)

        aggregated_states = torch.cat([aggregated_states, new_states_tensor])
        aggregated_actions = torch.cat([aggregated_actions, new_actions_tensor])

    # Save trained DAgger policy
    evaluate_policy(policy_dagger, env)
    torch.save(policy_dagger.state_dict(), "dagger_policy.pth")
    print("Trained policy saved as dagger_policy.pth")
