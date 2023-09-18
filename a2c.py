import torch
import torch.nn as nn

import random
import numpy as np
from collections import deque, namedtuple
from torch import optim


class A2C(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(A2C, self).__init__()
        self.actor_fc1 = nn.Linear(state_size, hidden_size)
        self.actor_fc2 = nn.Linear(hidden_size, hidden_size)
        self.actor_fc3 = nn.Linear(hidden_size, action_size)
        self.critic_fc1 = nn.Linear(state_size, hidden_size)
        self.critic_fc2 = nn.Linear(hidden_size, hidden_size)
        self.critic_fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        
        actor_x = torch.relu(self.actor_fc1(x))
        actor_x = torch.relu(self.actor_fc2(actor_x))
        actor_x = self.actor_fc3(actor_x)

        critic_x = torch.relu(self.critic_fc1(x))
        critic_x = torch.relu(self.critic_fc2(critic_x))
        critic_x = self.critic_fc3(critic_x)
        return actor_x, critic_x
    
    
class MemoryReply:
    def __init__(self):
        self.memory = deque([], maxlen=10000)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.memory)
    

class A2CAgent:
    def __init__(self, state_size, action_size, lr, gamma, epsilon, hidden_size, batch_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.model = A2C(state_size, action_size, hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = MemoryReply()

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)

        state = self.state_to_tensor(state)

        with torch.no_grad():
            actor_outputs, _ = self.model(state)
            return actor_outputs.argmax(dim=1).item()
        
    def state_to_tensor(self, state):
        return torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

    def update(self):
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)
        actor_outputs, critic_outputs = self.model(states)
        next_actor_outputs, next_critic_outputs = self.model(next_states)
        td_targets = rewards + self.gamma * next_critic_outputs * (1 - dones)
        td_errors = td_targets - critic_outputs
        actor_loss = -(actor_outputs.log_softmax(dim=1) * td_errors.detach()).sum(dim=1).mean()
        critic_loss = td_errors.pow(2).mean()
        loss = actor_loss + critic_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def push(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def save(self, max_total_player_games ):
        torch.save(self.model.state_dict(), f'./models/a2c-{max_total_player_games}.pth')
        