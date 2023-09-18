import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

from config import Config

from collections import deque, namedtuple


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

class MemoryReply:
    def __init__(self):
        self.memory = deque([], maxlen=10000)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = np.random.choice(len(self.memory), batch_size, replace=False)
        state, action, reward, next_state, done = zip(*[self.memory[i] for i in batch])
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.memory)


class DQNAgent:
    def __init__(self, state_size, action_size, load_model = False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.lr = Config.LR
        self.gamma = Config.GAMMA
        self.epsilon = Config.EPSILON
        self.batch_size = Config.BATCH_SIZE
        self.model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.memory = MemoryReply()

        if load_model:
            self.load()

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:

            state = np.expand_dims(state, 0)
            state = torch.FloatTensor(state).to(self.device)
            q_value = self.model(state)
            return q_value.argmax().item()
        
    def update(self):
        if len(self.memory) < self.batch_size:
            return
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)

        state = torch.FloatTensor(state).to(self.device)
        action = torch.LongTensor(action).unsqueeze(-1).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(-1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(-1).to(self.device)
        
        q_value = self.model(state)
        next_q_value = self.model(next_state)

        q_value = q_value.gather(1, action)
        next_q_value = torch.max(next_q_value, dim=-1, keepdim=True)[0]

        expected_q_value = reward + self.gamma * next_q_value * (1 - done)
        
        loss = F.mse_loss(q_value, expected_q_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def push(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def save(self, total_player_games):
        torch.save(self.model.state_dict(), f'./models/dqn_{total_player_games}.pth')

    def load(self, total_player_games):
        self.model.load_state_dict(torch.load(f'./models/dqn_{total_player_games}.pth'))
        self.model.eval()