import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim


import numpy as np

from collections import deque, namedtuple

from base.agent_base import AgentBase

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
    
class Actor(nn.Module):
    def __init__(self, state_size, action_size, max_action, hidden_size = 400):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_size)
        self.max_action = max_action
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.max_action * torch.tanh(self.fc3(x))
        return x
        
    

class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

        self.fc4 = nn.Linear(state_size + action_size, 400)
        self.fc5 = nn.Linear(400, 300)
        self.fc6 = nn.Linear(300, 1)

        
    def forward(self, x, u):
        xu = torch.cat([x, u], 1)
        
        x1 = F.relu(self.fc1(xu))
        x1 = F.relu(self.fc2(x1))
        x1 = self.fc3(x1)
        
        x2 = F.relu(self.fc4(xu))
        x2 = F.relu(self.fc5(x2))
        x2 = self.fc6(x2)
        
        return x1, x2
    
    def q1(self, x, u):
        xu = torch.cat([x, u], 1)
        
        x1 = F.relu(self.fc1(xu))
        x1 = F.relu(self.fc2(x1))
        x1 = self.fc3(x1)
        
        return x1
    

class TD3Agent(AgentBase):
    def __init__(self, state_size, action_size, max_action, 
            batch_size = 100, 
            discount = 0.99, 
            tau = 0.005, 
            policy_noise = 0.2, 
            noise_clip = 0.5, 
            policy_freq = 2,
            total_timesteps = 1000
        ):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(state_size, action_size, max_action).to(self.device)
        self.actor_target = Actor(state_size, action_size, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)

        self.critic = Critic(state_size, action_size).to(self.device)
        self.critic_target = Critic(state_size, action_size).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.max_action = max_action
        self.batch_size = batch_size

        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_timesteps = total_timesteps
        self.current_timestep = 0

        self.memory = MemoryReply()


    def init(self):
        pass
    
    def get_action(self, state):
        state = torch.Tensor(state.reshape(1, -1)).to(self.device)

        return self.actor(state).data.numpy().flatten()


    def push(self, state, action, reward, next_state, done):
        return super().push(state, action, reward, next_state, done)


    def train(self, iter):
        
        for it in range(iter):
            # Get transitions from replay buffer (state, action, reward, next_state, done)
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = self.memory.sample(self.batch_size)

            state = torch.Tensor(state).to(self.device)
            action = torch.Tensor(action).to(self.device)
            reward = torch.Tensor(reward).to(self.device)
            next_state = torch.Tensor(next_state).to(self.device)
            done = torch.Tensor(done).to(self.device)

            # from next state s', the actor target plays the next action a'
            next_action = self.actor_target(next_state)

            # add gaussian noise to the next action a' and clamp it to a legal range
            noise = torch.Tensor(batch_action).data.normal_(0, self.policy_noise).to(self.device)
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            # the two critic targets take each the couple (s', a') as input and return two Q values, Qt1(s', a') and Qt2(s', a') as outputs
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)

            # keep the minimum of these two Q-Values
            target_Q = torch.min(target_Q1, target_Q2)

            # get the final target of the two critic models, which is: Qt = r + gamma * min(Qt1, Qt2)
            target_Q = reward + ((1 - done) * self.discount * target_Q).detach()

            # two critic models take each the couple (s, a) as input and return two Q values
            current_Q1, current_Q2 = self.critic(state, action)

            # compute the loss coming from the two critic models
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # backpropagate this critic loss and update the parameters of the two critic models with a SGD optimizer
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # once every two iterations, update the actor model by performing gradient ascent on the output of the first critic model

            if it % self.policy_freq == 0:
                actor_loss = -self.critic.q1(state, self.actor(state)).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        
    def save(self):
        return torch.save(self.actor.state_dict(), './models/td3_actor.pth'), torch.save(self.critic.state_dict(), './models/td3_critic.pth')


    def load(self):
        return self.actor.load_state_dict(torch.load('./models/td3_actor.pth')), self.critic.load_state_dict(torch.load('./models/td3_critic.pth'))

    

    
