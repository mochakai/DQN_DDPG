import sys
import math
import random
import json

import gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(hidden_size, output_size)
        self.out.weight.data.normal_(0, 0.1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        actions_value = self.out(x)
        return actions_value 

class Agent(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.eval_net = Net(self.state_space_dim, 256, self.action_space_dim).to(device)
        self.target_net = Net(self.state_space_dim, 256, self.action_space_dim).to(device)
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()
        self.buffer = []
        self.steps = 0
        self.learning_step = 0
        
    def act(self, s0):
        self.steps += 1
        self.epsi_high = self.epsi_high * self.decay if self.epsi_high > self.epsi_low else self.epsi_low
        if random.random() < self.epsi_high:
            a0 = random.randrange(self.action_space_dim)
        else:
            s0 = torch.tensor(s0, dtype=torch.float).view(1,-1)
            action = self.eval_net(s0.to(device))
            a0 = torch.argmax(action).item()
        return a0

    def put(self, *transition):
        if len( self.buffer)==self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)
        
    def learn(self):
        if (len(self.buffer)) < self.batch_size:
            return
            
        # target parameter update
        if self.learning_step % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learning_step += 1
        
        samples = random.sample( self.buffer, self.batch_size)
        s0, a0, r1, s1 = zip(*samples)
        s0 = torch.tensor( s0, dtype=torch.float).to(device)
        a0 = torch.tensor( a0, dtype=torch.long).view(self.batch_size, -1).to(device)
        r1 = torch.tensor( r1, dtype=torch.float).view(self.batch_size, -1).to(device)
        s1 = torch.tensor( s1, dtype=torch.float).to(device)
        
        y_true = r1 + self.gamma * torch.max( self.target_net(s1).detach(), dim=1)[0].view(self.batch_size, -1)
        y_pred = self.eval_net(s0).gather(1, a0)
        
        loss = self.loss_fn(y_pred, y_true)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def run_dqn(param, game_id='CartPole-v0'):
    env = gym.make(game_id)
    # env = env.unwrapped
    param['state_space_dim'] = env.observation_space.shape[0]
    param['action_space_dim'] = env.action_space.n
    agent = Agent(**params)
    score = []
    mean = []
    for episode in range(1000):
        s0 = env.reset()
        total_reward = 1
        while True:
            a0 = agent.act(s0)
            s1, r1, done, _ = env.step(a0)

            r1 = -1 if done else r1
            agent.put(s0, a0, r1, s1)
            if done:
                print('Episode: ', episode,
                        '| reward: ', round(total_reward, 2))
                break

            total_reward += r1
            s0 = s1
            agent.learn()
            
        score.append(total_reward)
        mean.append( sum(score[-100:])/100)
        if episode % 100 == 0:
            record_dict = {
                'score': score,
                'mean_score': mean,
            }
            with open('DQN_record.json', 'w') as f:
                json.dump(record_dict, f)

params = {
    'gamma': 0.95,
    'epsi_high': 1,
    'epsi_low': 0.01,
    'decay': 0.995, 
    'lr': 0.0005,
    'capacity': 5000,
    'batch_size': 128,
    'target_update_frequency': 50,
}

if __name__ == "__main__":
    run_dqn(params, 'CartPole-v0')

    sys.stdout.flush()
    sys.exit()
