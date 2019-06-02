import gym
import math
import random
import json

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ActorNetwork(nn.Module):
    """
    A network for actor
    """
    def __init__(self, state_dim, hidden_size, output_size, output_act):
        super(ActorNetwork, self).__init__()
        hidden_size2 = int(hidden_size * 0.75)
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(hidden_size, hidden_size2)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.fc3.weight.data.normal_(0, 0.1)
        # activation function for the output
        self.output_act = output_act

    def __call__(self, state):
        out = nn.functional.relu(self.fc1(state))
        out = nn.functional.relu(self.fc2(out))
        out = self.output_act(self.fc3(out))
        return out


class CriticNetwork(nn.Module):
    """
    A network for critic
    """
    def __init__(self, state_dim, hidden_size, action_dim, output_size=1):
        super(CriticNetwork, self).__init__()
        hidden_size2 = int(hidden_size * 0.75)
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(hidden_size + action_dim, hidden_size2)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.fc3.weight.data.normal_(0, 0.1)

    def __call__(self, state, action):
        out = nn.functional.relu(self.fc1(state))
        out = torch.cat([out, action.type_as(out)], 1)
        out = nn.functional.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class Agent(object):
    def __init__(self, **kwargs ):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.actor = ActorNetwork(self.state_space_dim, 400, self.action_space_dim, nn.functional.tanh).to(device)
        self.critic = CriticNetwork(self.state_space_dim, 400, self.action_space_dim, 1).to(device)
        self.actor_target = ActorNetwork(self.state_space_dim, 400, self.action_space_dim, nn.functional.tanh).to(device)
        self.critic_target = CriticNetwork(self.state_space_dim, 400, self.action_space_dim, 1).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.loss_fn = nn.MSELoss()
        self.buffer = []
        self.steps = 0
        self.learning_step = 0
        
    def act(self, s0):
        s0 = torch.tensor(s0, dtype=torch.float).to(device)
        action = self.actor(s0)
        self.steps += 1
        epsilon = self.epsi_low + (self.epsi_high-self.epsi_low)*( math.exp(-1.0 * self.steps/self.decay))
        # add noise
        noise = np.random.randn(self.action_space_dim) * epsilon
        action += torch.tensor( noise, dtype=torch.float).to(device)
        return action
    
    def put(self, *transition):
        if len( self.buffer)==self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)
        
    def soft_update(self, target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(
                (1. - self.tau) * t.data + self.tau * s.data)

    def learn(self):
        if (len(self.buffer)) < self.batch_size:
            return
        
        samples = random.sample( self.buffer, self.batch_size)
        
        s0, a0, r1, s1 = zip(*samples)
        s0 = torch.tensor(s0, dtype=torch.float).to(device)
        a0 = torch.tensor(a0, dtype=torch.long).view(self.batch_size, -1).to(device)
        r1 = torch.tensor(r1, dtype=torch.float).view(self.batch_size, -1).to(device)
        s1 = torch.tensor(s1, dtype=torch.float).to(device)
        
        a1 = self.actor_target(s1)
        y_true = r1 + self.gamma * self.critic_target(s1, a1).detach()
        y_pred = self.critic(s0, a0)
        
        # update critic network
        self.critic_optimizer.zero_grad()
        critic_loss = self.loss_fn(y_pred, y_true)
        critic_loss.backward()
        self.critic_optimizer.step()

        # update actor network
        self.actor_optimizer.zero_grad()
        # the accurate action prediction
        action = self.actor(s0)
        # actor_loss is used to maximize the Q value for the predicted action
        actor_loss = - self.critic(s0, action)
        actor_loss = actor_loss.mean()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.learning_step += 1
        if self.learning_step % self.target_update_frequency == 0:
            self.soft_update(self.critic_target, self.critic)
            self.soft_update(self.actor_target, self.actor)
        

def run_ddpg(param, game_id='Pendulum-v0'):
    env = gym.make(game_id)
    param['state_space_dim'] = env.observation_space.shape[0]
    param['action_space_dim'] = env.action_space.shape[0]
    agent = Agent(**params)

    score = []
    mean = []
    for episode in range(3500):
        s0 = env.reset()
        total_reward = 1
        while True:
            a0 = agent.act(s0).data.cpu().numpy()
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
            with open('DDPG_record.json', 'w') as f:
                json.dump(record_dict, f)


params = {
    'gamma': 0.99,
    'epsi_high': 0.99,
    'epsi_low': 0.05,
    'decay': 500, 
    'actor_lr': 0.0001,
    'critic_lr': 0.001,
    'tau': 0.001,
    'target_update_frequency': 5,
    'capacity': 10000,
    'batch_size': 64,
}
if __name__ == "__main__":
    run_ddpg(params, 'Pendulum-v0')

    sys.stdout.flush()
    sys.exit()