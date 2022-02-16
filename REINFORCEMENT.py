import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt

import sys; sys.path.append('.')

lr = 0.0002
gamma = 0.98
eps = 1e-25

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(4,128)
        self.fc2 = nn.Linear(128,2)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x

    def put_data(self, item):
        self.data.append(item)

    def train_net(self):
        R = 0
        self.optimizer.zero_grad()
        for r, prob in self.data[::-1]:
            R = r + gamma * R
            loss = -torch.log(prob + eps) * R
            loss.backward()
        self.optimizer.step()
        self.data = []

def train():
    env = gym.make('CartPole-v1')
    pi = Policy()
    score = 0.0
    print_interval = 20
    num_episode = 10000
    scores = []


    for n_epi in range(num_episode):
        s = env.reset()
        env.render()
        done = False
        
        while not done:
            prob = pi(torch.from_numpy(s).float())
            m = Categorical(prob)
            a = m.sample()
            s_prime, r, done, info = env.step(a.item())
            pi.put_data((r,prob[a]))
            s = s_prime
            score += r
        
        pi.train_net()
        

        if n_epi % print_interval == 0 and n_epi!=0:
            print("# of episode : {}, avg score : {}".format(n_epi, score/print_interval))
            scores.append(score/print_interval)
            score = 0.0

        if n_epi == num_episode-1:
            torch.save(pi.state_dict(), 'model_REINFORCE.pt')

            fig, ax = plt.subplots(1,1)
            ax.grid()
            ax.plot(range(1,int(num_episode/print_interval)), scores)
            plt.show()
        

    env.close()

def inference():
    env = gym.make('CartPole-v1')
    
    pi = Policy()
    pi.load_state_dict(torch.load('model_REINFORCE.pt'))
    pi.eval()

    for n_epi in range(100):
        s = env.reset()
        env.render()
        done = False
        while not done:
            prob = pi(torch.from_numpy(s).float())
            m = Categorical(prob)
            a = m.sample()
            _, _, done, info = env.step(a.item())
    env.close()
            
def main():
    train()
    inference()
    

if __name__ == '__main__':
    main()