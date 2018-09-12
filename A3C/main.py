import gym
import multiprocessing as mp
import torch.nn as nn
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time

# I don't know
#import os
#os.environ["OMP_NUM_THREADS"] = "1"

# Env Settings
ENV_NAME = 'CartPole-v0'
THREAD_CNT = mp.cpu_count()
TRAIN_MODE = True
RENDER = True
MAX_EPISODE = 100000
UPDATE_INTERVAL = 10 # n-step
GAMMA = 0.9


class Net(nn.Module):

    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()

        self.s_dim = s_dim
        self.s_dim = a_dim

        self.dist = torch.distributions.Categorical

        self.actor = nn.Sequential(
            nn.Linear(s_dim, 100),
            nn.ReLU(),
            nn.Linear(100, a_dim),
            nn.Softmax(dim=-1) # 각 input 단위마다의 결과를 softmax 취함
        )
        self.critic = nn.Sequential(
            nn.Linear(s_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )

    def get_action(self, s):
        self.eval()
        action_prob, critic_val = self.forward(s)

        # dist = torch.distributions.Categorical(action_prob)
        # print('action_probs :', action_prob)
        # print('dist.probs :', dist.log_prob(torch.Tensor([0, 1])))
        # print('dist.sample :', dist.sample())

        return np.random.choice(self.s_dim, p=action_prob.data.numpy()), action_prob, critic_val

    def forward(self, s):
        return self.actor.forward(s), self.critic.forward(s)

    def get_loss(self, a_prob_buffer, c_buffer, c_target_buffer):
        self.train()

        c_buffer = torch.cat(c_buffer, dim=0).reshape(1, -1)
        c_target_buffer = np2torch(np.array(c_target_buffer))

        advantage = c_target_buffer - c_buffer

        critic_loss = advantage.pow(2).mean()


        a_prob_buffer = torch.cat(a_prob_buffer)
        actor_loss = -(a_prob_buffer * advantage.detach()).mean()

        return actor_loss, critic_loss


class SharedOptimizer(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, eps=1e-8,
                 weight_decay=0):
        super(SharedOptimizer, self).__init__(params, lr=lr, eps=eps, weight_decay=weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


class Worker(mp.Process):  # threads
    def __init__(self, g_net, g_opt, g_epi, s_dim, a_dim, id, g_q):
        super(Worker, self).__init__()

        self.env = gym.make(ENV_NAME).unwrapped
        self.env.reset()

        self.id = id # worker number

        self.g_net, self.g_opt, self.g_epi, self.g_q = g_net, g_opt, g_epi, g_q
        self.l_net = Net(s_dim, a_dim)

        self.l_net.load_state_dict(self.g_net.state_dict())


    def run(self):

        while self.g_epi.value < MAX_EPISODE: # episode loop
            s = self.env.reset()
            l_step = 0
            buffer_a_prob, buffer_r, buffer_c, buffer_s, buffer_a = [], [], [], [], [] # buffer for n-Step
            episode_reward = 0

            while True: # env loop
                a, a_prob, c = self.l_net.get_action(np2torch(s))
                s_, r, done, _ = self.env.step(a)
                #print(done)
                l_step += 1

                if done:
                    r = -1
                episode_reward += r

                m = torch.distributions.Categorical(a_prob)
                a_prob = m.log_prob(torch.Tensor([a]))


                buffer_a_prob.append(a_prob)
                buffer_r.append(r)
                buffer_c.append(c)
                buffer_s.append(s) # get critic for advantage function
                buffer_a.append(a) # get critic for advantage function

                if RENDER == True and self.id == 0:
                    self.env.render()

                # Update
                if l_step % UPDATE_INTERVAL == 0 or done:
                    self.push_and_pull(buffer_a_prob, buffer_r, buffer_c, s_, done, buffer_s, buffer_a)
                    buffer_a_prob, buffer_r, buffer_c, buffer_s, buffer_a = [], [], [], [], []

                    if done:
                        with self.g_epi.get_lock(): # Async task
                            self.g_epi.value += 1

                        self.g_q.put([self.g_epi.value, episode_reward])
                        #print('Worker {} | Episode : {} | Episode_Reward : {}'.format(self.id, self.g_epi.value, episode_reward))
                        break
                else:
                    s = s_

    def push_and_pull(self, buffer_a_prob, buffer_r, buffer_c, s_, done, buffer_s, buffer_a):

        if done: # critic for final
            s__critic = 0.
        else:
            _, s__critic = self.l_net.forward(np2torch(s_))
            s__critic = s__critic.data.numpy()[0]


        # 1. Preprocessing
        buffer_c_target = []
        buffer_c_target.append(s__critic)
        for r in buffer_r[::-1]:
            buffer_c_target.append(
                r + (GAMMA * buffer_c_target[-1])
            )
        buffer_c_target.reverse()
        buffer_c_target.pop()

        # 2. Get gradient from local net
        actor_loss, critic_loss = self.l_net.get_loss(buffer_a_prob, buffer_c, buffer_c_target)

        # 3. Update global net with local gradient

        self.g_opt.zero_grad()
        (critic_loss + actor_loss).backward()
        for lp, gp in zip(self.l_net.parameters(), self.g_net.parameters()):
            gp._grad =lp .grad
        self.g_opt.step()
        # 4. Copy global net to local net
        self.l_net.load_state_dict(self.g_net.state_dict())


def np2torch(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)


def initialize(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)



if __name__ == '__main__':
    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    global_net = Net(state_dim, action_dim)
    global_net.share_memory()

    global_net.apply(initialize)
    global_optimizer = SharedOptimizer(global_net.parameters(), lr=0.0001)

    # Set Global Variables
    global_episode = mp.Value('i', 0)
    global_queue = mp.Queue()

    workers = [Worker(global_net, global_optimizer, global_episode, state_dim, action_dim, i, global_queue) for i in range(THREAD_CNT)]
    [w.start() for w in workers]


    writer = SummaryWriter()
    while True:
        while not global_queue.empty():
            e = global_queue.get()
            writer.add_scalar('data/reward', e[1], e[0])

