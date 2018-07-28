import gym
import multiprocessing as mp
import torch.nn as nn
import torch
import numpy as np

# Env Settings
ENV_NAME = 'CartPole-v0'
THREAD_CNT = 1 #mp.cpu_count()
TRAIN_MODE = True
RENDER = True
MAX_EPISODE = 100
UPDATE_INTERVAL = 4 # n-step
GAMMA = 0.99

class Net(nn.Module):

    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()

        self.s_dim = s_dim
        self.s_dim = a_dim

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
        action_prob, critic_val = self.forward(s)
        return np.random.choice(self.s_dim, p=action_prob.data.numpy()), action_prob, critic_val

    def forward(self, s):
        return self.actor.forward(s), self.critic.forward(s)

    def get_loss(self, a_prob_buffer, c_buffer, c_target_buffer):

        pass

class SharedOptimizer(torch.optim.RMSprop):
    def __init__(self, params):
        super(SharedOptimizer, self).__init__(params)


class Worker(mp.Process):  # threads
    def __init__(self, g_net, g_opt, g_epi, s_dim, a_dim, id):
        super(Worker, self).__init__()

        self.env = gym.make(ENV_NAME)
        self.env.reset()

        self.id = id # worker number

        self.g_net, self.g_opt, self.g_epi = g_net, g_opt, g_epi
        self.l_net = Net(s_dim, a_dim)

    def run(self):

        while self.g_epi.value < MAX_EPISODE: # episode loop

            s = self.env.reset()
            l_step = 0
            buffer_s, buffer_a, buffer_r, buffer_c = [], [], [], [] # buffer for n-Step
            episode_reward = 0

            while True: # env loop

                a, a_prob, c = self.l_net.get_action(np2torch(s))
                s_, r, done, _ = self.env.step(a)

                if done:
                    r = -1
                episode_reward += r

                buffer_a_prob.append(a_prob)
                buffer_r.append(r)
                buffer_c.append(c) # get critic for advantage function

                if RENDER == True and self.id == 0:
                    self.env.render()

                # Update
                if l_step % UPDATE_INTERVAL == 0 or done:
                    self.update(buffer_a_prob, buffer_r, buffer_c, s_, done)

                    buffer_a_prob, buffer_r, buffer_c = [], [], []

                    if done:
                        with self.g_epi.get_lock(): # Async task
                            self.g_epi.value += 1

                        print('Worker {} | Episode : {} | Episode_Reward : {}'.format(self.id, self.g_epi.value, episode_reward))
                        break
                else:
                    s = s_

    def push_and_pull(self, buffer_a_prob, buffer_r, buffer_c, s_, done):
        if done: # critic for final
            s__critic = 0.
        else:
            s__critic, _ = self.l_net.forward(s_)

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
        loss_actor, loss_critic = self.l_net.get_loss(buffer_a_prob, buffer_c, buffer_c_target)

        # 3. Update global net with local gradient
        self.g_opt.zero_grad()
        loss_critic.backward()
        loss_actor.backward()
        for lp, gp in zip(self.l_net.parameters(), self.g_net.parameters()):
            gp._grad = lp.grad
        self.g_opt.step()

        # 4. Copy global net to local net
        self.l_net.load_state_dict(self.g_net.state_dict())

def np2torch(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)




if __name__ == '__main__':
    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    global_net = Net(state_dim, action_dim)
    global_net.share_memory()

    global_optimizer = SharedOptimizer(global_net.parameters())

    # Set Global Variables
    global_episode = mp.Value('i', 0)

    workers = [Worker(global_net, global_optimizer, global_episode, state_dim, action_dim, i) for i in range(THREAD_CNT)]
    [w.start() for w in workers]