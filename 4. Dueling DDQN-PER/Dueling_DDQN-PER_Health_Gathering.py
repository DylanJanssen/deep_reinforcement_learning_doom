import os
import math 
import random 
from vizdoom import *
import numpy as np 

import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.autograd as autograd 
import torch.nn.functional as F 

from collections import deque 

import matplotlib.pyplot as plt 
import itertools as it
from skimage import transform 

from tqdm import trange 
import time 


##################################################################################################
# HYPERPARAMETERS

# determine whether we can use CUDA
use_cuda = torch.cuda.is_available()
# lambda function to predefine Variable for using cuda 
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if use_cuda else autograd.Variable(*args, **kwargs) 

# configuration file path
config_file_path = "../scenarios/health_gathering.cfg" 
model_savefile = "./model-doom-health-gathering.pth"
training_rewards_savefile = "./training_rewards-health-gathering.csv"
skip_learning = True
load_model = True
testing_episodes = 200
render_testing_episodes = False
num_actions = 3

# Q learning parameters
learning_rate = 0.00025
gamma = 0.95
episodes = 500
max_steps = 5000

# Q target parameter
max_tau = 1000

# experience replay parameters 
pretrain_length = 10000
replay_buffer_size = 10000
batch_size = 64 

# exploration parameters for epsilon-greedy 
start_epsilon = 1.0
end_epsilon   = 0.01
decay_rate    = 0.00005

frame_repeat = 4

# recorded data 
all_losses = []
all_rewards = [] 

##################################################################################################


# frame stack class
# used to stack stack_size frames together to add a sense of motion for the network
# also performs preprocessing of the pushed frame 
# push has an optional parameter to reset the stack ie copy the pushed frame stack_size times 
# push returns the stacked frames
class FrameStack(object):
    def __init__(self, stack_size):
        self.stack_size = stack_size 
        self.stacked_frames = deque([np.zeros((1, 60, 80), dtype=np.int) for i in range(self.stack_size)], maxlen=self.stack_size) 
        

    def push(self, frame, reset=False, preprocess=True):
        if preprocess:
            frame = self._preprocess(frame)
        if reset:
            for i in range(self.stack_size):
                self.stacked_frames.append(frame) 
        else:
            self.stacked_frames.append(frame)
        return np.stack(self.stacked_frames)

    def get(self):
        return np.stack(self.stacked_frames)

    # preprocesses the frame
    def _preprocess(self, frame, height=60, width=80):
        # center crop the frame 
        #cropped_frame = frame[30:-10, :]

        # Normalise Pixel Values
        normalised_frame = frame / 255.0
        
        # Resize the frame 
        preprocessed_frame = transform.resize(normalised_frame, [height, width], anti_aliasing='true', mode='reflect')

        return preprocessed_frame


# memory buffer to store experience and retrieve random batches 
class ReplayBuffer(object):
    PER_e = 0.01    # avoid experiences to have 0 probability  
    PER_a = 0.6     # tradeoff between high priority and random 
    PER_b = 0.4     # importance-sampling
    PER_b_increment_per_sampling = 0.001
    absolute_error_upper = 1    # clipped absolute error 

    def __init__(self, capacity):
        # make the tree 
        self.tree = SumTree(capacity)
    
    # add new experience to tree
    # start at max priority, improved when used for training 
    def push(self, state, action, reward, next_state, terminal):
        # find max priority 
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        # if max priority = 0, we use a minimum priority 
        if max_priority == 0:
            max_priority = self.absolute_error_upper
        # setup our experience 
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        experience = state, action, reward, next_state, terminal
        # add the experience 
        self.tree.add(max_priority, experience) 
    

    def sample(self, batch_size):
        # create an array that will contain the minibatch 
        memory_b = []

        b_idx       = np.empty((batch_size,), dtype=np.int32)
        b_ISWeights = np.empty((batch_size, 1), dtype=np.float32)

        # calculate priority segment 
        priority_segment = self.tree.total_priority / batch_size

        # inclease the PER_b each time we sample a new minibatch
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])

        # calculate the max weight 
        p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority
        max_weight = (p_min * batch_size) ** (-self.PER_b)

        for i in range(batch_size):
            # uniformly sample a value from each range 
            a = priority_segment * i  
            b = priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            # retrieve experience that correspond to each value 
            index, priority, data = self.tree.get_leaf(value)

            sampling_probabilities = priority / self.tree.total_priority

            b_ISWeights[i, 0] = np.power(batch_size * sampling_probabilities, -self.PER_b) / max_weight

            b_idx[i] = index 

            experience = [data]

            memory_b.append(experience)
        
        return b_idx, memory_b, b_ISWeights

#        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
#        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    # update tree priorities 
    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.PER_e 
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


# dueling double deep q-network 
class DDDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DDDQN, self).__init__() 

        self.input_shape = input_shape 
        self.num_actions = num_actions 

        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU()
        )

        self.advantage = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
    

    def forward(self, x):
        x = self.features(x)        # feed forward through feature layers 
        x = x.view(x.size(0), -1)   # flatten 
        advantage = self.advantage(x)   # feed forward through advantage stream 
        value = self.value(x)           # feed forward through value stream 
        return value + advantage - advantage.mean()
    

    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
    

    def act(self, state, epsilon):
        if random.random() > epsilon:
            # convert the state to a torch tensor
            state = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0)) 
            # feed the state through the network 
            q_value = self.forward(state) 
            # argmax is the action 
            action = q_value.max(1)[1].data[0]
        else:  
            # otherwise select a random action 
            action = random.randint(0, self.num_actions - 1)
        return action 


class SumTree(object):

    data_pointer = 0 

    # initialise the tree with all nodes and data values to equal 0
    def __init__(self, capacity):
        # the number of leaf nodes that contain experiences 
        self.capacity = capacity 
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)

    # add priority score to the leaf and add experience to the data 
    def add(self, priority, data):
        tree_index = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data 
        self.update(tree_index, priority)
        self.data_pointer += 1 
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0 

    # update leaf priority and propagate the change through the tree 
    def update(self, tree_index, priority):
        # change is the new priority subtract the old priority 
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority    
        # propogate
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change 
    
    # get leaf returns the leaf index, priority value and data
    def get_leaf(self, v):
        parent_index = 0 
        while True: 
            left_child_index = 2 * parent_index + 1 
            right_child_index = left_child_index + 1
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break 
            # downward search
            else:
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index
            
        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0] 


# create and initialise the Doom environment
def initialise_environment(config_file_path):
    print("Initialising environment...")
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(ScreenFormat.GRAY8)
    game.set_screen_resolution(ScreenResolution.RES_160X120)
    game.init()
    print("Doom Initialised.")
    return game 


def compute_td_loss(batch_size):
    # retrieve a sample of minibatches to train with 
    tree_idx, batch, weights = replay_buffer.sample(batch_size) 

    state_mb = np.array([each[0][0] for each in batch], ndmin=3)
    action_mb = np.array([each[0][1] for each in batch])
    reward_mb = np.array([each[0][2] for each in batch]) 
    next_state_mb = np.array([each[0][3] for each in batch], ndmin=3)
    terminal_mb = np.array([each[0][4] for each in batch]).astype(int)

    # convert minibatches into torch variables 
    state_mb      = Variable(torch.FloatTensor(np.float32(np.concatenate(state_mb))))
    next_state_mb = Variable(torch.FloatTensor(np.float32(np.concatenate(next_state_mb))))
    action_mb     = Variable(torch.LongTensor(action_mb))
    reward_mb     = Variable(torch.FloatTensor(reward_mb))
    terminal_mb   = Variable(torch.FloatTensor(terminal_mb))
    weights = Variable(torch.FloatTensor(weights))

    q_values = current_model(state_mb)
    next_q_values = target_model(next_state_mb)
    
    q_value = q_values.gather(1, action_mb.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_values.max(1)[0]
    expected_q_value = reward_mb + gamma * next_q_value * (1 - terminal_mb)
    
    absolute_errors = torch.abs(q_value - expected_q_value.detach())
    loss = (q_value - expected_q_value.detach()).pow(2) * weights 
    loss = loss.mean() 
    
    # update priorities 
    replay_buffer.batch_update(tree_idx, absolute_errors.data.cpu().numpy())

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    return loss 


def update_target_network(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())


if __name__ == '__main__':
    # create the doom environment 
    game = initialise_environment(config_file_path)

    # actions 
    actions = np.identity(num_actions, dtype=int).tolist()
    # initialise the frame stack 
    frame_stack = FrameStack(4)

    # push the first state into the frame stack
    state = frame_stack.push(game.get_state().screen_buffer, reset=True)
    
    # Initialise the two neural networks 
    current_model = DDDQN(state.shape, len(actions))
    target_model  = DDDQN(state.shape, len(actions))

    if (load_model):
        print("Loading model from: ", model_savefile)
        current_model = torch.load(model_savefile)
    
    if (use_cuda):
        current_model = current_model.cuda() 
        target_model  = target_model.cuda()
    
    update_target_network(current_model, target_model)

    optimiser = optim.Adam(current_model.parameters(), learning_rate)

    replay_buffer = ReplayBuffer(replay_buffer_size)
    
    time_start = time.time()
    
    if not skip_learning:
        
        game.new_episode() 
        
        
        # first we need to fill out memory with pretrain_length experiences 
        print("Filling memory with %d pretrain experiences\n" % pretrain_length)
        for i in trange(pretrain_length):

            # if it's the first step, we need to reset the frame_stack 
            if i == 0:
                state = frame_stack.push(game.get_state().screen_buffer, reset=True)
            
            if load_model:
                # if we loaded the model, start with predicted actions 
                action = current_model.act(state, 0.0)
            else:
                # select a random action 
                action = random.randint(0, len(actions)-1)

            # perform the action and collect reward 
            reward = game.make_action(actions[action], frame_repeat)

            # check if the episode is over 
            terminal = game.is_episode_finished() 

            # if the episode is over 
            if terminal:
                # add empty next state to frame stack 
                next_state = frame_stack.push(np.zeros((60, 80), dtype=np.int), reset=False, preprocess=False)

                # add experience to memory 
                replay_buffer.push(state, action, reward, next_state, terminal)

                # begin a new episode 
                game.new_episode()

                # reset the frame stack 
                state = frame_stack.push(game.get_state().screen_buffer, reset=True) 
            
            else:
                # get the next state and add to frame stack 
                next_state = frame_stack.push(game.get_state().screen_buffer)
                # add the experience to memory 
                replay_buffer.push(state, action, reward, next_state, terminal)
                # the state is now the next state 
                state = next_state 
            
        print("Beginning training\n")
        
        decay_step = 0 
        tau = 0
        training_rewards = []
        for episode in range(episodes):
            step = 0      

            # initialise rewards for the episode 
            episode_rewards = []
            
            # make a new episode 
            game.new_episode()

            # start a new frame stack with the first state  
            state = frame_stack.push(game.get_state().screen_buffer, reset=True)

            while step < max_steps:
                step += 1

                # increase the tau step 
                tau += 1

                # increase decay step 
                decay_step += 1
                # calculate epsilon for epsilon-greedy strategy 
                epsilon = end_epsilon + (start_epsilon - end_epsilon) * np.exp(-decay_rate * decay_step)
                # predict the action to take 
                action = current_model.act(state, epsilon)
                # perform the action 
                reward = game.make_action(actions[action], frame_repeat)
                # check if the episode has finished 
                terminal = game.is_episode_finished()
                # add the reward to the total rewards 
                episode_rewards.append(reward)
                
                # if episode has finished 
                if terminal:
                    # episode ends, so no new state 
                    next_state = frame_stack.push(np.zeros((60, 80), dtype=np.int), reset=False, preprocess=False)
                    # set step = max steps to end the episode 
                    step = max_steps
                    # get the total reward for the episode 
                    total_reward = np.sum(episode_rewards)

                    # add experience to memory 
                    replay_buffer.push(state, action, reward, next_state, terminal)
                    
                    print("Episode: {}".format(episode),
                          "Total Reward: {}".format(total_reward),
                          "Training Loss: {:.4f}".format(loss),
                          "Epsilon P: {:.4f}".format(epsilon))
                else:
                    # get the next state 
                    next_state = frame_stack.push(game.get_state().screen_buffer, reset=False)
                    # add experience to memory 
                    replay_buffer.push(state, action, reward, next_state, terminal)
                    # state + 1 is now the current state 
                    state = next_state 
                
                # perform learning 
                loss = compute_td_loss(batch_size)

                if tau > max_tau:
                    update_target_network(current_model, target_model)
                    tau = 0
                    print("Model updated")

            training_rewards.append(np.sum(episode_rewards))

            # save the model every 5 episodes 
            if episode % 5 == 0:
                #print("Saving the network weights to: ", model_savefile)
                torch.save(current_model, model_savefile)
                
        np.savetxt(training_rewards_savefile, np.asarray(training_rewards), fmt="%s", delimiter=",", newline=",")

        print("Total elapsed time: %.2f minutes" % ((time.time() - time_start) / 60.0))

    game.close() 

    print("==================================================")
    print("Training finished.")

    game.init() 
    
    total_test_rewards = []
    for i in range(testing_episodes):
        game.new_episode("episode" + str(i) + "_rec.lmp")

        state = frame_stack.push(game.get_state().screen_buffer, reset=True)
        while not game.is_episode_finished():
            best_action_index = current_model.act(state, 0.0)
            # instead of make_action to make the animation smooth
            game.set_action(actions[best_action_index])
            for _ in range(frame_repeat):
                game.advance_action()
            done = game.is_episode_finished()
            if not done:
                state = frame_stack.push(game.get_state().screen_buffer, reset=False)            
        # sleep between episodes
        time.sleep(1.0)
        score = game.get_total_reward()
        total_test_rewards.append(score)
        print("Total score: ", score)
    print("Average total test score: ", np.mean(total_test_rewards))
    game.close() 

    if (render_testing_episodes):
        # New render settings for replay
        game.set_screen_resolution(ScreenResolution.RES_800X600)
        game.set_window_visible(True)
        # Replay can be played in any mode.
        game.set_mode(Mode.ASYNC_PLAYER)

        game.init()

        print("\nREPLAY OF EPISODE")
        print("************************\n")

        for i in range(testing_episodes):

            # Replays episodes stored in given file. Sending game command will interrupt playback.
            game.replay_episode("episode" + str(i) + "_rec.lmp")

            while not game.is_episode_finished():
                s = game.get_state()
                # Use advance_action instead of make_action.
                game.advance_action()

            print("Episode finished.")
            print("total reward:", game.get_total_reward())
            print("************************")

        game.close()

    # Delete recordings (*.lmp files).
    for i in range(testing_episodes):
        os.remove("episode" + str(i) + "_rec.lmp")