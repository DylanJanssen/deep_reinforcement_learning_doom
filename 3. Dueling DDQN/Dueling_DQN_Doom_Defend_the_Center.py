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
config_file_path = "../scenarios/defend_the_center.cfg" 
model_savefile = "./model-doom-defend-the-center.pth"
training_rewards_savefile = "./training_rewards-defend_the_center.csv"
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
        self.stacked_frames = deque([np.zeros((1,60,80), dtype=np.int) for i in range(self.stack_size)], maxlen=self.stack_size) 
        

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

        # Normalise Pixel Values
        normalised_frame = frame / 255.0
        
        # Resize the frame 
        preprocessed_frame = transform.resize(normalised_frame, [height, width], anti_aliasing='true', mode='reflect')

        return preprocessed_frame


# memory buffer to store experience and retrieve random batches 
class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, terminal):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self.buffer.append((state, action, reward, next_state, terminal)) 
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done
    
    def __len__(self):
        return len(self.buffer) 
    

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
    state_mb, action_mb, reward_mb, next_state_mb, terminal_mb = replay_buffer.sample(batch_size)

    # convert minibatches into torch variables 
    state_mb      = Variable(torch.FloatTensor(np.float32(state_mb)))
    next_state_mb = Variable(torch.FloatTensor(np.float32(next_state_mb)))
    action_mb     = Variable(torch.LongTensor(action_mb))
    reward_mb     = Variable(torch.FloatTensor(reward_mb))
    terminal_mb   = Variable(torch.FloatTensor(terminal_mb))

    q_values = current_model(state_mb)
    next_q_values = target_model(next_state_mb)
    
    q_value = q_values.gather(1, action_mb.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_values.max(1)[0]
    expected_q_value = reward_mb + gamma * next_q_value * (1 - terminal_mb)
    
    loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()
    
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