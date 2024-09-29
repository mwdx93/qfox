import numpy as np
import math


class QLAgent:
    def __init__(self, env, learning_rate, discount_factor, epsilon, episodes, optimizer):
        self.env = env

        self.learning_rate = learning_rate          #Alpha
        self.discount_factor = discount_factor      #Gamma
        self.epsilon = epsilon
        self.epsilon = epsilon
        self.episodes = episodes
        self.optimizer = optimizer

        self.processing_time = 0
        self.min_epsilon = 0.001
        self.epsilon_decay = (self.min_epsilon/self.epsilon)**(1/self.episodes)
        
        self.no_buckets = (1, 1, 6, 3) # for cartpole discretizaion

        self.cumulativeRewards = []
        self.averageReward = []
        self.cumulativeSteps = []
        self.mse = []
        self.experiences_list = []

        self.optimized_q = 0
        
    def select_action(self, s, q_table):
        if np.random.rand() < self.epsilon:
            # explore
            action = self.env.action_space.sample()
        else: 
            action = np.argmax(q_table[s,:])
        return action
    
    def select_action_cartpole(self, s, q_table):
        if np.random.rand() < self.epsilon:
            # explore
            action = self.env.action_space.sample()
        else: 
            action = np.argmax(q_table[s])
        return action
    
    def bucketize_state_value(self,state_value):
        state_value_bounds = list(zip(self.env.observation_space.low, self.env.observation_space.high))
        state_value_bounds[1] = (-0.5, 0.5)
        state_value_bounds[3] = (-math.radians(50), math.radians(50))

        bucket_indices = []
        for i in range(len(state_value)):
            if state_value[i] <= state_value_bounds[i][0]:
                # violates lower bound
                bucket_index = 0
            elif state_value[i] >= state_value_bounds[i][1]:
                # violates upper bound
                # put in the last bucket
                bucket_index = self.no_buckets[i] - 1
            else:
                bound_width = state_value_bounds[i][1] - \
                state_value_bounds[i][0]
                offset = (self.no_buckets[i]-1) * \
                state_value_bounds[i][0] / bound_width
                scaling = (self.no_buckets[i]-1) / bound_width
                bucket_index = int(round(scaling*state_value[i] -offset))
            bucket_indices.append(bucket_index)
        return tuple(bucket_indices)
    
    def calculate_mse(self, Q, experiences):
        mse = 0
        for state, action, reward, next_state in experiences:
            target = reward + self.discount_factor * np.max(Q[next_state])
            error = Q[state][action] - target
            mse += error**2
        mse /= len(experiences)
        return mse
    

    def update_q_table(self, s, a, r, s_, q):
        best_future_q = np.max(q[s_,:])
        return q[s, a] + self.learning_rate * (r + self.discount_factor * best_future_q - q[s, a])
    
    def update_q_table_cartpole(self, s, a, r, s_, q):
        best_future_q = np.max(q[s_][:])
        return q[s][a] + self.learning_rate * (r + self.discount_factor * best_future_q - q[s][a])
    
    
    def train(self):
        raise NotImplementedError("Train method should be implemented by the subclass.")
        