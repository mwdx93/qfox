import numpy as np
import gymnasium as gym
import time
from .QLAgent import QLAgent
from mealpy import FloatVar
from mealpy import optimizer
class QLCartpoleAgent(QLAgent):
    def __init__(self, env:gym, learning_rate:float, discount_factor:float, epsilon:float, episodes:int,optimizer:optimizer):
        super().__init__(env, learning_rate, discount_factor, epsilon,episodes, optimizer)
        
        self.q_table = np.zeros(self.no_buckets + (self.env.action_space.n,))
        
               
    
    
    def optimize(self, s, a, r, s_, episode):

        # Objective function
        def obj(params):
            lr, gamma = params
            q = np.copy(self.q_table)

            best_future_q = np.max(q[s_])
            q[s][a] +=  lr * (r + gamma * best_future_q - q[s][a])
            self.optimized_q = q[s][a]

            fitness = 1/(np.mean(q[:])+0.1)
            return fitness
        
        problem = {
            "bounds": FloatVar(lb=(0.01, 0.1), ub=(0.5, 0.99)),
            "obj_func": obj,
            "minmax": "min",
            "log_to": None,
            "ndim": 2
        }

        opt = self.optimizer(epoch = 40, pop_size = 14)
        opt.solve(problem)
        best_params = opt.g_best.solution
        fitness = opt.g_best.target.fitness
        
        # Update learning rate and discount factor
        print(f'{self.optimizer.__name__} ({episode})--> HP: {best_params} | R: {round(fitness,4)}')


    def train(self):
        if self.optimizer.__name__ == 'Fixed':
                self.learning_rate = 0.1
                self.discount_factor = 0.99
        elif self.optimizer.__name__ == 'Random':
                self.learning_rate = np.random.uniform(0.1, 0.01)
                self.discount_factor = np.random.uniform(0.5, 0.99)

        start_time = time.time()

        # Training main loop
        for episode in range(self.episodes):

            # initial state
            observation = self.env.reset(seed = 123)[0]
            s = self.bucketize_state_value(observation)

            total_reward = 0
            done = False
            tran = 0
            step_start = time.time()
            step = 0
            while not done:
                # Choose action
                a = self.select_action_cartpole(s, self.q_table)

                # Observe
                observation, r, _, done, _ = self.env.step(a)
                s_ = self.bucketize_state_value(observation)

                # Update q table
                if self.optimizer.__name__ == 'Fixed' or self.optimizer.__name__ == 'Random':
                    self.q_table[s][a] =  self.update_q_table_cartpole(s,a,r,s_,self.q_table)
                else:
                    self.optimize(s, a, r, s_, episode)
                    standared_q = self.update_q_table_cartpole(s,a,r,s_,self.q_table)
                    
                    if self.optimized_q > standared_q:
                        self.q_table[s][a] = self.optimized_q
                    else:
                        self.q_table[s][a] = standared_q

                # Update: State, total_reward, time
                s = s_
                total_reward += r
                step+=1
                # Store the experience to calculate MSE
                self.experiences_list.append((s, a, r, s_))
                


            # Store the total_rewards, totalsteps and mse for this episode
            self.cumulativeRewards.append(total_reward)
            self.averageReward.append(total_reward/(time.time()-step_start))
            self.cumulativeSteps.append(time.time()-step_start)
            
            self.mse.append(self.calculate_mse(self.q_table, self.experiences_list))

            #update epsilon
            if self.epsilon>self.min_epsilon:
                self.epsilon *= self.epsilon_decay
        

        # store the entire traning time
        self.processing_time = time.time()-start_time