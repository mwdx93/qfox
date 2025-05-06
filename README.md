# QFOX

QFOX is a hybrid reinforcement learning method that integrates the FOX optimization algorithm with Q-learning to enhance learning performance in the CartPole environment. This repository contains the implementation of the QFOX method, enabling users to experiment with adaptive hyperparameter tuning in reinforcement learning tasks.

## Features

- **Adaptive Hyperparameter Tuning**: QFOX dynamically adjusts learning rate and discount factor using the FOX algorithm.
- **Environment**: Implementation for the CartPole-v1 environment from OpenAI's Gymnasium.
- **QLearning Agent**: A customized Q-learning agent designed to work with the FOX optimizer.

## Installation

To run the QFOX code, ensure you have the following libraries installed:
``` bash
pip install gymnasium
pip install mealpy
```
## Usage
1. Clone the repository:
```bash
git clone https://github.com/yourusername/QFOX.git
cd QFOX
```

2. Run the main code:
```bash
python main.py
```

## Example Usage
```bash
import gymnasium as gym
from rl.qlearning_cartpole import QLCartpoleAgent
from mealpy import FOX 

window_size = 100
episodes = 500 + window_size

env = gym.make("CartPole-v1")

qlAgent = QLCartpoleAgent(env=env, 
                           learning_rate=0.1, 
                           discount_factor=0.99, 
                           epsilon=1,
                           episodes=episodes,
                           optimizer=FOX.OriginalFOX)
qlAgent.train()
```

## Cite us:
[1] Jumaah, Mahmood A., Yossra H. Ali, and Tarik A. Rashid. "Efficient Q-learning Hyperparameter Tuning Using FOX Optimization Algorithm." Results in Engineering (2025): 104341.

[2] Jumaah, M.A., Ali, Y.H. and Rashid, T.A., 2024. QF-tuner: Breaking Tradition in Reinforcement Learning. arXiv preprint arXiv:2402.16562.



