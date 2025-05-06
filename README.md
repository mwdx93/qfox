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

[2] Jumaah, Mahmood A.; Ali, Yossra H.; Rashid, Tarik A.; and Vimal, S. (2024) "FOXANN: A Method for Boosting Neural Network Performance," Journal of Soft Computing and Computer Applications: Vol. 1: Iss. 1, Article 1001. DOI: https://doi.org/10.70403/3008-1084.1001

[2]  Jumaah, M.A., Ali, Y.H. and Rashid, T.A., 2024. QF-tuner: Breaking Tradition in Reinforcement Learning. arXiv preprint arXiv:2402.16562.

```
@article{Jumaah2025_QFOX,
  title = {Efficient Q-learning hyperparameter tuning using FOX optimization algorithm},
  volume = {25},
  ISSN = {2590-1230},
  url = {http://dx.doi.org/10.1016/j.rineng.2025.104341},
  DOI = {10.1016/j.rineng.2025.104341},
  journal = {Results in Engineering},
  publisher = {Elsevier BV},
  author = {Jumaah,  Mahmood A. and Ali,  Yossra H. and Rashid,  Tarik A.},
  year = {2025},
  month = mar,
  pages = {104341}
}

@article{Jumaah2024_FOXANN,
  title = {FOXANN: A Method for Boosting Neural Network Performance},
  volume = {1},
  ISSN = {3008-1084},
  url = {http://dx.doi.org/10.70403/3008-1084.1001},
  DOI = {10.70403/3008-1084.1001},
  number = {1},
  journal = {Journal of Soft Computing and Computer Applications},
  publisher = {University of Technology - Iraq / Digital Commons},
  author = {Jumaah,  Mahmood A. and Ali,  Yossra H. and Rashid,  Tarik A. and Vimal,  S.},
  year = {2024},
  month = jun 
}

@misc{Jumaah2024_QF,
  doi = {10.48550/ARXIV.2402.16562},
  url = {https://arxiv.org/abs/2402.16562},
  author = {Jumaah,  Mahmood A. and Ali,  Yossra H. and Rashid,  Tarik A.},
  keywords = {Machine Learning (cs.LG),  Artificial Intelligence (cs.AI),  Neural and Evolutionary Computing (cs.NE),  FOS: Computer and information sciences,  FOS: Computer and information sciences},
  title = {QF-tuner: Breaking Tradition in Reinforcement Learning},
  publisher = {arXiv},
  year = {2024},
  copyright = {Creative Commons Attribution 4.0 International}
}

```


