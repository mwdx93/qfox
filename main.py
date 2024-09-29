import gymnasium as gym  # Import the Gymnasium library for reinforcement learning environments
from rl.qlearning_cartpole import QLCartpoleAgent  # Import the custom QLCartpoleAgent class
from mealpy import FOX  # Import the FOX optimizer from the mealpy library

# Global Parameters
window_size = 100  # Number of episodes to include in the moving average for performance evaluation
episodes = 500 + window_size  # Total number of episodes to train the agent
runs = 1  # Number of independent runs for evaluation (not used in this snippet)
env = gym.make("CartPole-v1")  # Create the CartPole environment

# Initialize the Q-learning agent with specified parameters
qlAgent = QLCartpoleAgent(
    env=env,  # Environment for the agent to interact with
    learning_rate=0.1,  # Learning rate for the Q-learning algorithm
    discount_factor=0.99,  # Discount factor for future rewards
    epsilon=1,  # Exploration rate (1 = explore, 0 = exploit)
    episodes=episodes,  # Total episodes for training
    optimizer=FOX.OriginalFOX  # Optimizer to use for adaptive hyperparameter tuning
)

# Start the training process for the Q-learning agent
qlAgent.train()
