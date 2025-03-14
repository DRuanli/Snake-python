"""
RL agent for Snake game - wraps DQN implementation.
"""
from learning.dqn import DQNAgent
from .base_agent import BaseAgent
import config

class RLAgent(BaseAgent):
    """
    RL agent that uses DQN for learning and decision making.
    Wraps the DQN implementation to provide the BaseAgent interface.
    """
    
    def __init__(self, use_dueling=True, use_priority=True, load_path=None):
        """
        Initialize the RL agent.
        
        Args:
            use_dueling (bool): Whether to use dueling network architecture
            use_priority (bool): Whether to use prioritized experience replay
            load_path (str): Path to load model from (None for new model)
        """
        super().__init__(config.OUTPUT_SIZE)
        
        # Initialize DQN agent
        self.dqn = DQNAgent(
            state_size=config.INPUT_SIZE,
            action_size=config.OUTPUT_SIZE,
            use_dueling=use_dueling,
            use_priority=use_priority
        )
        
        # Load model if provided
        if load_path:
            self.dqn.load(load_path)
    
    def act(self, state, deterministic=False):
        """
        Take an action based on the current state.
        
        Args:
            state: Current observation of the environment
            deterministic (bool): Whether to act deterministically
            
        Returns:
            int: Action to take
        """
        return self.dqn.act(state, deterministic)
    
    def train(self, state, action, next_state, reward, done):
        """
        Train the agent with a single experience tuple.
        
        Args:
            state: Current state
            action (int): Action taken
            next_state: Next state
            reward (float): Reward received
            done (bool): Whether the episode is done
        """
        self.dqn.train(state, action, next_state, reward, done)
    
    def save(self, path):
        """
        Save the agent's model.
        
        Args:
            path (str): Path to save to
        """
        self.dqn.save(path)
    
    def load(self, path):
        """
        Load the agent's model.
        
        Args:
            path (str): Path to load from
        """
        self.dqn.load(path)