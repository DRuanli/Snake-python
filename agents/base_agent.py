"""
Abstract base class for all agents.
"""
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """
    Abstract base class that all agent implementations must inherit from.
    Defines the common interface for agent behavior.
    """
    
    def __init__(self, action_space_size=4):
        """
        Initialize the agent.
        
        Args:
            action_space_size (int): Number of possible actions
        """
        self.action_space_size = action_space_size
    
    @abstractmethod
    def act(self, state, deterministic=False):
        """
        Take an action based on the current state.
        
        Args:
            state: Current observation of the environment
            deterministic (bool): Whether to act deterministically
            
        Returns:
            int: Action to take
        """
        pass
    
    def reset(self):
        """
        Reset the agent's state at the beginning of an episode.
        Override this method if the agent needs to reset internal state.
        """
        pass
    
    def train(self, state, action, next_state, reward, done):
        """
        Train the agent with a single experience tuple.
        Override this method for agents that can learn.
        
        Args:
            state: Current state
            action (int): Action taken
            next_state: Next state
            reward (float): Reward received
            done (bool): Whether the episode is done
        """
        pass
    
    def save(self, path):
        """
        Save the agent's model or policy.
        Override this method for agents that have savable state.
        
        Args:
            path (str): Path to save to
        """
        pass
    
    def load(self, path):
        """
        Load the agent's model or policy.
        Override this method for agents that have loadable state.
        
        Args:
            path (str): Path to load from
        """
        pass