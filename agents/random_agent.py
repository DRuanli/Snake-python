"""
Random agent for Snake RL - makes random valid moves.
"""
import random
import numpy as np
from .base_agent import BaseAgent

class RandomAgent(BaseAgent):
    """
    Agent that selects actions randomly from the valid action space.
    Used as a baseline for comparison.
    """
    
    def __init__(self, action_space_size=4):
        """
        Initialize the random agent.
        
        Args:
            action_space_size (int): Number of possible actions
        """
        super().__init__(action_space_size)
    
    def act(self, state, deterministic=False):
        """
        Take a random action.
        
        Args:
            state: Current observation of the environment (not used)
            deterministic (bool): Whether to act deterministically (not used)
            
        Returns:
            int: Random action
        """
        # Extract danger information from state to avoid immediate death if possible
        danger_straight = state[0]
        danger_right = state[1]
        danger_left = state[2]
        
        # Get current direction one-hot encoded from state
        current_direction = np.argmax(state[3:7])
        
        # Calculate opposite direction (can't go backwards)
        opposite_direction = (current_direction + 2) % 4
        
        # Create list of possible actions (excluding immediate death and opposite direction)
        possible_actions = []
        
        for action in range(self.action_space_size):
            # Skip opposite direction (can't go backwards)
            if action == opposite_direction:
                continue
                
            # Check if action would lead to immediate death
            if (action == current_direction and danger_straight) or \
               (action == (current_direction + 1) % 4 and danger_right) or \
               (action == (current_direction - 1) % 4 and danger_left):
                continue
                
            possible_actions.append(action)
        
        # If no safe actions, take any action except opposite direction
        if not possible_actions:
            possible_actions = [a for a in range(self.action_space_size) if a != opposite_direction]
        
        # Return random action from possible actions
        return random.choice(possible_actions)