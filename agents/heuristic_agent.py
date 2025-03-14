"""
Heuristic agent for Snake RL - uses simple rules to make decisions.
"""
import numpy as np
from .base_agent import BaseAgent

class HeuristicAgent(BaseAgent):
    """
    Agent that uses predefined rules to navigate the game.
    More sophisticated than RandomAgent but simpler than RL.
    """
    
    def __init__(self, action_space_size=4):
        """
        Initialize the heuristic agent.
        
        Args:
            action_space_size (int): Number of possible actions
        """
        super().__init__(action_space_size)
    
    def act(self, state, deterministic=False):
        """
        Select action based on heuristic rules.
        
        Args:
            state: Current observation of the environment
            deterministic (bool): Whether to act deterministically
            
        Returns:
            int: Action to take
        """
        # Extract danger information
        danger_straight = state[0]
        danger_right = state[1]
        danger_left = state[2]
        
        # Get current direction one-hot encoded
        current_direction = np.argmax(state[3:7])
        
        # Get snake position and food position
        head_x, head_y = state[7], state[8]
        food_x, food_y = state[9], state[10]
        
        # Calculate direction to food (normalized)
        food_dir_x = food_x - head_x
        food_dir_y = food_y - head_y
        
        # Determine the best direction to move towards food
        preferred_action = self._get_preferred_direction(current_direction, food_dir_x, food_dir_y)
        
        # Check if preferred action would lead to danger
        if (preferred_action == current_direction and danger_straight) or \
           (preferred_action == (current_direction + 1) % 4 and danger_right) or \
           (preferred_action == (current_direction - 1) % 4 and danger_left):
            
            # Try alternative safe directions (prioritize turning over going straight)
            if not danger_right:
                return (current_direction + 1) % 4
            elif not danger_left:
                return (current_direction - 1) % 4
            elif not danger_straight:
                return current_direction
            else:
                # All directions are dangerous, pick the one that seems best
                # (This is a last resort, will likely lead to collision)
                return preferred_action
        else:
            # Preferred action is safe
            return preferred_action
    
    def _get_preferred_direction(self, current_direction, food_dir_x, food_dir_y):
        """
        Determine preferred direction to move towards food.
        
        Args:
            current_direction (int): Current snake direction
            food_dir_x (float): X-direction to food
            food_dir_y (float): Y-direction to food
            
        Returns:
            int: Preferred action
        """
        # Directions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
        
        # Prioritize the axis with the larger difference
        if abs(food_dir_x) > abs(food_dir_y):
            # Prioritize horizontal movement
            if food_dir_x > 0:  # Food is to the right
                preferred_action = 1  # RIGHT
            else:  # Food is to the left
                preferred_action = 3  # LEFT
        else:
            # Prioritize vertical movement
            if food_dir_y > 0:  # Food is below
                preferred_action = 2  # DOWN
            else:  # Food is above
                preferred_action = 0  # UP
        
        # Calculate opposite direction (can't go backwards)
        opposite_direction = (current_direction + 2) % 4
        
        # If preferred action is opposite direction, choose a perpendicular direction
        if preferred_action == opposite_direction:
            if current_direction in [0, 2]:  # UP or DOWN
                # Choose LEFT or RIGHT based on food position
                preferred_action = 1 if food_dir_x > 0 else 3
            else:  # LEFT or RIGHT
                # Choose UP or DOWN based on food position
                preferred_action = 0 if food_dir_y < 0 else 2
        
        return preferred_action