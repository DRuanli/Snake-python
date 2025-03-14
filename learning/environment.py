"""
RL environment wrapper for Snake game.
"""
import numpy as np
from game.game_engine import GameEngine
from game.visualizer import Visualizer

class SnakeEnvironment:
    """
    Reinforcement learning environment wrapper for the Snake game.
    Follows a standard RL environment interface similar to OpenAI Gym.
    """
    
    def __init__(self, grid_size, obstacles=None, render_mode=None):
        """
        Initialize the environment.
        
        Args:
            grid_size (int): Size of the game grid
            obstacles (list): List of obstacle positions
            render_mode (str): 'human' to render, None for no rendering
        """
        self.game_engine = GameEngine(grid_size, obstacles)
        self.render_mode = render_mode
        self.visualizer = Visualizer(grid_size) if render_mode == 'human' else None
        
        # Define action and observation spaces
        self.action_space_size = 4  # UP, RIGHT, DOWN, LEFT
        self.observation_space_size = 11  # Based on _get_observation in GameEngine
    
    def reset(self):
        """
        Reset the environment to initial state.
        
        Returns:
            np.array: Initial observation
        """
        observation = self.game_engine.reset()
        
        if self.render_mode == 'human':
            self._render_frame()
        
        return observation
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action (int): Action to take (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT)
            
        Returns:
            tuple: (observation, reward, done, info)
        """
        observation, reward, done, info = self.game_engine.step(action)
        
        if self.render_mode == 'human':
            self._render_frame()
        
        return observation, reward, done, info
    
    def _render_frame(self):
        """Render the current game state."""
        if self.visualizer:
            self.visualizer.render(
                self.game_engine.state,
                self.game_engine.snake,
                self.game_engine.food_position,
                self.game_engine.obstacles,
                {
                    'score': self.game_engine.score,
                    'steps': self.game_engine.steps
                }
            )
    
    def render(self):
        """Render the current game state (for compatibility with gym-like interface)."""
        self._render_frame()
    
    def close(self):
        """Close the environment."""
        if self.visualizer:
            self.visualizer.close()