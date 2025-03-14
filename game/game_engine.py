"""
Core game mechanics for Snake RL.
"""
import random
import numpy as np
from .snake import Snake, Direction
import config

class GameState:
    """Represents the current state of the game."""
    def __init__(self, grid_size=config.GRID_SIZE):
        self.grid_size = grid_size
        self.score = 0
        self.steps = 0
        self.game_over = False
        
        # Initialize grid
        self.grid = np.zeros((grid_size, grid_size), dtype=int)
    
    def update(self, snake, food_position, obstacles):
        """
        Update grid representation with current game objects.
        
        Args:
            snake (Snake): The snake object
            food_position (tuple): (x, y) coordinates of food
            obstacles (list): List of obstacle positions
        """
        # Reset grid
        self.grid.fill(0)
        
        # Add obstacles
        for x, y in obstacles:
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                self.grid[y, x] = 3
        
        # Add snake body
        for x, y in snake.get_body_positions():
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                self.grid[y, x] = 2
        
        # Add snake head
        x, y = snake.get_head_position()
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            self.grid[y, x] = 1
        
        # Add food
        x, y = food_position
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            self.grid[y, x] = 4

class GameEngine:
    """Core game mechanics and logic."""
    def __init__(self, grid_size=config.GRID_SIZE, obstacles=None, max_steps=config.MAX_STEPS):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.obstacles = obstacles if obstacles is not None else []
        self.reset()
    
    def reset(self):
        """Reset the game to initial state."""
        # Initialize at center
        center = self.grid_size // 2
        
        # Create snake at center facing right
        self.snake = Snake((center, center), initial_direction=Direction.RIGHT)
        
        # Place food in random position
        self.food_position = self._place_food()
        
        # Initialize game state
        self.state = GameState(self.grid_size)
        self.state.update(self.snake, self.food_position, self.obstacles)
        
        # Reset metrics
        self.score = 0
        self.steps = 0
        self.game_over = False
        
        return self._get_observation()
    
    def step(self, action):
        """
        Take a step in the game based on the action.
        
        Args:
            action (int): 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
            
        Returns:
            tuple: (observation, reward, done, info)
        """
        # Convert action to Direction
        direction = Direction(action)
        
        # Try to change direction
        self.snake.change_direction(direction)
        
        # Check if food will be eaten
        prev_distance = self._get_food_distance()
        
        # Move snake
        new_head = self.snake.move(grow=False)
        self.steps += 1
        
        # Default reward
        reward = -0.01  # Small negative reward for each step
        done = False
        ate_food = False
        
        # Check for food consumption
        if new_head == self.food_position:
            # Grow snake
            self.snake.body.append(self.snake.body[-1])
            
            # Place new food
            self.food_position = self._place_food()
            
            # Update score and give reward
            self.score += 1
            reward = 10.0  # Big reward for eating food
            ate_food = True
        
        # Check for collisions and game over conditions
        if self._check_collision() or self.steps >= self.max_steps:
            reward = -10.0  # Penalty for dying
            done = True
            self.game_over = True
        
        # Calculate reward based on distance to food if didn't eat
        if not ate_food and not done:
            current_distance = self._get_food_distance()
            # Reward for getting closer to food, penalize for moving away
            reward += 0.1 if current_distance < prev_distance else -0.1
        
        # Update game state
        self.state.update(self.snake, self.food_position, self.obstacles)
        self.state.score = self.score
        self.state.steps = self.steps
        self.state.game_over = self.game_over
        
        # Additional info
        info = {
            'score': self.score,
            'steps': self.steps,
            'snake_length': len(self.snake.body)
        }
        
        return self._get_observation(), reward, done, info
    
    def _check_collision(self):
        """
        Check if snake collided with walls, obstacles, or itself.
        
        Returns:
            bool: True if collision detected, False otherwise
        """
        head_x, head_y = self.snake.get_head_position()
        
        # Check wall collision
        if (head_x < 0 or head_x >= self.grid_size or 
            head_y < 0 or head_y >= self.grid_size):
            return True
        
        # Check obstacle collision
        if (head_x, head_y) in self.obstacles:
            return True
        
        # Check self collision
        return self.snake.check_self_collision()
    
    def _place_food(self):
        """
        Place food in a random empty cell.
        
        Returns:
            tuple: (x, y) coordinates of food
        """
        while True:
            # Generate random position
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)
            
            # Check if position is valid (not occupied by snake or obstacles)
            if ((x, y) not in self.snake.get_all_positions() and 
                (x, y) not in self.obstacles):
                return (x, y)
    
    def _get_food_distance(self):
        """
        Calculate Manhattan distance from snake's head to food.
        
        Returns:
            float: Distance to food
        """
        head_x, head_y = self.snake.get_head_position()
        food_x, food_y = self.food_position
        return abs(head_x - food_x) + abs(head_y - food_y)
    
    def _get_observation(self):
        """
        Get the current observation of the game state for RL.
        
        Returns:
            np.array: Observation for the agent
        """
        head_x, head_y = self.snake.get_head_position()
        food_x, food_y = self.food_position
        
        # Direction one-hot encoding
        direction_one_hot = np.zeros(4)
        direction_one_hot[self.snake.direction.value] = 1
        
        # Danger detection (walls, obstacles, body)
        danger_straight = self._is_direction_dangerous(self.snake.direction)
        danger_right = self._is_direction_dangerous(Direction((self.snake.direction.value + 1) % 4))
        danger_left = self._is_direction_dangerous(Direction((self.snake.direction.value - 1) % 4))
        
        # Normalized positions and distances
        norm_head_x = head_x / self.grid_size
        norm_head_y = head_y / self.grid_size
        norm_food_x = food_x / self.grid_size
        norm_food_y = food_y / self.grid_size
        
        # Create observation array
        observation = np.array([
            danger_straight,
            danger_right,
            danger_left,
            direction_one_hot[0],  # up
            direction_one_hot[1],  # right
            direction_one_hot[2],  # down
            direction_one_hot[3],  # left
            norm_head_x,
            norm_head_y,
            norm_food_x,
            norm_food_y
        ], dtype=np.float32)
        
        return observation
    
    def _is_direction_dangerous(self, direction):
        """
        Check if moving in a direction would result in collision.
        
        Args:
            direction (Direction): Direction to check
            
        Returns:
            bool: True if collision would occur, False otherwise
        """
        head_x, head_y = self.snake.get_head_position()
        
        # Calculate position in the given direction
        if direction == Direction.UP:
            check_x, check_y = head_x, head_y - 1
        elif direction == Direction.RIGHT:
            check_x, check_y = head_x + 1, head_y
        elif direction == Direction.DOWN:
            check_x, check_y = head_x, head_y + 1
        elif direction == Direction.LEFT:
            check_x, check_y = head_x - 1, head_y
        
        # Check if position is dangerous
        return (
            check_x < 0 or check_x >= self.grid_size or  # Wall collision
            check_y < 0 or check_y >= self.grid_size or  # Wall collision
            (check_x, check_y) in self.obstacles or      # Obstacle collision
            (check_x, check_y) in self.snake.get_body_positions()  # Self collision
        )