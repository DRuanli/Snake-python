"""
Snake class and movement logic.
"""
import numpy as np
from enum import Enum

class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

class Snake:
    def __init__(self, head_position, initial_length=3, initial_direction=Direction.RIGHT):
        """
        Initialize snake with head position, length, and direction.
        
        Args:
            head_position (tuple): (x, y) coordinates of head
            initial_length (int): Starting length of snake
            initial_direction (Direction): Starting direction
        """
        self.direction = initial_direction
        self.head_position = head_position
        
        # Initialize body segments based on initial direction
        self.body = [head_position]
        x, y = head_position
        
        # Create body segments extending opposite to the direction
        for i in range(1, initial_length):
            if initial_direction == Direction.RIGHT:
                self.body.append((x - i, y))
            elif initial_direction == Direction.LEFT:
                self.body.append((x + i, y))
            elif initial_direction == Direction.UP:
                self.body.append((x, y + i))
            elif initial_direction == Direction.DOWN:
                self.body.append((x, y - i))
    
    def get_head_position(self):
        """Return the current position of the snake's head."""
        return self.body[0]
    
    def get_body_positions(self):
        """Return positions of all body segments (excluding head)."""
        return self.body[1:]
    
    def get_all_positions(self):
        """Return positions of all segments (including head)."""
        return self.body
    
    def change_direction(self, new_direction):
        """
        Change snake's direction if valid (can't go directly backwards).
        
        Args:
            new_direction (Direction): New direction to move
            
        Returns:
            bool: Whether the direction change was successful
        """
        # Prevent 180-degree turns
        if (self.direction == Direction.UP and new_direction == Direction.DOWN) or \
           (self.direction == Direction.DOWN and new_direction == Direction.UP) or \
           (self.direction == Direction.LEFT and new_direction == Direction.RIGHT) or \
           (self.direction == Direction.RIGHT and new_direction == Direction.LEFT):
            return False
        
        self.direction = new_direction
        return True
    
    def move(self, grow=False):
        """
        Move the snake one step in the current direction.
        
        Args:
            grow (bool): Whether to grow the snake (after eating food)
            
        Returns:
            tuple: New head position after movement
        """
        x, y = self.get_head_position()
        
        # Calculate new head position based on direction
        if self.direction == Direction.UP:
            new_head = (x, y - 1)
        elif self.direction == Direction.DOWN:
            new_head = (x, y + 1)
        elif self.direction == Direction.LEFT:
            new_head = (x - 1, y)
        elif self.direction == Direction.RIGHT:
            new_head = (x + 1, y)
        
        # Insert new head at the beginning
        self.body.insert(0, new_head)
        
        # Remove the last segment if not growing
        if not grow:
            self.body.pop()
            
        return new_head
    
    def check_self_collision(self):
        """
        Check if snake's head collides with its body.
        
        Returns:
            bool: True if collision detected, False otherwise
        """
        return self.get_head_position() in self.get_body_positions()