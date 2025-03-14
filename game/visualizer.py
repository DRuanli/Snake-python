"""
Game visualization using Pygame.
"""
import pygame
import config
from .snake import Direction

class Visualizer:
    """Handles game visualization using Pygame."""
    
    def __init__(self, grid_size=config.GRID_SIZE, cell_size=config.CELL_SIZE):
        """
        Initialize the visualizer.
        
        Args:
            grid_size (int): Size of the game grid
            cell_size (int): Size of each cell in pixels
        """
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.window_width = grid_size * cell_size
        self.window_height = grid_size * cell_size
        
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Snake RL")
        
        # Initialize fonts
        self.font = pygame.font.SysFont('arial', 20)
        
        # Initialize clock
        self.clock = pygame.time.Clock()
    
    def render(self, game_state, snake, food_position, obstacles, info=None):
        """
        Render the current game state.
        
        Args:
            game_state (GameState): Current game state
            snake (Snake): Snake object
            food_position (tuple): (x, y) coordinates of food
            obstacles (list): List of obstacle positions
            info (dict): Additional information to display
        """
        # Clear screen
        self.screen.fill(config.BACKGROUND_COLOR)
        
        # Draw grid lines
        for x in range(0, self.window_width, self.cell_size):
            pygame.draw.line(self.screen, (50, 50, 50), (x, 0), (x, self.window_height))
        for y in range(0, self.window_height, self.cell_size):
            pygame.draw.line(self.screen, (50, 50, 50), (0, y), (self.window_width, y))
        
        # Draw obstacles
        for x, y in obstacles:
            pygame.draw.rect(self.screen, config.OBSTACLE_COLOR,
                            (x * self.cell_size, y * self.cell_size,
                             self.cell_size, self.cell_size))
        
        # Draw snake body
        for segment in snake.get_body_positions():
            x, y = segment
            pygame.draw.rect(self.screen, config.SNAKE_COLOR,
                            (x * self.cell_size, y * self.cell_size,
                             self.cell_size, self.cell_size))
        
        # Draw snake head (slightly different color)
        x, y = snake.get_head_position()
        pygame.draw.rect(self.screen, (0, 200, 0),
                        (x * self.cell_size, y * self.cell_size,
                         self.cell_size, self.cell_size))
        
        # Draw food
        x, y = food_position
        pygame.draw.rect(self.screen, config.FOOD_COLOR,
                        (x * self.cell_size, y * self.cell_size,
                         self.cell_size, self.cell_size))
        
        # Draw score and other info
        if info:
            score_text = self.font.render(f"Score: {info.get('score', 0)}", True, config.TEXT_COLOR)
            steps_text = self.font.render(f"Steps: {info.get('steps', 0)}", True, config.TEXT_COLOR)
            
            self.screen.blit(score_text, (10, 10))
            self.screen.blit(steps_text, (10, 40))
        
        # Update display
        pygame.display.flip()
        
        # Cap the frame rate
        self.clock.tick(config.FPS)
    
    def get_user_action(self):
        """
        Get user action from keyboard input.
        
        Returns:
            Direction or None: Direction to move or None for no input
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "QUIT"
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    return Direction.UP
                elif event.key == pygame.K_RIGHT:
                    return Direction.RIGHT
                elif event.key == pygame.K_DOWN:
                    return Direction.DOWN
                elif event.key == pygame.K_LEFT:
                    return Direction.LEFT
                elif event.key == pygame.K_ESCAPE:
                    return "QUIT"
        
        return None
    
    def close(self):
        """Close the Pygame window."""
        pygame.quit()
    
    def show_game_over(self, score):
        """
        Display game over screen.
        
        Args:
            score (int): Final score
        """
        self.screen.fill(config.BACKGROUND_COLOR)
        
        # Game over text
        game_over_text = self.font.render("GAME OVER", True, config.TEXT_COLOR)
        score_text = self.font.render(f"Final Score: {score}", True, config.TEXT_COLOR)
        restart_text = self.font.render("Press SPACE to restart, ESC to quit", True, config.TEXT_COLOR)
        
        self.screen.blit(game_over_text, (self.window_width // 2 - game_over_text.get_width() // 2, 
                                         self.window_height // 2 - 60))
        self.screen.blit(score_text, (self.window_width // 2 - score_text.get_width() // 2, 
                                     self.window_height // 2 - 20))
        self.screen.blit(restart_text, (self.window_width // 2 - restart_text.get_width() // 2, 
                                       self.window_height // 2 + 20))
        
        pygame.display.flip()
        
        # Wait for user input
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "QUIT"
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        return "RESTART"
                    elif event.key == pygame.K_ESCAPE:
                        return "QUIT"
            
            self.clock.tick(10)