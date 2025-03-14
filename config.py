"""
Configuration settings for the Snake RL game.
"""

# Game settings
GRID_SIZE = 20  # Size of the grid (GRID_SIZE x GRID_SIZE)
CELL_SIZE = 20  # Size of each cell in pixels
FPS = 10  # Frames per second for visualization
MAX_STEPS = 2000  # Maximum steps per episode

# Display settings
WINDOW_WIDTH = GRID_SIZE * CELL_SIZE
WINDOW_HEIGHT = GRID_SIZE * CELL_SIZE
BACKGROUND_COLOR = (0, 0, 0)  # Black
SNAKE_COLOR = (0, 255, 0)  # Green
FOOD_COLOR = (255, 0, 0)  # Red
OBSTACLE_COLOR = (128, 128, 128)  # Gray
TEXT_COLOR = (255, 255, 255)  # White

# Map symbols
EMPTY_CELL = '.'
SNAKE_HEAD = 'H'
SNAKE_BODY = 'B'
FOOD = 'F'
OBSTACLE = 'X'

# RL settings
LEARNING_RATE = 0.001
BATCH_SIZE = 128
GAMMA = 0.99  # Discount factor
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 10000  # Number of steps to decay epsilon over
TARGET_UPDATE = 1000  # Steps between target network updates
MEMORY_SIZE = 100000  # Size of replay buffer

# Neural network settings
INPUT_SIZE = 11  # State representation size
HIDDEN_SIZE = 128  # Size of hidden layer
OUTPUT_SIZE = 4  # Number of possible actions (up, down, left, right)

# Training settings
NUM_EPISODES = 10000
SAVE_INTERVAL = 1000  # Episodes between model saves
EVAL_INTERVAL = 500  # Episodes between evaluations