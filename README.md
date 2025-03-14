# Snake RL - A Reinforcement Learning Snake Game

This project implements the classic Snake game with reinforcement learning capabilities. The snake agent can learn to play the game on custom maps through Deep Q-Learning.

## Features

- Classic Snake game implementation with Pygame
- Custom map support through text-based map files
- Multiple agent types:
  - Random agent (baseline)
  - Heuristic agent (rule-based)
  - Reinforcement learning agent (DQN)
- Interactive gameplay mode
- Training mode with performance metrics and visualization
- Testing mode to evaluate trained agents
- Agent comparison mode

## Project Structure

```
snake_rl/
├── main.py                    # Main entry point
├── config.py                  # Configuration parameters
├── game/
│   ├── __init__.py
│   ├── snake.py               # Snake class and movement logic
│   ├── game_engine.py         # Core game mechanics
│   ├── map_loader.py          # Load and validate custom maps
│   └── visualizer.py          # Game visualization
├── agents/
│   ├── __init__.py
│   ├── base_agent.py          # Abstract base class for agents
│   ├── random_agent.py        # Simple random movement agent (baseline)
│   ├── heuristic_agent.py     # Rule-based agent 
│   └── rl_agent.py            # Reinforcement learning agent
├── learning/
│   ├── __init__.py
│   ├── environment.py         # RL environment (game wrapper)
│   ├── model.py               # Neural network model definition
│   ├── dqn.py                 # Deep Q-Network implementation
│   ├── replay_buffer.py       # Experience replay buffer
│   └── trainer.py             # Training orchestration
├── utils/
│   ├── __init__.py
│   ├── logger.py              # Logging utilities
│   └── metrics.py             # Performance metrics tracking
├── maps/                      # Directory for map files
│   ├── default.txt            # Default empty map
│   ├── obstacles.txt          # Map with simple obstacles
│   └── maze.txt               # Complex maze map
└── README.md                  # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/snake-rl.git
cd snake-rl
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Playing the Game Manually

```bash
python main.py --play --map default.txt
```

Controls:
- Arrow keys to control the snake
- ESC to quit

### Training an Agent

```bash
python main.py --train --episodes 1000 --map obstacles.txt
```

Options:
- `--no-dueling`: Disable dueling DQN architecture
- `--no-priority`: Disable prioritized experience replay

### Testing a Trained Agent

```bash
python main.py --test --agent rl --model-path models/best_model.pt --episodes 10
```

Options:
- `--agent`: Choose between `random`, `heuristic`, or `rl`
- `--fps`: Set visualization speed (default: 10)
- `--no-render`: Disable visualization for faster testing

### Comparing Agent Performance

```bash
python main.py --compare --episodes 20 --map maze.txt
```

## Creating Custom Maps

Maps are text files where:
- `.` represents an empty cell
- `X` represents an obstacle

Example:
```
XXXXXXXXXXXX
X..........X
X..........X
X..........X
X..........X
XXXXXXXXXXXX
```

Place custom maps in the `maps/` directory.

## Advanced Features

### DQN Implementation

The reinforcement learning agent uses:
- Deep Q-Network with experience replay
- Optional dueling architecture
- Optional prioritized experience replay
- Target network for stable learning
- Epsilon-greedy exploration strategy

### Reward Structure

- Positive reward for eating food
- Small negative reward for each step
- Large negative reward for collisions
- Small positive/negative rewards for moving toward/away from food

## Performance Metrics

Training generates:
- Reward plots
- Score plots
- Evaluation metrics
- JSON files with detailed statistics

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PyTorch for the neural network implementation
- Pygame for the visualization