"""
Main entry point for Snake RL application.
"""
import os
import argparse
import pygame
import time
import torch

from game.snake import Direction
from game.game_engine import GameEngine
from game.visualizer import Visualizer
from game.map_loader import MapLoader
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent
from agents.rl_agent import RLAgent
from learning.environment import SnakeEnvironment
from learning.trainer import Trainer
from utils.logger import Logger
import config

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Snake RL')
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--play', action='store_true', help='Play the game manually')
    mode_group.add_argument('--train', action='store_true', help='Train RL agent')
    mode_group.add_argument('--test', action='store_true', help='Test trained agent')
    mode_group.add_argument('--compare', action='store_true', help='Compare agent performance')
    
    # Map options
    parser.add_argument('--map', type=str, default='default.txt', help='Map file to use')
    parser.add_argument('--grid-size', type=int, default=config.GRID_SIZE, help='Grid size')
    
    # Agent options
    parser.add_argument('--agent', type=str, default='rl', 
                      choices=['random', 'heuristic', 'rl'], help='Agent type')
    parser.add_argument('--model-path', type=str, help='Path to trained model')
    
    # Training options
    parser.add_argument('--episodes', type=int, default=config.NUM_EPISODES, 
                      help='Number of episodes to train/test')
    parser.add_argument('--no-dueling', action='store_true', help='Disable dueling DQN')
    parser.add_argument('--no-priority', action='store_true', help='Disable prioritized replay')
    
    # Display options
    parser.add_argument('--fps', type=int, default=config.FPS, help='Frames per second')
    parser.add_argument('--no-render', action='store_true', help='Disable rendering')
    
    return parser.parse_args()

def play_game(map_file, grid_size, fps):
    """Play the game manually."""
    logger = Logger()
    logger.info("Starting manual play mode")
    
    # Load map
    map_loader = MapLoader()
    grid_size, obstacles = map_loader.load_map(map_file)
    
    # Initialize game
    game = GameEngine(grid_size, obstacles)
    visualizer = Visualizer(grid_size)
    
    # Set FPS
    config.FPS = fps
    
    # Game loop
    running = True
    state = game.reset()
    
    while running:
        # Render game
        visualizer.render(game.state, game.snake, game.food_position, game.obstacles, 
                         {'score': game.score, 'steps': game.steps})
        
        # Get user input
        action = visualizer.get_user_action()
        
        if action == "QUIT":
            running = False
        elif action is not None:
            # Convert Direction to action index
            action_idx = action.value
            
            # Take action
            state, reward, done, info = game.step(action_idx)
            
            if done:
                # Show game over screen
                result = visualizer.show_game_over(game.score)
                
                if result == "RESTART":
                    state = game.reset()
                elif result == "QUIT":
                    running = False
    
    visualizer.close()
    logger.info(f"Game ended | Final Score: {game.score}")

def train_agent(map_file, grid_size, agent_type, episodes, use_dueling, use_priority):
    """Train an RL agent."""
    logger = Logger()
    logger.info(f"Starting training mode with {agent_type} agent")
    
    # Load map
    map_loader = MapLoader()
    grid_size, obstacles = map_loader.load_map(map_file)
    
    # Initialize trainer
    trainer = Trainer(
        grid_size=grid_size,
        obstacles=obstacles,
        use_dueling=use_dueling,
        use_priority=use_priority
    )
    
    # Train agent
    trainer.num_episodes = episodes
    trainer.train()
    
    logger.info("Training completed")

def test_agent(map_file, grid_size, agent_type, model_path, episodes, fps, render):
    """Test a trained agent."""
    logger = Logger()
    logger.info(f"Starting test mode with {agent_type} agent")
    
    # Load map
    map_loader = MapLoader()
    grid_size, obstacles = map_loader.load_map(map_file)
    
    # Set FPS
    config.FPS = fps
    
    # Initialize environment
    env = SnakeEnvironment(
        grid_size=grid_size,
        obstacles=obstacles,
        render_mode='human' if render else None
    )
    
    # Initialize agent
    if agent_type == 'random':
        agent = RandomAgent()
    elif agent_type == 'heuristic':
        agent = HeuristicAgent()
    elif agent_type == 'rl':
        if model_path is None:
            logger.error("Model path is required for RL agent")
            return
        agent = RLAgent(load_path=model_path)
    
    # Test agent
    total_score = 0
    total_steps = 0
    
    for episode in range(episodes):
        state = env.reset()
        episode_score = 0
        episode_steps = 0
        done = False
        
        while not done:
            # Select action
            action = agent.act(state, deterministic=True)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Update state
            state = next_state
            
            # Update counters
            episode_score = info['score']
            episode_steps += 1
            
            # Add small delay for visualization
            if render:
                time.sleep(1 / fps)
        
        total_score += episode_score
        total_steps += episode_steps
        
        logger.info(f"Episode {episode+1}/{episodes} | Score: {episode_score} | Steps: {episode_steps}")
    
    avg_score = total_score / episodes
    avg_steps = total_steps / episodes
    
    logger.info(f"Testing completed | Avg Score: {avg_score:.2f} | Avg Steps: {avg_steps:.2f}")
    
    # Close environment
    env.close()

def compare_agents(map_file, grid_size, episodes, render):
    """Compare different agent types."""
    logger = Logger()
    logger.info("Starting agent comparison mode")
    
    # Load map
    map_loader = MapLoader()
    grid_size, obstacles = map_loader.load_map(map_file)
    
    # Initialize environment
    env = SnakeEnvironment(
        grid_size=grid_size,
        obstacles=obstacles,
        render_mode='human' if render else None
    )
    
    # Initialize agents
    random_agent = RandomAgent()
    heuristic_agent = HeuristicAgent()
    
    # Try to load RL agent
    try:
        rl_agent = RLAgent(load_path='models/best_model.pt')
        agents = [
            ('Random', random_agent),
            ('Heuristic', heuristic_agent),
            ('RL', rl_agent)
        ]
    except:
        logger.warning("Could not load RL agent, comparing only Random and Heuristic")
        agents = [
            ('Random', random_agent),
            ('Heuristic', heuristic_agent)
        ]
    
    # Compare agents
    results = {}
    
    for name, agent in agents:
        logger.info(f"Testing {name} agent")
        
        scores = []
        steps = []
        
        for episode in range(episodes):
            state = env.reset()
            episode_score = 0
            episode_steps = 0
            done = False
            
            while not done:
                # Select action
                action = agent.act(state, deterministic=True)
                
                # Take action
                next_state, reward, done, info = env.step(action)
                
                # Update state
                state = next_state
                
                # Update counters
                episode_score = info['score']
                episode_steps += 1
            
            scores.append(episode_score)
            steps.append(episode_steps)
            
            logger.info(f"{name} Episode {episode+1}/{episodes} | Score: {episode_score}")
        
        avg_score = sum(scores) / len(scores)
        avg_steps = sum(steps) / len(steps)
        max_score = max(scores)
        
        results[name] = {
            'avg_score': avg_score,
            'avg_steps': avg_steps,
            'max_score': max_score
        }
        
        logger.info(f"{name} - Avg Score: {avg_score:.2f} | Max Score: {max_score} | Avg Steps: {avg_steps:.2f}")
    
    # Print comparison
    logger.info("\nAgent Comparison Results:")
    for name, metrics in results.items():
        logger.info(f"{name:10} | Avg Score: {metrics['avg_score']:.2f} | Max Score: {metrics['max_score']}")
    
    # Close environment
    env.close()

def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('metrics', exist_ok=True)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # Run selected mode
    if args.play:
        play_game(args.map, args.grid_size, args.fps)
    elif args.train:
        train_agent(
            args.map, 
            args.grid_size, 
            args.agent, 
            args.episodes,
            not args.no_dueling,
            not args.no_priority
        )
    elif args.test:
        test_agent(
            args.map,
            args.grid_size,
            args.agent,
            args.model_path,
            args.episodes,
            args.fps,
            not args.no_render
        )
    elif args.compare:
        compare_agents(
            args.map,
            args.grid_size,
            args.episodes,
            not args.no_render
        )

if __name__ == '__main__':
    main()