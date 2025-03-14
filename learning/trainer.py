"""
Trainer for RL agents in Snake game.
"""
import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
from agents.rl_agent import RLAgent
from learning.environment import SnakeEnvironment
from utils.logger import Logger
from utils.metrics import MetricsTracker
import config

class Trainer:
    """
    Trainer class for RL agents in Snake game.
    Handles training, evaluation, and model management.
    """
    
    def __init__(self, grid_size=config.GRID_SIZE, obstacles=None, 
                 use_dueling=True, use_priority=True, save_dir='models'):
        """
        Initialize the trainer.
        
        Args:
            grid_size (int): Size of the game grid
            obstacles (list): List of obstacle positions
            use_dueling (bool): Whether to use dueling network architecture
            use_priority (bool): Whether to use prioritized experience replay
            save_dir (str): Directory to save models
        """
        # Initialize environment
        self.env = SnakeEnvironment(grid_size, obstacles)
        
        # Initialize agent
        self.agent = RLAgent(use_dueling=use_dueling, use_priority=use_priority)
        
        # Initialize logger and metrics
        self.logger = Logger()
        self.metrics = MetricsTracker()
        
        # Create save directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save_dir = save_dir
        
        # Training settings
        self.num_episodes = config.NUM_EPISODES
        self.save_interval = config.SAVE_INTERVAL
        self.eval_interval = config.EVAL_INTERVAL
        
        # Tracking variables
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_scores = []
        self.eval_scores = []
        self.best_eval_score = -float('inf')
    
    def train(self):
        """
        Train the agent for a specified number of episodes.
        
        Returns:
            tuple: (episode_rewards, episode_lengths, episode_scores, eval_scores)
        """
        self.logger.info(f"Starting training for {self.num_episodes} episodes...")
        
        # Track moving average of rewards
        reward_window = deque(maxlen=100)
        
        start_time = time.time()
        
        for episode in range(1, self.num_episodes + 1):
            # Reset environment
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            # Episode loop
            while not done:
                # Select action
                action = self.agent.act(state)
                
                # Take action
                next_state, reward, done, info = self.env.step(action)
                
                # Train agent
                self.agent.train(state, action, next_state, reward, done)
                
                # Update state
                state = next_state
                
                # Update tracking variables
                episode_reward += reward
                episode_length += 1
            
            # Record episode metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.episode_scores.append(info['score'])
            
            # Update reward window
            reward_window.append(episode_reward)
            
            # Log progress
            if episode % 10 == 0:
                avg_reward = np.mean(reward_window)
                time_elapsed = time.time() - start_time
                
                self.logger.info(f"Episode {episode}/{self.num_episodes} | " +
                               f"Score: {info['score']} | " +
                               f"Avg Reward: {avg_reward:.2f} | " +
                               f"Epsilon: {self.agent.dqn.epsilon:.2f} | " +
                               f"Time: {time_elapsed:.2f}s")
            
            # Periodically save model
            if episode % self.save_interval == 0:
                model_path = os.path.join(self.save_dir, f"model_episode_{episode}.pt")
                self.agent.save(model_path)
                self.logger.info(f"Model saved to {model_path}")
            
            # Periodically evaluate agent
            if episode % self.eval_interval == 0:
                eval_score = self.evaluate(10)
                self.eval_scores.append(eval_score)
                
                # Save best model
                if eval_score > self.best_eval_score:
                    self.best_eval_score = eval_score
                    best_model_path = os.path.join(self.save_dir, "best_model.pt")
                    self.agent.save(best_model_path)
                    self.logger.info(f"New best model saved with score {eval_score}")
        
        # Save final model
        final_model_path = os.path.join(self.save_dir, "final_model.pt")
        self.agent.save(final_model_path)
        self.logger.info(f"Final model saved to {final_model_path}")
        
        # Plot training metrics
        self._plot_training_metrics()
        
        return self.episode_rewards, self.episode_lengths, self.episode_scores, self.eval_scores
    
    def evaluate(self, num_episodes=10):
        """
        Evaluate the agent's performance.
        
        Args:
            num_episodes (int): Number of episodes to evaluate
            
        Returns:
            float: Average score over episodes
        """
        self.logger.info(f"Evaluating agent for {num_episodes} episodes...")
        
        scores = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_score = 0
            done = False
            
            while not done:
                # Select action deterministically
                action = self.agent.act(state, deterministic=True)
                
                # Take action
                next_state, _, done, info = self.env.step(action)
                
                # Update state
                state = next_state
                
                # Record score
                episode_score = info['score']
            
            scores.append(episode_score)
        
        avg_score = np.mean(scores)
        self.logger.info(f"Evaluation complete | Avg Score: {avg_score:.2f}")
        
        return avg_score
    
    def _plot_training_metrics(self):
        """Plot training metrics."""
        plt.figure(figsize=(15, 10))
        
        # Plot episode rewards
        plt.subplot(2, 2, 1)
        plt.plot(self.episode_rewards)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        # Plot episode lengths
        plt.subplot(2, 2, 2)
        plt.plot(self.episode_lengths)
        plt.title('Episode Lengths')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        
        # Plot episode scores
        plt.subplot(2, 2, 3)
        plt.plot(self.episode_scores)
        plt.title('Episode Scores')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        
        # Plot evaluation scores
        plt.subplot(2, 2, 4)
        eval_episodes = np.arange(self.eval_interval, self.num_episodes + 1, self.eval_interval)
        plt.plot(eval_episodes, self.eval_scores)
        plt.title('Evaluation Scores')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        
        plt.tight_layout()
        plt.savefig('training_metrics.png')
        plt.close()
        
        self.logger.info("Training metrics plotted and saved to training_metrics.png")