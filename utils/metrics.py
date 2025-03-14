"""
Performance metrics tracking for Snake RL.
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time
import json
import os

class MetricsTracker:
    """
    Tracks and records metrics during training and evaluation.
    """
    
    def __init__(self, save_dir='metrics', window_size=100):
        """
        Initialize metrics tracker.
        
        Args:
            save_dir (str): Directory to save metrics
            window_size (int): Size of the moving average window
        """
        # Create metrics directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        self.save_dir = save_dir
        self.window_size = window_size
        
        # Initialize metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_scores = []
        self.evaluation_scores = []
        
        # Moving averages
        self.reward_window = deque(maxlen=window_size)
        self.length_window = deque(maxlen=window_size)
        self.score_window = deque(maxlen=window_size)
        
        # Training time tracking
        self.start_time = None
        self.total_training_time = 0
        
        # Timestamp for saving
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    def start_training(self):
        """Record the start of training."""
        self.start_time = time.time()
    
    def end_training(self):
        """Record the end of training and calculate total time."""
        if self.start_time is not None:
            self.total_training_time = time.time() - self.start_time
            self.start_time = None
    
    def record_episode(self, reward, length, score):
        """
        Record metrics for a single episode.
        
        Args:
            reward (float): Total reward for the episode
            length (int): Length of the episode
            score (int): Game score for the episode
        """
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.episode_scores.append(score)
        
        self.reward_window.append(reward)
        self.length_window.append(length)
        self.score_window.append(score)
    
    def record_evaluation(self, score):
        """
        Record evaluation score.
        
        Args:
            score (float): Average score from evaluation
        """
        self.evaluation_scores.append(score)
    
    def get_averages(self):
        """
        Get moving averages of metrics.
        
        Returns:
            tuple: (avg_reward, avg_length, avg_score)
        """
        avg_reward = np.mean(self.reward_window) if self.reward_window else 0
        avg_length = np.mean(self.length_window) if self.length_window else 0
        avg_score = np.mean(self.score_window) if self.score_window else 0
        
        return avg_reward, avg_length, avg_score
    
    def plot_metrics(self, save=True):
        """
        Plot training metrics.
        
        Args:
            save (bool): Whether to save the plots
            
        Returns:
            tuple: (fig_rewards, fig_scores, fig_eval) - matplotlib figures
        """
        # Plot episode rewards
        fig_rewards, ax_rewards = plt.subplots(figsize=(10, 5))
        ax_rewards.plot(self.episode_rewards, label='Episode Reward')
        
        # Plot moving average
        if len(self.episode_rewards) >= self.window_size:
            moving_avg = []
            for i in range(len(self.episode_rewards) - self.window_size + 1):
                moving_avg.append(np.mean(self.episode_rewards[i:i+self.window_size]))
            ax_rewards.plot(range(self.window_size-1, len(self.episode_rewards)), 
                           moving_avg, label=f'{self.window_size}-Episode Moving Average')
        
        ax_rewards.set_xlabel('Episode')
        ax_rewards.set_ylabel('Reward')
        ax_rewards.set_title('Episode Rewards')
        ax_rewards.legend()
        ax_rewards.grid(True, alpha=0.3)
        
        # Plot episode scores
        fig_scores, ax_scores = plt.subplots(figsize=(10, 5))
        ax_scores.plot(self.episode_scores, label='Episode Score')
        
        # Plot moving average
        if len(self.episode_scores) >= self.window_size:
            moving_avg = []
            for i in range(len(self.episode_scores) - self.window_size + 1):
                moving_avg.append(np.mean(self.episode_scores[i:i+self.window_size]))
            ax_scores.plot(range(self.window_size-1, len(self.episode_scores)), 
                          moving_avg, label=f'{self.window_size}-Episode Moving Average')
        
        ax_scores.set_xlabel('Episode')
        ax_scores.set_ylabel('Score')
        ax_scores.set_title('Episode Scores')
        ax_scores.legend()
        ax_scores.grid(True, alpha=0.3)
        
        # Plot evaluation scores
        fig_eval = None
        if self.evaluation_scores:
            fig_eval, ax_eval = plt.subplots(figsize=(10, 5))
            ax_eval.plot(self.evaluation_scores, marker='o')
            ax_eval.set_xlabel('Evaluation')
            ax_eval.set_ylabel('Average Score')
            ax_eval.set_title('Evaluation Scores')
            ax_eval.grid(True, alpha=0.3)
        
        # Save figures
        if save:
            rewards_path = os.path.join(self.save_dir, f'rewards_{self.timestamp}.png')
            scores_path = os.path.join(self.save_dir, f'scores_{self.timestamp}.png')
            
            fig_rewards.savefig(rewards_path)
            fig_scores.savefig(scores_path)
            
            if fig_eval:
                eval_path = os.path.join(self.save_dir, f'eval_{self.timestamp}.png')
                fig_eval.savefig(eval_path)
        
        return fig_rewards, fig_scores, fig_eval
    
    def save_metrics(self):
        """
        Save metrics to JSON file.
        
        Returns:
            str: Path to saved metrics file
        """
        metrics = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_scores': self.episode_scores,
            'evaluation_scores': self.evaluation_scores,
            'training_time': self.total_training_time,
            'timestamp': self.timestamp
        }
        
        # Calculate summary statistics
        if self.episode_rewards:
            metrics['summary'] = {
                'max_reward': max(self.episode_rewards),
                'avg_reward': np.mean(self.episode_rewards),
                'max_score': max(self.episode_scores),
                'avg_score': np.mean(self.episode_scores),
                'max_eval_score': max(self.evaluation_scores) if self.evaluation_scores else None,
                'episodes': len(self.episode_rewards)
            }
        
        file_path = os.path.join(self.save_dir, f'metrics_{self.timestamp}.json')
        with open(file_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        return file_path
    
    def load_metrics(self, file_path):
        """
        Load metrics from JSON file.
        
        Args:
            file_path (str): Path to metrics file
            
        Returns:
            bool: Whether loading was successful
        """
        try:
            with open(file_path, 'r') as f:
                metrics = json.load(f)
            
            self.episode_rewards = metrics['episode_rewards']
            self.episode_lengths = metrics['episode_lengths']
            self.episode_scores = metrics['episode_scores']
            self.evaluation_scores = metrics['evaluation_scores']
            self.total_training_time = metrics['training_time']
            self.timestamp = metrics['timestamp']
            
            # Update moving averages
            self.reward_window.clear()
            self.length_window.clear()
            self.score_window.clear()
            
            for i in range(min(self.window_size, len(self.episode_rewards))):
                idx = len(self.episode_rewards) - self.window_size + i
                if idx >= 0:
                    self.reward_window.append(self.episode_rewards[idx])
                    self.length_window.append(self.episode_lengths[idx])
                    self.score_window.append(self.episode_scores[idx])
            
            return True
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            print(f"Error loading metrics: {e}")
            return False
    
    def compare_metrics(self, other_metrics_path, plot=True):
        """
        Compare current metrics with another set of metrics.
        
        Args:
            other_metrics_path (str): Path to other metrics file
            plot (bool): Whether to plot comparison
            
        Returns:
            dict: Comparison results
        """
        try:
            with open(other_metrics_path, 'r') as f:
                other_metrics = json.load(f)
            
            other_rewards = other_metrics['episode_rewards']
            other_scores = other_metrics['episode_scores']
            other_eval = other_metrics['evaluation_scores']
            
            # Calculate comparison statistics
            comparison = {
                'current_max_reward': max(self.episode_rewards) if self.episode_rewards else None,
                'other_max_reward': max(other_rewards) if other_rewards else None,
                'current_avg_reward': np.mean(self.episode_rewards) if self.episode_rewards else None,
                'other_avg_reward': np.mean(other_rewards) if other_rewards else None,
                'current_max_score': max(self.episode_scores) if self.episode_scores else None,
                'other_max_score': max(other_scores) if other_scores else None,
                'current_avg_score': np.mean(self.episode_scores) if self.episode_scores else None,
                'other_avg_score': np.mean(other_scores) if other_scores else None,
                'current_max_eval': max(self.evaluation_scores) if self.evaluation_scores else None,
                'other_max_eval': max(other_eval) if other_eval else None
            }
            
            # Plot comparison
            if plot:
                # Compare rewards
                plt.figure(figsize=(12, 6))
                plt.plot(self.episode_rewards, label='Current')
                plt.plot(other_rewards, label='Other')
                plt.xlabel('Episode')
                plt.ylabel('Reward')
                plt.title('Reward Comparison')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                compare_path = os.path.join(self.save_dir, f'compare_rewards_{self.timestamp}.png')
                plt.savefig(compare_path)
                
                # Compare scores
                plt.figure(figsize=(12, 6))
                plt.plot(self.episode_scores, label='Current')
                plt.plot(other_scores, label='Other')
                plt.xlabel('Episode')
                plt.ylabel('Score')
                plt.title('Score Comparison')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                compare_path = os.path.join(self.save_dir, f'compare_scores_{self.timestamp}.png')
                plt.savefig(compare_path)
            
            return comparison
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            print(f"Error comparing metrics: {e}")
            return None