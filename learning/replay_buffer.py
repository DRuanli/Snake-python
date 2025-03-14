"""
Experience replay buffer for DQN training.
"""
import random
import numpy as np
import torch
from collections import deque, namedtuple
import config

# Define an experience tuple
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions.
    Used to break correlation between consecutive samples.
    """
    
    def __init__(self, capacity=config.MEMORY_SIZE):
        """
        Initialize the replay buffer.
        
        Args:
            capacity (int): Maximum capacity of the buffer
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """
        Sample a random batch of experiences from the buffer.
        
        Args:
            batch_size (int): Size of the batch to sample
            
        Returns:
            tuple: Batch of (states, actions, rewards, next_states, dones)
        """
        experiences = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        # Convert to numpy arrays
        states = np.vstack([e.state for e in experiences])
        actions = np.vstack([e.action for e in experiences])
        rewards = np.vstack([e.reward for e in experiences])
        next_states = np.vstack([e.next_state for e in experiences])
        dones = np.vstack([e.done for e in experiences]).astype(np.uint8)
        
        # Convert to torch tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """
    Prioritized experience replay buffer that samples important transitions more frequently.
    Uses TD error as a measure of importance.
    """
    
    def __init__(self, capacity=config.MEMORY_SIZE, alpha=0.6, beta=0.4, beta_increment=0.001):
        """
        Initialize the prioritized replay buffer.
        
        Args:
            capacity (int): Maximum capacity of the buffer
            alpha (float): How much prioritization to use (0 - no prioritization, 1 - full prioritization)
            beta (float): Importance sampling weight (0 - no correction, 1 - full correction)
            beta_increment (float): Increment to beta after each sampling
        """
        self.buffer = []
        self.capacity = capacity
        self.position = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_priority = 1.0
    
    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        experience = Experience(state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        # Use max priority for new experiences
        self.priorities[self.position] = self.max_priority
        
        # Increment position
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """
        Sample a batch of experiences based on their priorities.
        
        Args:
            batch_size (int): Size of the batch to sample
            
        Returns:
            tuple: Batch of (states, actions, rewards, next_states, dones, indices, weights)
        """
        # Calculate sampling probabilities
        n_samples = min(len(self.buffer), self.capacity)
        priorities = self.priorities[:n_samples]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices based on probabilities
        indices = np.random.choice(n_samples, batch_size, replace=False, p=probabilities)
        experiences = [self.buffer[idx] for idx in indices]
        
        # Calculate importance sampling weights
        weights = (n_samples * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = torch.FloatTensor(weights)
        
        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Convert to numpy arrays
        states = np.vstack([e.state for e in experiences])
        actions = np.vstack([e.action for e in experiences])
        rewards = np.vstack([e.reward for e in experiences])
        next_states = np.vstack([e.next_state for e in experiences])
        dones = np.vstack([e.done for e in experiences]).astype(np.uint8)
        
        # Convert to torch tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices, td_errors):
        """
        Update priorities based on TD errors.
        
        Args:
            indices (list): Indices of experiences to update
            td_errors (list): TD errors for each experience
        """
        for idx, td_error in zip(indices, td_errors):
            # Add a small constant to avoid zero priority
            priority = abs(td_error) + 1e-5
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        """Return the current size of the buffer."""
        return min(len(self.buffer), self.capacity)