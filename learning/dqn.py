"""
DQN agent implementation for Snake RL.
"""
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
from agents.base_agent import BaseAgent
from learning.model import DQNModel, DuelingDQNModel
from learning.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
import config

class DQNAgent(BaseAgent):
    """
    Deep Q-Network agent with experience replay and target network.
    """
    
    def __init__(self, state_size=config.INPUT_SIZE, action_size=config.OUTPUT_SIZE, 
                 use_dueling=False, use_priority=False, device=None):
        """
        Initialize the DQN agent.
        
        Args:
            state_size (int): Size of the state space
            action_size (int): Size of the action space
            use_dueling (bool): Whether to use dueling network architecture
            use_priority (bool): Whether to use prioritized experience replay
            device (torch.device): Device to run the model on
        """
        super().__init__(action_size)
        
        # Set device
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize Q-networks
        if use_dueling:
            self.policy_net = DuelingDQNModel(state_size, config.HIDDEN_SIZE, action_size).to(self.device)
            self.target_net = DuelingDQNModel(state_size, config.HIDDEN_SIZE, action_size).to(self.device)
        else:
            self.policy_net = DQNModel(state_size, config.HIDDEN_SIZE, action_size).to(self.device)
            self.target_net = DQNModel(state_size, config.HIDDEN_SIZE, action_size).to(self.device)
        
        # Copy parameters from policy net to target net
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target net in evaluation mode
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.LEARNING_RATE)
        
        # Initialize replay buffer
        if use_priority:
            self.memory = PrioritizedReplayBuffer(config.MEMORY_SIZE)
            self.use_priority = True
        else:
            self.memory = ReplayBuffer(config.MEMORY_SIZE)
            self.use_priority = False
        
        # Training parameters
        self.batch_size = config.BATCH_SIZE
        self.gamma = config.GAMMA
        self.epsilon = config.EPSILON_START
        self.epsilon_end = config.EPSILON_END
        self.epsilon_decay = config.EPSILON_DECAY
        self.target_update = config.TARGET_UPDATE
        
        # Training stats
        self.steps_done = 0
        self.episode_rewards = []
    
    def act(self, state, deterministic=False):
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state of the environment
            deterministic (bool): Whether to act deterministically (no exploration)
            
        Returns:
            int: Action to take
        """
        # Calculate epsilon (linearly decreasing)
        self.epsilon = max(self.epsilon_end, self.epsilon - (self.epsilon - self.epsilon_end) / self.epsilon_decay)
        
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Epsilon-greedy action selection
        if not deterministic and random.random() < self.epsilon:
            # Random action
            return random.randint(0, self.action_space_size - 1)
        else:
            # Greedy action
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                return q_values.max(1)[1].item()
    
    def train(self, state, action, next_state, reward, done):
        """
        Store experience in replay buffer and perform learning if enough samples.
        
        Args:
            state: Current state
            action (int): Action taken
            next_state: Next state
            reward (float): Reward received
            done (bool): Whether the episode is done
        """
        # Store transition in replay buffer
        self.memory.add(state, np.array([action]), np.array([reward]), next_state, np.array([done]))
        
        # Increment steps
        self.steps_done += 1
        
        # Update target network periodically
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Only train if enough samples in memory
        if len(self.memory) < self.batch_size:
            return
        
        # Sample transitions from replay buffer
        if self.use_priority:
            states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.batch_size)
        else:
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
            weights = torch.ones_like(rewards)
        
        # Move tensors to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)
        
        # Compute Q values for current states
        q_values = self.policy_net(states).gather(1, actions)
        
        # Compute V value for next states using target network (Double DQN)
        with torch.no_grad():
            # Get actions from policy net
            next_actions = self.policy_net(next_states).max(1, keepdim=True)[1]
            # Get Q values from target net
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            # Compute the expected Q values
            expected_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = (weights * F.smooth_l1_loss(q_values, expected_q_values, reduction='none')).mean()
        
        # Update priorities in buffer if using prioritized replay
        if self.use_priority:
            with torch.no_grad():
                td_errors = (q_values - expected_q_values).abs().cpu().numpy()
                self.memory.update_priorities(indices, td_errors)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to stabilize training
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
    
    def save(self, path):
        """
        Save the agent's model.
        
        Args:
            path (str): Path to save to
        """
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done
        }, path)
    
    def load(self, path):
        """
        Load the agent's model.
        
        Args:
            path (str): Path to load from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        
        # Set target net to eval mode
        self.target_net.eval()