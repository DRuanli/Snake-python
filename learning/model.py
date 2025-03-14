"""
Neural network model for Snake RL.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class DQNModel(nn.Module):
    """
    Deep Q-Network model architecture.
    Takes state observations and outputs Q-values for each action.
    """
    
    def __init__(self, input_size=config.INPUT_SIZE, hidden_size=config.HIDDEN_SIZE, 
                 output_size=config.OUTPUT_SIZE):
        """
        Initialize the neural network.
        
        Args:
            input_size (int): Size of the input (state)
            hidden_size (int): Size of the hidden layer
            output_size (int): Size of the output (actions)
        """
        super(DQNModel, self).__init__()
        
        # Network layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor representing the state
            
        Returns:
            torch.Tensor: Output tensor representing Q-values for each action
        """
        # First hidden layer with ReLU activation
        x = F.relu(self.fc1(x))
        
        # Second hidden layer with ReLU activation
        x = F.relu(self.fc2(x))
        
        # Output layer (no activation - raw Q-values)
        x = self.fc3(x)
        
        return x

class DuelingDQNModel(nn.Module):
    """
    Dueling DQN architecture that separates state value and action advantage.
    Can learn which states are valuable without having to learn the effect of
    each action for each state.
    """
    
    def __init__(self, input_size=config.INPUT_SIZE, hidden_size=config.HIDDEN_SIZE, 
                 output_size=config.OUTPUT_SIZE):
        """
        Initialize the neural network.
        
        Args:
            input_size (int): Size of the input (state)
            hidden_size (int): Size of the hidden layer
            output_size (int): Size of the output (actions)
        """
        super(DuelingDQNModel, self).__init__()
        
        # Shared feature layer
        self.feature = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor representing the state
            
        Returns:
            torch.Tensor: Output tensor representing Q-values for each action
        """
        # Extract features
        features = self.feature(x)
        
        # Calculate value
        value = self.value_stream(features)
        
        # Calculate advantage
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage to get Q-values
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values