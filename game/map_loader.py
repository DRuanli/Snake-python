"""
Map loader for Snake RL - loads and validates custom maps.
"""
import os
import numpy as np
import config

class MapLoader:
    """Loads, validates, and processes custom map files."""
    
    def __init__(self, maps_dir="maps"):
        """Initialize the map loader with maps directory."""
        self.maps_dir = maps_dir
        self._ensure_maps_dir_exists()
    
    def _ensure_maps_dir_exists(self):
        """Create maps directory if it doesn't exist."""
        if not os.path.exists(self.maps_dir):
            os.makedirs(self.maps_dir)
            self._create_default_maps()
    
    def _create_default_maps(self):
        """Create some default map files if none exist."""
        # Create a default empty map
        default_map = self._create_empty_map(config.GRID_SIZE, config.GRID_SIZE)
        self.save_map(default_map, "default.txt")
        
        # Create a small map
        small_map = self._create_empty_map(10, 10)
        self.save_map(small_map, "small.txt")
        
        # Create a map with obstacles
        obstacles_map = self._create_empty_map(config.GRID_SIZE, config.GRID_SIZE)
        # Add some obstacles
        for i in range(5, 15):
            obstacles_map[i][5] = config.OBSTACLE
        for i in range(5, 15):
            obstacles_map[5][i] = config.OBSTACLE
        self.save_map(obstacles_map, "obstacles.txt")
    
    def _create_empty_map(self, width, height):
        """Create an empty map grid."""
        return [[config.EMPTY_CELL for _ in range(width)] for _ in range(height)]
    
    def load_map(self, map_name):
        """
        Load a map from file.
        
        Args:
            map_name (str): Name of the map file
            
        Returns:
            tuple: (grid_size, obstacles_list)
        """
        map_path = os.path.join(self.maps_dir, map_name)
        
        if not os.path.exists(map_path):
            print(f"Map file {map_path} not found. Using default map.")
            map_path = os.path.join(self.maps_dir, "default.txt")
        
        with open(map_path, 'r') as f:
            lines = f.readlines()
        
        # Remove whitespace and newlines
        lines = [line.strip() for line in lines if line.strip()]
        
        # Validate map dimensions
        height = len(lines)
        width = len(lines[0])
        
        if not all(len(line) == width for line in lines):
            raise ValueError("Map must be rectangular (all rows must have the same length)")
        
        # Extract obstacles
        obstacles = []
        for y, line in enumerate(lines):
            for x, cell in enumerate(line):
                if cell == config.OBSTACLE:
                    obstacles.append((x, y))
        
        return max(width, height), obstacles
    
    def save_map(self, map_grid, map_name):
        """
        Save a map to file.
        
        Args:
            map_grid (list): 2D list representing the map
            map_name (str): Name of the map file
        """
        map_path = os.path.join(self.maps_dir, map_name)
        
        with open(map_path, 'w') as f:
            for row in map_grid:
                f.write(''.join(row) + '\n')
    
    def get_available_maps(self):
        """
        Get a list of available map files.
        
        Returns:
            list: List of map filenames
        """
        return [f for f in os.listdir(self.maps_dir) if f.endswith('.txt')]
    
    def generate_random_map(self, width, height, obstacle_density=0.1):
        """
        Generate a random map with obstacles.
        
        Args:
            width (int): Map width
            height (int): Map height
            obstacle_density (float): Percentage of cells to be obstacles
            
        Returns:
            list: 2D list representing the map
        """
        map_grid = self._create_empty_map(width, height)
        
        # Add obstacles
        total_cells = width * height
        obstacle_count = int(total_cells * obstacle_density)
        
        # Keep center clear for snake starting position
        center_x, center_y = width // 2, height // 2
        clear_zone = [(x, y) for x in range(center_x-2, center_x+3) 
                           for y in range(center_y-2, center_y+3)]
        
        obstacles_placed = 0
        while obstacles_placed < obstacle_count:
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            
            # Avoid placing obstacles in the clear zone
            if (x, y) not in clear_zone and map_grid[y][x] != config.OBSTACLE:
                map_grid[y][x] = config.OBSTACLE
                obstacles_placed += 1
        
        return map_grid