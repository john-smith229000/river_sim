#!/usr/bin/env python3
from dataclasses import dataclass, field
from pathlib import Path
import yaml

@dataclass
class SimulationConfig:
    # Domain
    width: int = 200
    height: int = 50
    dx: float = 5.0
    dy: float = 5.0
    
    # Time
    base_dt: float = 0.5
    total_time: float = 5000.0
    output_interval: int = 200
    
    # Hydraulics
    discharge: float = 10.0
    manning_n: float = 0.03
    min_depth: float = 0.01
    
    # Erosion
    erosion_rate: float = 5e-5
    critical_shear: float = 0.1
    deposition_rate: float = 2e-5
    transport_capacity_coeff: float = 0.001
    sediment_supply_concentration: float = 0.005

    bed_porosity: float = 0.3

    # Coefficient for lateral erosion / bank slumping.
    # A small value (e.g., 0.1) helps stabilize the channel by preventing
    # the formation of overly steep canyon walls.
    bed_diffusion_coeff: float = 0.01
    
    # Initial terrain
    initial_slope: float = 0.003
    valley_depth: float = 2.0
    noise_amplitude: float = 0.1
    
    # Inlet configuration
    inlet_width: int = 5
    
    # Output
    output_dir: str = "output"
    experiment_name: str = "fast_stable_river"

def load_config(config_path: Path) -> SimulationConfig:
    """Loads configuration from a YAML file into the dataclass."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create a config instance and update it from the file
    config = SimulationConfig()
    for key, value in config_dict.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"Warning: Unknown configuration key '{key}' in {config_path}")
            
    return config