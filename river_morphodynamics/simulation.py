#!/usr/bin/env python3
import logging
from pathlib import Path

import numpy as np

from config import SimulationConfig
from erosion import compute_erosion_deposition
from hydraulics import compute_flow
from visualization import create_visualization

logger = logging.getLogger(__name__)

class RiverSimulation:
    """Orchestrates the river morphodynamics simulation."""

    def __init__(self, config: SimulationConfig):
        self.cfg = config
        self.time = 0.0
        self.step = 0
        self.dt = config.base_dt
        
        # Initialize grid arrays
        self.z = np.zeros((config.width, config.height))  # Bed elevation
        self.h = np.zeros((config.width, config.height))  # Water depth
        self.u = np.zeros((config.width, config.height))  # X-velocity
        self.v = np.zeros((config.width, config.height))  # Y-velocity
        self.c = np.zeros((config.width, config.height))  # Sediment concentration
        self.z0 = None # Initial bed elevation for incision calculation
        
        # Metrics tracking
        self.metrics = {'max_incision': [], 'mean_velocity': []}
        
        # Output directory setup
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def initialize(self):
        """Initializes the terrain and water."""
        w, h = self.cfg.width, self.cfg.height
        
        # Create a base slope
        # Create the full 2D bed array with the correct shape first
        self.z = np.zeros((w, h))

        # Calculate the downstream slope component (a 1D array)
        slope_component = (w - np.arange(w)) * self.cfg.dx * self.cfg.initial_slope

        # Use broadcasting to apply the 1D slope across the 2D bed
        self.z += slope_component[:, np.newaxis]
        
        # Carve a meandering valley
        y_coords = np.arange(h)
        valley_center = h / 2
        for i in range(w):
            offset = 3 * np.sin(2 * np.pi * i / (w * 0.3))
            dist = np.abs(y_coords - valley_center - offset) * self.cfg.dy
            valley_factor = np.exp(-(dist**2) / 100)
            self.z[i, :] -= self.cfg.valley_depth * valley_factor
            
        # Add random noise for realism
        self.z += np.random.normal(0, self.cfg.noise_amplitude, (w, h))
        
        # Initialize shallow water and sediment
        self.h.fill(self.cfg.min_depth)
        self.c.fill(0.0)
        
        self.z0 = self.z.copy()
        logger.info(f"Initialized {w}x{h} domain for experiment '{self.cfg.experiment_name}'")

    def simulate_step(self):
        """Performs one full step of the simulation."""
        # 1. Calculate water flow and update depth
        self.h, self.u, self.v, S, self.dt = compute_flow(self)
        
        # 2. Calculate erosion and deposition
        self.z, self.c = compute_erosion_deposition(self, S)
        
        self.time += self.dt
        self.step += 1
        self._update_metrics()

    def _update_metrics(self):
        """Calculates and stores metrics for the current step."""
        incision = np.max(self.z0 - self.z)
        self.metrics['max_incision'].append(incision)
        
        active_flow = self.h > self.cfg.min_depth
        if np.any(active_flow):
            vel_mag = np.sqrt(self.u**2 + self.v**2)
            mean_vel = np.mean(vel_mag[active_flow])
            self.metrics['mean_velocity'].append(mean_vel)
        else:
            self.metrics['mean_velocity'].append(0)

        if self.step % 100 == 0:
            logger.info(
                f"Step {self.step}, Time {self.time:.1f}s, dt={self.dt:.4f}, "
                f"Max depth: {np.max(self.h):.3f}m, Max incision: {incision:.4f}m"
            )

    def run(self):
        """Runs the entire simulation."""
        logger.info("Starting simulation...")
        self.initialize()
        create_visualization(self, save=True)
        
        while self.time < self.cfg.total_time:
            self.simulate_step()
            
            if self.step % self.cfg.output_interval == 0:
                create_visualization(self, save=True)
        
        create_visualization(self, save=True)
        self.save_final_metrics()
        logger.info(f"Simulation completed in {self.step} steps.")

    def save_final_metrics(self):
        """Saves final summary metrics to a text file."""
        metrics_file = self.output_dir / "summary_metrics.txt"
        with open(metrics_file, 'w') as f:
            f.write(f"Simulation: {self.cfg.experiment_name}\n")
            f.write(f"Total simulated time: {self.time:.1f}s\n")
            f.write(f"Total steps: {self.step}\n")
            f.write(f"Final max incision: {self.metrics['max_incision'][-1]:.4f}m\n")
            f.write(f"Average mean velocity: {np.mean(self.metrics['mean_velocity']):.4f}m/s\n")
            f.write(f"Final max depth: {np.max(self.h):.4f}m\n")