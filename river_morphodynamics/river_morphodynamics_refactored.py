#!/usr/bin/env python3
"""
Refactored River Morphodynamics Simulator
Key improvements:
1. Proper discharge-based water input (volumetric flow rate)
2. Better drainage and water routing
3. Channel incision with proper feedback
4. More readable code structure
5. Optimized visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
from pathlib import Path
import argparse
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SimulationConfig:
    """Configuration parameters for the simulation"""
    # Domain parameters
    width: int = 100
    height: int = 50
    dx: float = 10.0
    dy: float = 10.0
    
    # Time parameters
    dt: float = 0.1
    total_time: float = 500.0
    output_interval: int = 100
    
    # Physical parameters
    discharge: float = 50.0  # m³/s - actual volumetric flow rate
    slope: float = 0.005  # Keep as 'slope' for compatibility
    roughness: float = 0.03  # Manning's n
    gravity: float = 9.81
    
    # Sediment parameters
    erosion_coefficient: float = 0.001  # Scaled for incision
    deposition_coefficient: float = 0.0005
    transport_capacity_coeff: float = 0.1  # Match your config naming
    
    # Initial terrain parameters
    valley_depth: float = 3.0
    terrain_roughness: float = 0.2  # Match your config
    noise_scale: float = 0.2  # Match your config
    
    # Boundary conditions
    upstream_discharge: float = 50.0  # Match your config
    upstream_source_y: int = 25  # Match your config
    upstream_source_width: int = 3  # Match your config
    sediment_supply_concentration: float = 0.01
    
    # Output parameters
    output_dir: str = "output"
    experiment_name: str = "river_sim"
    
    # Optional parameters not in your config (with defaults)
    min_flow_depth: float = 0.05

class RiverMorphodynamics:
    """Refactored river morphodynamics simulator with proper physics"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.time = 0.0
        self.step_count = 0
        
        # Initialize grids
        self.bed_elevation = None
        self.water_depth = None
        self.water_surface = None
        self.velocity_x = None
        self.velocity_y = None
        self.sediment_concentration = None
        
        # Metrics tracking
        self.channel_depth_history = []
        self.max_incision_history = []
        
        # Output setup
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def initialize_terrain(self):
        """Create initial terrain with valley and natural variations"""
        w, h = self.config.width, self.config.height
        
        # Create coordinate grids
        x = np.arange(w) * self.config.dx
        y = np.arange(h) * self.config.dy
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Base slope (high upstream, low downstream)
        base_elevation = np.zeros((w, h))
        for i in range(w):
            base_elevation[i, :] = (w - i) * self.config.dx * self.config.slope
        
        # Add valley shape with slight meandering
        valley_center = h * self.config.dy / 2
        for i in range(w):
            # Slight sinusoidal meandering
            meander = 0.1 * h * self.config.dy * np.sin(2 * np.pi * i / (w * 0.5))
            y_offset = np.abs(y - valley_center - meander)
            
            # Parabolic valley cross-section
            valley_factor = 1 - (y_offset / (h * self.config.dy * 0.4))**2
            valley_factor = np.maximum(0, valley_factor)
            # Fix: ensure valley_factor is 1D array matching the width dimension
            base_elevation[i, :] = base_elevation[i, :] - self.config.valley_depth * valley_factor
        
        # Add random terrain variations
        noise = np.random.normal(0, self.config.noise_scale, (w, h))
        
        # Smooth the noise slightly
        from scipy.ndimage import gaussian_filter
        noise = gaussian_filter(noise, sigma=1.0)
        
        self.bed_elevation = base_elevation + noise
        
        # Initialize water depth (thin film initially)
        self.water_depth = np.full((w, h), 0.01)
        
        # Initialize velocities
        self.velocity_x = np.zeros((w, h))
        self.velocity_y = np.zeros((w, h))
        
        # Initialize sediment
        self.sediment_concentration = np.full((w, h), 0.001)
        
        # Store initial state for comparison
        self.initial_bed = self.bed_elevation.copy()
        
        logger.info(f"Initialized terrain: {w}x{h} cells")
        
    def compute_flow(self):
        """Compute water flow using simplified shallow water approach with stability"""
        # Water surface elevation
        self.water_surface = self.bed_elevation + self.water_depth
        
        # Compute surface gradients with safety checks
        dh_dx = np.zeros_like(self.water_surface)
        dh_dy = np.zeros_like(self.water_surface)
        
        # Use simple finite differences for more stability
        dh_dx[1:-1, :] = (self.water_surface[2:, :] - self.water_surface[:-2, :]) / (2 * self.config.dx)
        dh_dy[:, 1:-1] = (self.water_surface[:, 2:] - self.water_surface[:, :-2]) / (2 * self.config.dy)
        
        # Handle boundaries
        dh_dx[0, :] = (self.water_surface[1, :] - self.water_surface[0, :]) / self.config.dx
        dh_dx[-1, :] = (self.water_surface[-1, :] - self.water_surface[-2, :]) / self.config.dx
        dh_dy[:, 0] = (self.water_surface[:, 1] - self.water_surface[:, 0]) / self.config.dy
        dh_dy[:, -1] = (self.water_surface[:, -1] - self.water_surface[:, -2]) / self.config.dy
        
        # Compute slope magnitude
        slope_magnitude = np.sqrt(dh_dx**2 + dh_dy**2)
        slope_magnitude = np.maximum(slope_magnitude, 1e-6)  # Avoid division by zero
        
        # Only compute velocity where there's enough water
        wet_mask = self.water_depth > self.config.min_flow_depth
        
        # Hydraulic radius approximated as depth for wide channels
        hydraulic_radius = np.maximum(self.water_depth, 0.001)
        
        # Velocity magnitude using Manning's equation
        velocity_magnitude = np.zeros_like(self.water_depth)
        velocity_magnitude[wet_mask] = (1.0 / self.config.roughness) * \
                                      (hydraulic_radius[wet_mask]**(2/3)) * \
                                      np.sqrt(slope_magnitude[wet_mask])
        
        # Limit maximum velocity for stability (Courant condition)
        max_velocity = min(self.config.dx, self.config.dy) / (2 * self.config.dt)
        velocity_magnitude = np.minimum(velocity_magnitude, max_velocity)
        
        # Velocity components
        self.velocity_x = np.zeros_like(self.water_depth)
        self.velocity_y = np.zeros_like(self.water_depth)
        
        self.velocity_x[wet_mask] = -velocity_magnitude[wet_mask] * dh_dx[wet_mask] / slope_magnitude[wet_mask]
        self.velocity_y[wet_mask] = -velocity_magnitude[wet_mask] * dh_dy[wet_mask] / slope_magnitude[wet_mask]
        
    def update_water(self):
        """Update water depth using stable finite volume method"""
        # Save old depth for stability check
        old_depth = self.water_depth.copy()
        
        # Compute fluxes with upwind scheme for stability
        # X-direction fluxes
        flux_x = np.zeros_like(self.water_depth)
        # Positive velocity (flow to the right)
        flux_x[:-1, :] = np.where(self.velocity_x[:-1, :] > 0,
                                  self.velocity_x[:-1, :] * self.water_depth[:-1, :],
                                  self.velocity_x[:-1, :] * self.water_depth[1:, :])
        
        # Y-direction fluxes  
        flux_y = np.zeros_like(self.water_depth)
        # Positive velocity (flow down)
        flux_y[:, :-1] = np.where(self.velocity_y[:, :-1] > 0,
                                  self.velocity_y[:, :-1] * self.water_depth[:, :-1],
                                  self.velocity_y[:, :-1] * self.water_depth[:, 1:])
        
        # Update using finite volume method
        # Limit flux to prevent negative depths
        max_flux_x = 0.25 * self.water_depth / self.config.dt * self.config.dx
        max_flux_y = 0.25 * self.water_depth / self.config.dt * self.config.dy
        
        flux_x = np.sign(flux_x) * np.minimum(np.abs(flux_x), max_flux_x)
        flux_y = np.sign(flux_y) * np.minimum(np.abs(flux_y), max_flux_y)
        
        # Apply fluxes
        self.water_depth[1:, :] += (flux_x[:-1, :] - flux_x[1:, :]) * self.config.dt / self.config.dx
        self.water_depth[:, 1:] += (flux_y[:, :-1] - flux_y[:, 1:]) * self.config.dt / self.config.dy
        
        # Apply boundary conditions
        self.apply_boundary_conditions()
        
        # Stability check - prevent explosive growth
        max_change = np.max(np.abs(self.water_depth - old_depth))
        if max_change > 1.0:  # If change is too large
            # Revert and use smaller effective timestep
            self.water_depth = old_depth * 0.9 + self.water_depth * 0.1
            logger.warning(f"Large depth change detected ({max_change:.2f}m), applying damping")
        
        # Ensure minimum depth and cap maximum
        self.water_depth = np.clip(self.water_depth, 0.001, 10.0)  # Max 10m depth
        
    def apply_boundary_conditions(self):
        """Apply upstream discharge and downstream boundary conditions"""
        # Upstream boundary - add discharge as volumetric flow rate
        y_center = self.config.upstream_source_y
        y_width = self.config.upstream_source_width
        y_start = max(0, y_center - y_width // 2)
        y_end = min(self.config.height, y_start + y_width)
        
        # Use the upstream_discharge from config
        discharge = self.config.upstream_discharge
        
        # Calculate water volume to add per timestep
        discharge_per_cell = discharge / max(1, (y_end - y_start))
        volume_per_cell = discharge_per_cell * self.config.dt
        depth_to_add = volume_per_cell / (self.config.dx * self.config.dy)
        
        # Add water at upstream boundary
        self.water_depth[0, y_start:y_end] += depth_to_add
        
        # Add sediment supply at upstream using config value
        self.sediment_concentration[0, y_start:y_end] = self.config.sediment_supply_concentration
        
        # Downstream boundary - gradient boundary (allows outflow)
        self.water_depth[-1, :] = self.water_depth[-2, :]
        self.water_surface[-1, :] = self.water_surface[-2, :] - self.config.slope * self.config.dx
        
        # Lateral boundaries - no flux
        self.water_depth[:, 0] = self.water_depth[:, 1]
        self.water_depth[:, -1] = self.water_depth[:, -2]
        
    def compute_erosion_deposition(self):
        """Compute bed elevation changes from erosion and deposition"""
        # Shear stress (simplified)
        velocity_magnitude = np.sqrt(self.velocity_x**2 + self.velocity_y**2)
        shear_stress = self.water_depth * velocity_magnitude**2
        
        # Transport capacity
        transport_capacity = self.config.transport_capacity_coeff * shear_stress
        
        # Only erode/deposit where there's significant water
        active_mask = self.water_depth > 0.05
        
        # Erosion where capacity exceeds concentration
        erosion_potential = transport_capacity - self.sediment_concentration
        erosion = np.where(
            (erosion_potential > 0) & active_mask,
            self.config.erosion_coefficient * erosion_potential,
            0
        )
        
        # Deposition where concentration exceeds capacity
        deposition_potential = self.sediment_concentration - transport_capacity
        deposition = np.where(
            (deposition_potential > 0) & active_mask,
            self.config.deposition_coefficient * deposition_potential,
            0
        )
        
        # Update bed elevation
        bed_change = deposition - erosion
        
        # Limit bed change rate for stability
        max_change = 0.01  # Maximum change per timestep
        bed_change = np.clip(bed_change, -max_change, max_change)
        
        self.bed_elevation += bed_change * self.config.dt
        
        # Update sediment concentration
        self.sediment_concentration += (erosion - deposition) * self.config.dt
        self.sediment_concentration = np.maximum(self.sediment_concentration, 0)
        
        # Track incision
        incision = self.initial_bed - self.bed_elevation
        self.max_incision_history.append(np.max(incision))
        
        # Track channel depth
        channel_mask = self.water_depth > 0.1
        if np.any(channel_mask):
            mean_channel_depth = np.mean(self.water_depth[channel_mask])
            self.channel_depth_history.append(mean_channel_depth)
        
    def transport_sediment(self):
        """Transport sediment with the flow"""
        # Simple advection using upwind scheme
        # Transport in x-direction
        flux_x = self.sediment_concentration * self.velocity_x * self.config.dt / self.config.dx
        
        # Upwind differencing
        self.sediment_concentration[1:, :] -= np.maximum(flux_x[1:, :], 0)
        self.sediment_concentration[:-1, :] += np.maximum(flux_x[1:, :], 0)
        self.sediment_concentration[:-1, :] -= np.minimum(flux_x[:-1, :], 0)
        self.sediment_concentration[1:, :] += np.minimum(flux_x[:-1, :], 0)
        
        # Transport in y-direction
        flux_y = self.sediment_concentration * self.velocity_y * self.config.dt / self.config.dy
        
        self.sediment_concentration[:, 1:] -= np.maximum(flux_y[:, 1:], 0)
        self.sediment_concentration[:, :-1] += np.maximum(flux_y[:, 1:], 0)
        self.sediment_concentration[:, :-1] -= np.minimum(flux_y[:, :-1], 0)
        self.sediment_concentration[:, 1:] += np.minimum(flux_y[:, :-1], 0)
        
        # Ensure non-negative
        self.sediment_concentration = np.maximum(self.sediment_concentration, 0)
        
    def step(self):
        """Execute one time step of the simulation"""
        # Flow computation
        self.compute_flow()
        
        # Water transport
        self.update_water()
        
        # Sediment transport
        self.transport_sediment()
        
        # Morphodynamics
        self.compute_erosion_deposition()
        
        # Update time
        self.time += self.config.dt
        self.step_count += 1
        
        # Log progress
        if self.step_count % 100 == 0:
            max_depth = np.max(self.water_depth)
            mean_velocity = np.mean(np.sqrt(self.velocity_x**2 + self.velocity_y**2))
            max_incision = np.max(self.initial_bed - self.bed_elevation)
            logger.info(f"Step {self.step_count}, Time {self.time:.1f}s, "
                       f"Max depth: {max_depth:.3f}m, "
                       f"Mean velocity: {mean_velocity:.2f}m/s, "
                       f"Max incision: {max_incision:.3f}m")
    
    def visualize(self, save=True):
        """Create visualization of current state"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'River Morphodynamics - Time: {self.time:.1f}s', fontsize=14)
        
        # Bed elevation
        ax = axes[0, 0]
        im = ax.imshow(self.bed_elevation.T, cmap='terrain', aspect='auto', origin='lower')
        ax.set_title('Bed Elevation')
        ax.set_xlabel('Distance (cells)')
        ax.set_ylabel('Width (cells)')
        plt.colorbar(im, ax=ax, label='Elevation (m)')
        
        # Water depth
        ax = axes[0, 1]
        water_masked = np.ma.masked_where(self.water_depth < 0.05, self.water_depth)
        im = ax.imshow(water_masked.T, cmap='Blues', aspect='auto', origin='lower')
        ax.set_title('Water Depth')
        ax.set_xlabel('Distance (cells)')
        ax.set_ylabel('Width (cells)')
        plt.colorbar(im, ax=ax, label='Depth (m)')
        
        # Velocity magnitude
        ax = axes[0, 2]
        velocity_mag = np.sqrt(self.velocity_x**2 + self.velocity_y**2)
        velocity_masked = np.ma.masked_where(self.water_depth < 0.05, velocity_mag)
        im = ax.imshow(velocity_masked.T, cmap='Reds', aspect='auto', origin='lower')
        ax.set_title('Velocity Magnitude')
        ax.set_xlabel('Distance (cells)')
        ax.set_ylabel('Width (cells)')
        plt.colorbar(im, ax=ax, label='Velocity (m/s)')
        
        # Bed change (incision/deposition)
        ax = axes[1, 0]
        bed_change = self.bed_elevation - self.initial_bed
        im = ax.imshow(bed_change.T, cmap='RdBu_r', aspect='auto', origin='lower',
                      vmin=-1, vmax=1)
        ax.set_title('Bed Change (Blue=Incision, Red=Deposition)')
        ax.set_xlabel('Distance (cells)')
        ax.set_ylabel('Width (cells)')
        plt.colorbar(im, ax=ax, label='Change (m)')
        
        # Sediment concentration
        ax = axes[1, 1]
        sed_masked = np.ma.masked_where(self.sediment_concentration < 0.001, 
                                       self.sediment_concentration)
        im = ax.imshow(sed_masked.T, cmap='YlOrBr', aspect='auto', origin='lower')
        ax.set_title('Sediment Concentration')
        ax.set_xlabel('Distance (cells)')
        ax.set_ylabel('Width (cells)')
        plt.colorbar(im, ax=ax, label='Concentration')
        
        # Cross-section at middle
        ax = axes[1, 2]
        mid_x = self.config.width // 2
        ax.plot(self.initial_bed[mid_x, :], 'k--', label='Initial bed', alpha=0.5)
        ax.plot(self.bed_elevation[mid_x, :], 'brown', label='Current bed', linewidth=2)
        water_surface = self.bed_elevation[mid_x, :] + self.water_depth[mid_x, :]
        ax.fill_between(range(self.config.height), 
                        self.bed_elevation[mid_x, :], 
                        water_surface,
                        color='cyan', alpha=0.5, label='Water')
        ax.set_title(f'Cross-section at x={mid_x}')
        ax.set_xlabel('Width (cells)')
        ax.set_ylabel('Elevation (m)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / f"viz_{self.step_count:06d}.png"
            plt.savefig(filepath, dpi=100, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def run(self):
        """Run the full simulation"""
        logger.info(f"Starting simulation: {self.config.experiment_name}")
        
        # Initialize
        self.initialize_terrain()
        
        # Initial visualization
        self.visualize()
        
        # Main simulation loop
        n_steps = int(self.config.total_time / self.config.dt)
        
        for i in range(n_steps):
            self.step()
            
            # Periodic output
            if self.step_count % self.config.output_interval == 0:
                self.visualize()
        
        # Final visualization
        self.visualize()
        
        # Save metrics
        self.save_metrics()
        
        logger.info(f"Simulation complete. Results in {self.output_dir}")
        
    def save_metrics(self):
        """Save simulation metrics to file"""
        metrics_file = self.output_dir / "metrics.txt"
        with open(metrics_file, 'w') as f:
            f.write(f"Simulation: {self.config.experiment_name}\n")
            f.write(f"Total time: {self.time:.1f}s\n")
            f.write(f"Max incision: {np.max(self.max_incision_history):.3f}m\n")
            if self.channel_depth_history:
                f.write(f"Mean channel depth: {np.mean(self.channel_depth_history):.3f}m\n")
            f.write(f"Final max water depth: {np.max(self.water_depth):.3f}m\n")
        
        logger.info(f"Metrics saved to {metrics_file}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='River Morphodynamics Simulator')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--quick', action='store_true', help='Quick test run')
    args = parser.parse_args()
    
    # Start with default configuration
    config = SimulationConfig()
    
    # Load configuration file if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Update config with values from file, only if they exist as attributes
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                logger.warning(f"Unknown config parameter: {key}")
    
    # Quick test mode
    if args.quick:
        config.total_time = 50.0
        config.output_interval = 50
        config.experiment_name = "quick_test"
    
    # Run simulation
    sim = RiverMorphodynamics(config)
    sim.run()

if __name__ == "__main__":
    main()