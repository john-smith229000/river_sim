#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

def create_visualization(sim, save=True, show=False):
    """
    Generates and saves a comprehensive visualization of the current simulation state.
    """
    cfg = sim.cfg
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    fig.suptitle(f'River Simulation: {cfg.experiment_name} - Time: {sim.time:.1f}s', fontsize=16)

    # --- Row 1: 2D Maps ---
    
    # Bed Elevation
    ax = axes[0, 0]
    im = ax.imshow(sim.z.T, cmap='terrain', aspect='auto', origin='lower')
    ax.set_title('Bed Elevation (z)')
    plt.colorbar(im, ax=ax, label='Elevation (m)')

    # Water Depth
    ax = axes[0, 1]
    h_masked = np.ma.masked_where(sim.h < cfg.min_depth, sim.h)
    im = ax.imshow(h_masked.T, cmap='Blues', aspect='auto', origin='lower', vmin=0)
    ax.set_title('Water Depth (h)')
    plt.colorbar(im, ax=ax, label='Depth (m)')

    # Velocity Magnitude
    ax = axes[0, 2]
    vel_mag = np.sqrt(sim.u**2 + sim.v**2)
    vel_masked = np.ma.masked_where(sim.h < cfg.min_depth, vel_mag)
    im = ax.imshow(vel_masked.T, cmap='Reds', aspect='auto', origin='lower', vmin=0)
    ax.set_title('Flow Velocity |v|')
    plt.colorbar(im, ax=ax, label='Velocity (m/s)')
    
    # --- Row 2: Analysis Plots ---

    # Incision / Deposition
    ax = axes[1, 0]
    incision = sim.z0 - sim.z
    vmax = max(0.1, np.max(np.abs(incision))) # Dynamic range
    im = ax.imshow(incision.T, cmap='RdBu', aspect='auto', origin='lower', vmin=-vmax, vmax=vmax)
    ax.set_title('Incision (Red) / Deposition (Blue)')
    plt.colorbar(im, ax=ax, label='Elevation Change (m)')
    
    # Cross-section at Mid-point
    ax = axes[1, 1]
    mid_x = cfg.width // 2
    ax.plot(sim.z0[mid_x, :], color='gray', linestyle='--', label='Initial Bed')
    ax.plot(sim.z[mid_x, :], 'brown', label='Current Bed')
    water_surface = sim.z[mid_x, :] + sim.h[mid_x, :]
    ax.fill_between(np.arange(cfg.height), sim.z[mid_x, :], water_surface, 
                   where=sim.h[mid_x, :] > cfg.min_depth, color='cyan', alpha=0.6, label='Water')
    ax.set_title(f'Cross-Section at x={mid_x}')
    ax.set_xlabel('Transverse Position (cells)')
    ax.set_ylabel('Elevation (m)')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)

    # Metrics over Time
    ax = axes[1, 2]
    if len(sim.metrics['max_incision']) > 1:
        time_steps = np.linspace(0, sim.time, len(sim.metrics['max_incision']))
        ax.plot(time_steps, sim.metrics['max_incision'], 'r-', label='Max Incision')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Max Incision (m)', color='r')
        ax.tick_params(axis='y', labelcolor='r')
        
        ax2 = ax.twinx()
        ax2.plot(time_steps, sim.metrics['mean_velocity'], 'b-', label='Mean Velocity')
        ax2.set_ylabel('Mean Velocity (m/s)', color='b')
        ax2.tick_params(axis='y', labelcolor='b')
        
    ax.set_title('Metrics Evolution')
    ax.grid(True, linestyle=':', alpha=0.6)

    for ax_row in axes:
        for ax_col in ax_row:
            ax_col.set_xlabel('Downstream Position (cells)')
            ax_col.set_ylabel('Transverse Position (cells)')

    if save:
        filepath = sim.output_dir / f"frame_{sim.step:06d}.png"
        plt.savefig(filepath, dpi=120)
    
    if show:
        plt.show()
    
    plt.close(fig)