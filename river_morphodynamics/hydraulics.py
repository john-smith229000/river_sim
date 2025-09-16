#!/usr/bin/env python3
import numpy as np

def compute_flow(sim):
    """
    Computes one step of the hydraulic simulation.
    """
    cfg = sim.cfg
    h = sim.h
    z = sim.z
    dt = sim.dt

    # Water surface elevation
    eta = z + h
    
    # Calculate water surface gradients (slope)
    eta_dx, eta_dy = np.gradient(eta, cfg.dx, cfg.dy)
    S = np.sqrt(eta_dx**2 + eta_dy**2)
    S = np.maximum(S, 1e-6)  # Avoid division by zero

    # Dampen instability by capping the maximum possible slope.
    S = np.minimum(S, 0.5)

    # Calculate velocity using Manning's equation
    wet_cells = h > cfg.min_depth
    vel = np.zeros_like(h)
    
    h_for_vel = np.maximum(0.0, h)
    
    vel[wet_cells] = (h_for_vel[wet_cells]**(2/3) * np.sqrt(S[wet_cells])) / cfg.manning_n

    # CFL condition for stability
    max_vel_actual = np.max(vel)
    if max_vel_actual > 1e-6:
        cfl_dt = 0.2 * min(cfg.dx, cfg.dy) / max_vel_actual
        dt = min(cfg.base_dt, cfl_dt)
    else: 
        dt = cfg.base_dt

    # Decompose velocity into u and v components
    u = -vel * eta_dx / S
    v = -vel * eta_dy / S
    
    # Calculate water flux
    qx = h * u
    qy = h * v
    
    # Divergence of flux
    dqx_dx, dqy_dy = np.gradient(qx, cfg.dx, axis=0), np.gradient(qy, cfg.dy, axis=1)
    
    # Update water depth
    h_new = h - (dqx_dx + dqy_dy) * dt
    
    # Apply diffusion to stabilize the scheme
    diffusion_coeff = 0.01
    h_padded = np.pad(h_new, 1, mode='edge')
    h_neighbors_avg = (h_padded[:-2, 1:-1] + h_padded[2:, 1:-1] + 
                       h_padded[1:-1, :-2] + h_padded[1:-1, 2:]) / 4
    h_new = (1 - diffusion_coeff) * h_new + diffusion_coeff * h_neighbors_avg
    
    # Apply boundary conditions
    # --- UPDATE THIS LINE: The 'z' argument is no longer needed ---
    h_new = _apply_boundaries(h_new, cfg)
    
    # Ensure minimum water depth
    h_new = np.maximum(h_new, cfg.min_depth)
    
    return h_new, u, v, S, dt

def _apply_boundaries(h, cfg):
    """Applies inlet, outlet, and wall boundary conditions."""
    # --- REVISED INLET CONDITION ---
    # This new condition sets a constant "normal depth" at the inlet. This is the
    # depth required to sustain the target discharge, providing a stable source.
    # It prevents both water pile-ups and the river running dry.
    
    # Calculate the target normal depth based on Manning's equation
    inlet_width_m = cfg.inlet_width * cfg.dy
    slope_sqrt = np.sqrt(cfg.initial_slope)
    # This formula solves for the depth 'h' that produces the target discharge 'Q'
    h_normal = (cfg.discharge * cfg.manning_n / (inlet_width_m * slope_sqrt))**0.6
    
    # Set the inlet cells to this target depth
    inlet_start = cfg.height // 2 - cfg.inlet_width // 2
    inlet_end = inlet_start + cfg.inlet_width
    h[0, inlet_start:inlet_end] = h_normal
    
    # Outlet: Zero-gradient condition
    h[-1, :] = h[-2, :]
    
    # Walls: No-flow condition
    h[:, 0] = h[:, 1]
    h[:, -1] = h[:, -2]
    
    return h