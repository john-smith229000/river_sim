#!/usr/bin/env python3
import numpy as np

RHO_WATER = 1000  # kg/m^3
GRAVITY = 9.81    # m/s^2

def compute_erosion_deposition(sim, S):
    """
    Computes bed elevation changes using a stable, mass-balanced approach
    where deposition is correctly suppressed by high flow energy.
    """
    cfg, dt = sim.cfg, sim.dt
    z, h, u, v, c = sim.z, sim.h, sim.u, sim.v, sim.c
    
    # Clean incoming sediment concentration to ensure it is physically possible.
    c = np.nan_to_num(np.clip(c, 0.0, 0.5))
    
    # --- FINAL, PHYSICS-BASED EROSION/DEPOSITION ENGINE ---
    
    # 1. Calculate the bed shear stress, which represents the energy of the flow.
    tau = RHO_WATER * GRAVITY * h * S
    
    # 2. EROSION (Entrainment): Occurs only where flow energy exceeds a critical threshold.
    excess_shear = np.maximum(0, tau - cfg.critical_shear)
    E = cfg.erosion_rate * excess_shear

    # 3. DEPOSITION: Occurs only where flow energy is *below* the critical threshold.
    # This is the crucial fix. We define a "deposition factor" that is 1
    # for calm water (tau=0) and 0 for energetic water (tau >= critical_shear).
    # This prevents deposition in fast-flowing parts of the channel.
    deposition_factor = np.maximum(0, 1.0 - tau / (cfg.critical_shear + 1e-9))
    D = cfg.deposition_rate * c * deposition_factor
    
    # 4. NET BED CHANGE: The difference between deposition and erosion.
    # We revert to a simple formulation without porosity for maximum stability first.
    dz_dt = D - E
    
    # Limit the maximum rate of change for numerical stability.
    max_allowable_rate = 0.05 * cfg.dx / (dt + 1e-9) # Increased limit slightly
    dz_dt = np.clip(dz_dt, -max_allowable_rate, max_allowable_rate)
    
    # Apply the change to the bed.
    active_cells = h > cfg.min_depth
    dz = dz_dt * dt
    z_new = z.copy()
    z_new[active_cells] += dz[active_cells]

    # 5. CONSERVATION OF MASS: Update sediment concentration based on net bed change.
    c_new = c.copy()
    c_new[active_cells] -= dz_dt[active_cells] * dt / (h[active_cells] + 1e-6)
    c_new = np.nan_to_num(np.clip(c_new, 0.0, 0.5))

    # Advect the updated sediment concentration downstream.
    c_new = _advect_sediment(c_new, u, v, cfg, dt)
    
    # Apply the inlet boundary condition for sediment.
    inlet_start = cfg.height // 2 - cfg.inlet_width // 2
    inlet_end = inlet_start + cfg.inlet_width
    c_new[0, inlet_start:inlet_end] = cfg.sediment_supply_concentration
    
    # Apply localized bank slumping (diffusion).
    if cfg.bed_diffusion_coeff > 0:
        z_padded = np.pad(z_new, 1, mode='edge')
        z_neighbors_avg = (z_padded[:-2, 1:-1] + z_padded[2:, 1:-1] +
                           z_padded[1:-1, :-2] + z_padded[1:-1, 2:]) / 4
        dz_diff = z_neighbors_avg - z_new
        z_new[active_cells] += cfg.bed_diffusion_coeff * dz_diff[active_cells]

    return z_new, c_new

def _advect_sediment(c, u, v, cfg, dt):
    """Transports sediment using a simple upwind advection scheme."""
    c_adv = c.copy()
    
    flux_x = u * c * dt / cfg.dx
    c_adv[1:, :] -= np.maximum(0, flux_x[:-1, :])
    c_adv[:-1, :] += np.minimum(0, flux_x[1:, :])

    flux_y = v * c * dt / cfg.dy
    c_adv[:, 1:] -= np.maximum(0, flux_y[:, :-1])
    c_adv[:, :-1] += np.minimum(0, flux_y[:, 1:])
    
    return np.nan_to_num(np.clip(c_adv, 0.0, 0.5))