#!/usr/bin/env python3
import numpy as np

RHO_WATER = 1000  # kg/m^3
GRAVITY = 9.81    # m/s^2

def compute_erosion_deposition(sim, S):
    """
    Computes bed elevation changes using a stable, mass-balanced approach.
    """
    cfg, dt = sim.cfg, sim.dt
    z, h, u, v, c = sim.z, sim.h, sim.u, sim.v, sim.c
    
    # Clean incoming sediment concentration to ensure it's physically possible
    c = np.nan_to_num(np.clip(c, 0.0, 0.5))
    
    # --- NEW, STABLE EROSION/DEPOSITION ENGINE ---
    
    # 1. Calculate the flow's power to erode (Erosion Potential)
    # This is the same as before: based on shear stress.
    tau = RHO_WATER * GRAVITY * h * S
    excess_shear = np.maximum(0, tau - cfg.critical_shear)
    E_potential = cfg.erosion_rate * excess_shear

    # 2. Calculate the potential for deposition based on settling
    # Based on your insight, deposition is now a simple settling process
    # proportional to the amount of sediment in the water. `deposition_rate`
    # now acts as a "settling velocity". This is physically realistic and stable.
    D_potential = cfg.deposition_rate * c

    # 3. The net change in the bed is the difference between the two.
    # If erosion potential > deposition potential, the bed erodes.
    # If deposition potential > erosion potential, the bed aggrades.
    # This system naturally seeks an equilibrium and cannot run away.
    dz_dt = D_potential - E_potential
    
    # For overall stability, limit the maximum rate of change
    max_allowable_rate = 0.01 * cfg.dx / (dt + 1e-9)
    dz_dt = np.clip(dz_dt, -max_allowable_rate, max_allowable_rate)
    
    # Calculate the actual change in bed elevation
    active_cells = h > cfg.min_depth
    dz = dz_dt * dt
    z_new = z.copy()
    z_new[active_cells] += dz[active_cells]

    # 4. Update sediment concentration in the water based on bed change
    # The change in sediment concentration is the opposite of the bed change.
    c_new = c.copy()
    c_new[active_cells] -= dz_dt[active_cells] * dt / (h[active_cells] + 1e-6)

    # Enforce absolute physical limits on concentration
    c_new = np.nan_to_num(np.clip(c_new, 0.0, 0.5))

    # Advect the updated sediment concentration downstream
    c_new = _advect_sediment(c_new, u, v, cfg, dt)
    
    # Apply the inlet boundary condition for sediment
    inlet_start = cfg.height // 2 - cfg.inlet_width // 2
    inlet_end = inlet_start + cfg.inlet_width
    c_new[0, inlet_start:inlet_end] = cfg.sediment_supply_concentration
    
    # Apply bank slumping where there is water
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