# Copyright 2026 Rupayan Bandyopadhyay
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ==============================================================================
# Title: The Ultimate Cyclic Manifold (v13.0)
# Framework: Loop Quantum Cosmology & Dynamic Quintessence
# ==============================================================================

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import jax
import jax.numpy as jnp
import numpy as np
import warnings

warnings.filterwarnings("ignore")

print(f"System Online: CPU Calibrated. (v13.0 Ultimate Cyclic Engine Active)")

MAX_BODIES = 500 
BASE_GRID_SIZE = 400.0
GRID_BINS = 26 
TOTAL_CELLS = GRID_BINS ** 3 
BASE_THETA_THRESHOLD = 30.0 

masses = np.zeros(MAX_BODIES, dtype=float)
pos = np.zeros((MAX_BODIES, 3), dtype=float)
vel = np.zeros((MAX_BODIES, 3), dtype=float)
stored_microstates = np.zeros(MAX_BODIES, dtype=float) 
excitation_clock = np.zeros(MAX_BODIES, dtype=float)

dark_matter_nodes = np.array([
    [120.0, 120.0, 20.0],  [-120.0, 120.0, -20.0], 
    [120.0, -120.0, -20.0], [-120.0, -120.0, 20.0],
    [0.0, 180.0, 50.0],    [0.0, -180.0, -50.0], 
    [180.0, 0.0, -50.0],   [-180.0, 0.0, 50.0]
], dtype=float)

# --- EXACT PHYSICAL CONSTANTS (Natural Units) ---
G = 1.0 
C = 1.0 
H_BAR = 1.0 
K_B = 1.0 
GAMMA = 0.1 

RHO_PLANCK = 2.5 
V_PLANCK_STAR = 100.0 # The fixed quantum volume of the black hole core

INITIAL_UNIVERSE_BITS = 1000.0 
dark_matter_pool = INITIAL_UNIVERSE_BITS 

# --- JAX ARCHITECTURE ---
@jax.jit
def compute_macro_nodes(pos_array, effective_masses, grid_min, grid_max, grid_res):
    clipped_pos = jnp.clip(pos_array, grid_min, grid_max - 1e-5)
    grid_indices = jnp.floor((clipped_pos - grid_min) / grid_res).astype(jnp.int32)
    cell_hashes = (grid_indices[:, 2] * (GRID_BINS ** 2) + 
                   grid_indices[:, 1] * GRID_BINS + 
                   grid_indices[:, 0])
    cell_masses = jax.ops.segment_sum(effective_masses, cell_hashes, num_segments=TOTAL_CELLS)
    weighted_pos = pos_array * effective_masses[:, None]
    cell_weighted_pos = jax.ops.segment_sum(weighted_pos, cell_hashes, num_segments=TOTAL_CELLS)
    safe_cell_masses = jnp.where(cell_masses > 0, cell_masses, 1.0)
    cell_com = cell_weighted_pos / safe_cell_masses[:, None]
    return cell_masses, cell_com

@jax.jit
def lqg_thermodynamic_step(mass_ledger, clock_ledger, is_white_hole):
    is_fuzzball = (mass_ledger >= 10.0)
    safe_mass = jnp.maximum(mass_ledger, 1.0)
    
    thresholds = jnp.maximum(GAMMA * safe_mass, 1.0)
    next_clock = jnp.mod(clock_ledger + 1.0, thresholds)
    is_rollover = (next_clock < 1.0) & (clock_ledger + 1.0 >= thresholds)
    
    super_dump = jnp.floor(safe_mass * 0.5) 
    
    rad_quanta = jnp.where(is_white_hole, super_dump, jnp.where(is_rollover & is_fuzzball, 5.0, 0.0))
    
    next_clock = jnp.where(is_fuzzball, next_clock, 0.0)
    next_clock = jnp.where(is_white_hole, 0.0, next_clock)
    
    return rad_quanta, next_clock

print("\n=======================================================")
print("  INITIATING v13.0: THE ULTIMATE CYCLIC MANIFOLD       ")
print("=======================================================")

prev_grid_size = BASE_GRID_SIZE

for epoch in range(1, 501):
    
    # --- DYNAMIC QUINTESSENCE ---
    vac_ratio = dark_matter_pool / INITIAL_UNIVERSE_BITS
    contracted_volume = max((BASE_GRID_SIZE * (vac_ratio**(1.0/3.0)))**3, 1.0)
    vacuum_density = dark_matter_pool / contracted_volume
    KAPPA = 150.0 
    Lambda = KAPPA * vacuum_density
    
    curr_grid_size = BASE_GRID_SIZE * (vac_ratio + Lambda)**(1.0/3.0)
    curr_grid_size = max(curr_grid_size, 10.0)
    quantum_expansion_ratio = curr_grid_size / max(prev_grid_size, 10.0)
    
    curr_min, curr_max = -curr_grid_size / 2.0, curr_grid_size / 2.0
    curr_res = curr_grid_size / GRID_BINS
    curr_theta = BASE_THETA_THRESHOLD * (curr_grid_size / BASE_GRID_SIZE)
    
    active = [i for i in range(MAX_BODIES) if (masses[i] + stored_microstates[i]) > 0]
    
    # RECOMBINATION
    if len(active) < 150:
        available_slots = np.where((masses + stored_microstates) == 0)[0]
        for node_pos in dark_matter_nodes[:4]:
            if len(available_slots) >= 2 and dark_matter_pool >= 30.0:
                s_pair = available_slots[:2]
                masses[s_pair], dark_matter_pool = 15.0, dark_matter_pool - 30.0
                pos[s_pair] = node_pos + np.random.uniform(-10, 10, size=(2, 3))
                available_slots = available_slots[2:]
                for s in s_pair:
                    if int(s) not in active: active.append(int(s))

    # KINEMATICS & QUANTUM GRAVITY
    eff_mass_snapshot = masses + stored_microstates
    cell_masses, cell_com = compute_macro_nodes(jnp.array(pos), jnp.array(eff_mass_snapshot), curr_min, curr_max, curr_res)
    cell_masses, cell_com = np.array(cell_masses), np.array(cell_com)
    alpha_sum = 0.0 
    
    white_hole_flags = np.zeros(MAX_BODIES, dtype=bool)

    for i in active:
        m_eff_i = masses[i] + stored_microstates[i]
        if m_eff_i <= 0: continue
        
        # Internal Density Check (The Planck Star Core)
        rho_internal = m_eff_i / V_PLANCK_STAR
        if rho_internal >= RHO_PLANCK:
            white_hole_flags[i] = True

        force, local_phi = np.zeros(3), 0.0
        v_mag = np.linalg.norm(vel[i])
        gamma_rel = 1.0 / np.sqrt(1.0 - (min(v_mag, 0.99 * C)**2) / (C**2))
        
        m_diff = (cell_com - pos[i] - np.round((cell_com - pos[i]) / curr_grid_size) * curr_grid_size)
        m_dist = np.linalg.norm(m_diff, axis=1)

        matter_mask = (cell_masses > 0) & (m_dist > curr_theta)
        if np.any(matter_mask):
            m_m, m_d = cell_masses[matter_mask], m_dist[matter_mask]
            force += np.sum((m_diff[matter_mask] / m_d[:, None]) * (G * m_eff_i * m_m / (m_d**2))[:, None], axis=0)
            local_phi -= np.sum((G * m_m) / m_d)
        
        for j in active:
            if i == j: continue
            m_eff_j = masses[j] + stored_microstates[j]
            if m_eff_j <= 0: continue
            diff = (pos[j] - pos[i] - np.round((pos[j] - pos[i]) / curr_grid_size) * curr_grid_size)
            dist = np.linalg.norm(diff) + 0.1
            
            if dist <= curr_theta:
                r_s_pair = 2.0 * G * (m_eff_i + m_eff_j) / (C**2)
                if dist < r_s_pair:
                    stored_microstates[i] += (masses[j] + stored_microstates[j])
                    masses[j], stored_microstates[j], excitation_clock[j] = 0.0, 0.0, 0.0
                else:
                    rho_local = (m_eff_i + m_eff_j) / (dist**3)
                    lqc_modifier = 1.0 - (rho_local / RHO_PLANCK)
                    
                    if rho_local >= RHO_PLANCK:
                        white_hole_flags[i] = True
                    
                    f_gravity = ((G * m_eff_i * m_eff_j) / (dist**2)) * lqc_modifier
                    f_lambda = (1.0 / 3.0) * Lambda * (C**2) * dist
                    
                    force += (diff / dist) * (f_gravity - f_lambda)
                    local_phi -= ((G * m_eff_j) / dist) * lqc_modifier

        effective_phi = local_phi - (1.0 / 6.0) * Lambda * (C**2) * (curr_grid_size**2)
        alpha = np.sqrt(max(0.01, min(1.0, 1.0 + (2.0 * effective_phi) / (C**2))))
        
        alpha_sum += alpha
        vel[i] = (vel[i] + (force * alpha) / (m_eff_i * gamma_rel))
        pos[i] = ((pos[i] + (vel[i] * alpha)) * quantum_expansion_ratio - curr_min) % curr_grid_size + curr_min
        
    avg_alpha = alpha_sum / (len(active) or 1)
    dark_matter_nodes *= quantum_expansion_ratio
    prev_grid_size = curr_grid_size
        
    # WHITE HOLE THERMODYNAMICS
    rad_quanta, next_clk = lqg_thermodynamic_step(jnp.array(masses + stored_microstates), jnp.array(excitation_clock), jnp.array(white_hole_flags))
    rad_quanta, excitation_clock = np.array(rad_quanta), np.array(next_clk)
    
    for i in active:
        if rad_quanta[i] > 0:
            rad_amount = min(rad_quanta[i], stored_microstates[i] + masses[i])
            if stored_microstates[i] >= rad_amount: stored_microstates[i] -= rad_amount
            else: masses[i] -= rad_amount
            dark_matter_pool += rad_amount

    if epoch % 50 == 0 or epoch == 1:
        max_m = np.max(masses + stored_microstates) if len(active) > 0 else 0
        total_inf = np.sum(masses) + np.sum(stored_microstates) + dark_matter_pool
        status = "BOUNCE" if np.any(white_hole_flags) else "CLUMP"
        print(f"EPOCH {epoch:3d} | Vac: {dark_matter_pool:5.1f} | L: {curr_grid_size:6.1f} | α: {avg_alpha:.3f} | Max_F: {max_m:5.1f} | S: {status} | I: {total_inf/INITIAL_UNIVERSE_BITS:.6f}")

print("\n>>> v13.0 Engine Operational. The Universe Breathes.")
