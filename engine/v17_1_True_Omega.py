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

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import jax
import jax.numpy as jnp
import numpy as np
import time
import warnings
import csv

warnings.filterwarnings("ignore")

# --- 1. HARDWARE & BASE GRID ALLOCATION ---
print(f"System Online: {jax.devices()[0].device_kind.upper()} Calibrated.")

MAX_BODIES = 500 
BASE_GRID_MIN, BASE_GRID_MAX, BASE_GRID_RES = -200.0, 200.0, 15.0
GRID_BINS = int((BASE_GRID_MAX - BASE_GRID_MIN) / BASE_GRID_RES)
TOTAL_CELLS = GRID_BINS ** 3 
BASE_THETA_THRESHOLD = 30.0 

masses = np.zeros(MAX_BODIES, dtype=float)
pos = np.zeros((MAX_BODIES, 3), dtype=float)
vel = np.zeros((MAX_BODIES, 3), dtype=float)
stored_microstates = np.zeros(MAX_BODIES, dtype=int)   
entanglement_map = np.full(MAX_BODIES, -1, dtype=int) 

dark_matter_nodes = np.array([
    [120.0, 120.0, 20.0],  [-120.0, 120.0, -20.0], 
    [120.0, -120.0, -20.0], [-120.0, -120.0, 20.0],
    [0.0, 180.0, 50.0],    [0.0, -180.0, -50.0], 
    [180.0, 0.0, -50.0],   [-180.0, 0.0, 50.0]
], dtype=float)

NODE_RADIUS, TARGET_DENSITY = 60.0, 4
dark_matter_pool = 1000.0 
INITIAL_UNIVERSE_BITS = dark_matter_pool

# --- 2. PLANCK PHYSICS CONSTANTS (G=c=1) ---
G, C = 1.0, 1.0
GW_TIMESTEP = 0.001     
HAWKING_COEFF = 15000.0 
PLANCK_MASS_KG = 2.176470e-8

# NEW: String framework constants
STRING_TENSION = 0.5 # Scales the physical volume of the Fuzzball surface

# --- DYNAMIC COSMOLOGY STATE ---
scale_factor = 1.0  

# --- 3. JAX STRETCHY SPATIAL HASHING ---
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

print("\n==========================================")
print("  INITIATING OMEGA-FLRW (RELATIVISTIC) RUN ")
print("==========================================")

telemetry_data = [] 

for epoch in range(1, 101):
    events = []
    
    # 0. UNIFIED FRIEDMANN EQUATION
    universe_radius = BASE_GRID_MAX * scale_factor
    universe_volume = (4.0 / 3.0) * np.pi * (universe_radius ** 3)
    p_bulk, s_surf = int(np.sum(masses)), int(np.sum(stored_microstates))
    total_inf = p_bulk + s_surf + int(dark_matter_pool)
    
    matter_density = (p_bulk + s_surf) / universe_volume
    vacuum_density = int(dark_matter_pool) / universe_volume
    dynamic_lambda = 8.0 * np.pi * G * vacuum_density
    
    H = np.sqrt((8.0 * np.pi * G / 3.0) * matter_density + (dynamic_lambda / 3.0))
    scale_factor += H 
    
    # 1. RECOMBINATION
    active = [i for i in range(MAX_BODIES) if (masses[i] + stored_microstates[i]) > 0]
    for node_idx, node_pos in enumerate(dark_matter_nodes):
        local_count = len([idx for idx in active if np.linalg.norm(pos[idx] - node_pos) < NODE_RADIUS])
        if local_count < TARGET_DENSITY and dark_matter_pool >= 2:
            slots = [i for i in range(MAX_BODIES) if (masses[i] + stored_microstates[i]) == 0][:2]
            if len(slots) == 2:
                for s in slots:
                    m_val = float(np.random.choice([1.0, 2.0, 3.0]))
                    masses[s] = m_val
                    pos[s] = node_pos + np.random.uniform(-15, 15, size=3)
                    vel[s] = np.random.uniform(-0.1, 0.1, size=3)
                    dark_matter_pool -= m_val
                entanglement_map[slots], entanglement_map[slots] = slots, slots
                events.append(f"*** [NODE {node_idx}] Recombined Matter.")

    # --- HIERARCHICAL RELATIVISTIC KINEMATICS ---
    eff_mass_array = masses + stored_microstates
    jax_pos = jnp.array(pos)
    jax_eff_mass = jnp.array(eff_mass_array)
    
    curr_min = BASE_GRID_MIN * scale_factor
    curr_max = BASE_GRID_MAX * scale_factor
    curr_res = BASE_GRID_RES * scale_factor
    curr_theta = BASE_THETA_THRESHOLD * scale_factor
    
    cell_masses_jnp, cell_com_jnp = compute_macro_nodes(jax_pos, jax_eff_mass, curr_min, curr_max, curr_res)
    cell_masses = np.array(cell_masses_jnp)
    cell_com = np.array(cell_com_jnp)

    alpha_sum = 0.0 # Tracking time dilation

    for i in active:
        m_eff_i = eff_mass_array[i]
        force = np.zeros(3)
        v_mag = np.linalg.norm(vel[i])
        gamma = 1.0 / np.sqrt(1.0 - (min(v_mag, 0.99)**2))
        
        # Calculate local scalar potential for Time Dilation
        local_phi = 0.0 
        
        # FAR-FIELD
        for c in range(TOTAL_CELLS):
            if cell_masses[c] <= 0: continue
            diff = cell_com[c] - pos[i]
            dist = np.linalg.norm(diff)
            if dist > curr_theta:
                force += (diff / dist) * (G * m_eff_i * cell_masses[c] / (dist**2))
                local_phi -= (G * cell_masses[c]) / dist
        
        # NEAR-FIELD
        for j in active:
            if i == j: continue
            m_eff_j = eff_mass_array[j]
            diff = pos[j] - pos[i]
            dist = np.linalg.norm(diff) + 0.1
            
            if dist <= curr_theta:
                # BENDING SPACE: Fuzzball String Surface Area Expansion
                # The radius is a function of its stored microstates (String Tension)
                r_fuzzball_i = 2.0 * m_eff_i + (STRING_TENSION * np.sqrt(stored_microstates[i]))
                r_fuzzball_j = 2.0 * m_eff_j + (STRING_TENSION * np.sqrt(stored_microstates[j]))
                r_s = r_fuzzball_i + r_fuzzball_j
                
                if dist < r_s:
                    stored_microstates[i] += int(m_eff_j)
                    masses[j], stored_microstates[j], entanglement_map[j] = 0.0, 0, -1
                    eff_mass_array[j] = 0.0               
                    eff_mass_array[i] += int(m_eff_j)     
                    m_eff_i = eff_mass_array[i]           
                    events.append("!!! [HOLOGRAPHIC SHIFT] Mass converted.")
                else:
                    local_phi -= (G * m_eff_j) / dist
                    
                    # BENDING SPACE: Relativistic Effective Potential (Curvature)
                    rel_v = vel[i] - vel[j]
                    h_vec = np.cross(diff, rel_v)
                    h2 = np.dot(h_vec, h_vec)
                    a_gr = (3.0 * G * m_eff_j * h2) / (dist**4)
                    a_gr = min(a_gr, 1.0) # Cap numeric singularity
                    
                    # Gravitational Wave Dissipation
                    gw_power = (32.0 / 5.0) * (m_eff_i**2 * m_eff_j**2 * (m_eff_i + m_eff_j)) / (dist**5)
                    gw_loss_factor = min(0.1, gw_power * GW_TIMESTEP) 
                    vel[i] *= (1.0 - gw_loss_factor)
                    
                    # Apply total warped force (Newtonian + Curvature)
                    force += (diff / dist) * (G * m_eff_i * m_eff_j / (dist**2) + m_eff_i * a_gr)

        # BENDING TIME: Gravitational Time Dilation (Lapse Function alpha)
        # alpha = sqrt(1 + 2Phi/c^2). Approaching zero freezes time for the particle.
        alpha = np.sqrt(max(0.01, 1.0 + 2.0 * local_phi))
        alpha_sum += alpha
        
        # 1. Update Peculiar Velocity (Subject to Time Dilation)
        vel[i] = (vel[i] + (force * alpha) / (m_eff_i * gamma))
        v_limit = np.linalg.norm(vel[i])
        if v_limit > 0.99: vel[i] = (vel[i] / v_limit) * 0.99
        
        # 2. Update Position (Peculiar Motion is dilated; FTL Metric Expansion is NOT)
        pos[i] += (vel[i] * alpha) + (H * pos[i])
        
    dark_matter_nodes += H * dark_matter_nodes

    # 3. THERMODYNAMICS
    for i in active:
        m_eff_i = masses[i] + stored_microstates[i]
        if m_eff_i >= 20.0:
            if stored_microstates[i] > 0:
                capacity = 3.14159 * (2 * m_eff_i)**2
                while capacity < stored_microstates[i] and stored_microstates[i] > 0:
                    stored_microstates[i] -= 1
                    dark_matter_pool += 1
            elif masses[i] > 0:
                evap_calculation = HAWKING_COEFF / (m_eff_i**2)
                temp_rate = int(evap_calculation)
                if temp_rate < 1: temp_rate = 1 
                masses[i] -= temp_rate
                dark_matter_pool += temp_rate 
                if masses[i] < 15.0:
                    dark_matter_pool += int(masses[i])
                    masses[i] = 0

    print(f"\n--- EPOCH {epoch} ---")
    epoch_events = list(set(events))[:2] if events else ["None"]
    if events:
        for e in epoch_events: print(e)
        
    avg_lapse = alpha_sum / len(active) if active else 1.0
    print(f"Bulk: {p_bulk} | Horizon: {s_surf} | Vacuum: {int(dark_matter_pool)}")
    print(f"Avg Time Dilation (Lapse α): {avg_lapse:.4f} (1.0 = normal, 0.0 = frozen)")
    print(f"Expansion Rate (H): {H:.5f} | Scale Factor (a): {scale_factor:.2f}")
    print(f"Integrity: {total_inf}/{int(INITIAL_UNIVERSE_BITS)}")
    
    telemetry_data.append([
        epoch, p_bulk, s_surf, int(dark_matter_pool), total_inf, 
        f"{H:.5f}", f"{scale_factor:.2f}", f"{avg_lapse:.4f}", 
        " | ".join(epoch_events)
    ])
    time.sleep(0.3)

def export_telemetry_to_csv(filename="telemetry_flrw_gr.csv"):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Epoch", "Bulk_Bits", "Horizon_Bits", "Vacuum_Bits", 
            "Total_Integrity", "Hubble_H", "Scale_Factor_a", 
            "Avg_Lapse_Alpha", "Notable_Events"
        ])
        writer.writerows(telemetry_data)
    print(f"\n>>> Unified FLRW+GR Telemetry successfully exported.")

export_telemetry_to_csv()
