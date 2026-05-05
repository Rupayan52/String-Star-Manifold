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
# Title: The Immutable Entropy Update (Sequential Fractal Protocol)
# ==============================================================================

import os
import sys
import subprocess
import random 
import warnings
import glob

# ==============================================================================
# --- 0. AUTO-DEPENDENCY INSTALLER & HARDWARE SAFETY ---
# ==============================================================================
try:
    import h5py
    import numpy as np
    from sklearn.cluster import DBSCAN
except ImportError:
    print("\033[1;33m[SYSTEM] Dependencies missing. Auto-installing h5py and scikit-learn...\033[0m")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "h5py", "scikit-learn", "numpy", "-q"])
    import h5py
    import numpy as np
    from sklearn.cluster import DBSCAN

# HARDWARE SAFETY: Prevent JAX from eating 100% of VRAM. Sequential mode relies on this.
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
os.environ['JAX_ENABLE_X64'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import jax
import jax.numpy as jnp
from jax import jit, config

config.update("jax_enable_x64", True)

# ==============================================================================
# --- 1. THE DUAL-COMPONENT PARAMETERS ---
# ==============================================================================
N_GRID = 128             
COMOVING_BOX = 400.0     
DT = 0.0015              

N_DM = 2000
MASS_DM = 120.0          
N_BARYON = 1000
MASS_BARYON = 5.2
SPH_SMOOTHING_H = 7.5    
GAMMA = 5.0 / 3.0        
COOLING_COEFF = 5.0e-3       
MIN_U = 1.0              

V0_QUINTESSENCE = 0.10       
PHI_INIT = 1.0           
PHI_DOT_INIT = 0.0       

RHO_THRESH_STAR = 12.0       
U_THRESH_STAR = 250.0        
STAR_FORM_EFF = 0.05         
SN_FEEDBACK_ENERGY = 1.2e4   
METAL_YIELD = 0.50           
COOLING_Z_MULTIPLIER = 25000.0 
RHO_THRESH_BH = 15.0         
Z_THRESH_BH = 0.0005         
CRITICAL_STRING_BITS = 1.2e4 

# --- COSMETIC & LOGGING INTERFACE ---
def print_header(gen_label, description):
    print("\033[95m" + "="*80 + "\033[0m")
    print(f"\033[1;36m        THE BANDYOPADHYAY CYCLIC MANIFOLD | {gen_label} \033[0m")
    print(f"\033[1;34m            {description}\033[0m")
    print("\033[95m" + "="*80 + "\033[0m")

def log_and_print_telemetry(epoch, a_t, max_rho, max_u, max_sf, max_zf, total_bh, total_bits, pinch_off, gen_label="GEN 1"):
    bar_width = 24
    filled = int(min(a_t / 10.0, 1.0) * bar_width)
    bar = "█" * filled + "░" * (bar_width - filled)

    print(f"\033[1mEPOCH {epoch:04d} [{gen_label}]\033[0m")
    print(f"  ├─ Scale Factor a(t)  : {a_t:.4f} x   |{bar}|")
    print(f"  ├─ Max Baryon Density : {max_rho:7.2f}")
    print(f"  ├─ Max Thermal Energy : {max_u:7.2f} (Internal u)")
    print(f"  ├─ Max Stellar Frac.  : {max_sf * 100:.2f}% (S_f)")
    print(f"  ├─ Max Metallicity    : {max_zf * 100:.2f}% (Z)")
    print(f"  ├─ Active Black Holes : {int(total_bh)}")
    print(f"  └─ Quantum String Bits: {total_bits:.2e} / {CRITICAL_STRING_BITS:.2e}")
    if pinch_off > 0:
        print(f"  \033[1;35m[ANOMALY: DIMENSIONAL PINCH-OFF DETECTED! NEW BUBBLE UNIVERSE SPAWNED]\033[0m")
    print("")

# ==============================================================================
# --- 2. PHYSICS KERNELS (JIT OPTIMIZED) ---
# ==============================================================================
@jit
def solve_potential_flrw(rho_total_comoving, a_scale):
    dx_physical = (COMOVING_BOX * a_scale) / N_GRID 
    rho_physical = rho_total_comoving / (a_scale ** 3) 
    rho_f32 = rho_physical.astype(jnp.float32)
    rho_k = jnp.fft.rfftn(rho_f32)
    kx = jnp.fft.fftfreq(N_GRID, d=dx_physical)
    ky = jnp.fft.fftfreq(N_GRID, d=dx_physical)
    kz = jnp.fft.rfftfreq(N_GRID, d=dx_physical)
    KX, KY, KZ = jnp.meshgrid(kx, ky, kz, indexing='ij')
    k_sq = jnp.where((2 * jnp.pi)**2 * (KX**2 + KY**2 + KZ**2) == 0, 1e-12, (2 * jnp.pi)**2 * (KX**2 + KY**2 + KZ**2)).astype(jnp.float32)
    phi_k = (4.0 * jnp.pi * rho_k * jnp.exp(-0.5 * k_sq * 0.85**2)) / k_sq
    phi = jnp.fft.irfftn(phi_k.at[0,0,0].set(0.0), s=(N_GRID, N_GRID, N_GRID))
    return jnp.maximum(1.0 + phi.astype(jnp.float64), 1.0)

@jit
def compute_multiverse_sph(pos_b, vel_b, masses_b, u_b, s_f, z_f, bh_mask, acc_bits, a_scale, current_cooling_coeff):
    diff = pos_b[:, None, :] - pos_b[None, :, :]
    physical_diff = diff * a_scale
    dist_sq = jnp.where(jnp.sum(physical_diff**2, axis=-1) == 0, 1e-10, jnp.sum(physical_diff**2, axis=-1))
    dist = jnp.sqrt(dist_sq)
    q = dist / SPH_SMOOTHING_H
    
    kernel_val = jnp.where(q < 1.0, (1.0 - q)**3, 0.0)
    rho_total = jnp.sum(masses_b[:, None] * kernel_val, axis=1)
    
    star_forming_mask = jnp.where((rho_total > RHO_THRESH_STAR) & (u_b < U_THRESH_STAR) & (bh_mask == 0.0), 1.0, 0.0)
    delta_s_f = star_forming_mask * STAR_FORM_EFF * (rho_total / RHO_THRESH_STAR) * DT
    delta_s_f = jnp.minimum(delta_s_f, 1.0 - s_f) 
    sn_heating = delta_s_f * SN_FEEDBACK_ENERGY
    delta_z_f = delta_s_f * METAL_YIELD 
    
    new_bh_mask = jnp.where((rho_total > RHO_THRESH_BH) & (z_f > Z_THRESH_BH), 1.0, bh_mask)
    current_bits = new_bh_mask * (masses_b ** 2) * (rho_total / RHO_THRESH_BH)
    new_acc_bits = jnp.maximum(acc_bits, current_bits)
    total_string_bits = jnp.sum(new_acc_bits)
    pinch_off_trigger = jnp.where(total_string_bits > CRITICAL_STRING_BITS, 1.0, 0.0)
    
    active_gas_mask = (1.0 - new_bh_mask)
    gas_fraction = jnp.clip(1.0 - s_f, 0.01, 1.0)
    cooling_rate = current_cooling_coeff * rho_total * jnp.sqrt(u_b) * (1.0 + (z_f * COOLING_Z_MULTIPLIER)) * active_gas_mask
    
    effective_gas_weight = active_gas_mask * gas_fraction
    pressure = (GAMMA - 1.0) * rho_total * u_b * effective_gas_weight
    pressure_term = (pressure[:, None] / rho_total[:, None]**2) + (pressure[None, :] / rho_total[None, :]**2)
    grad_kernel = jnp.where(q < 1.0, -3.0 * (1.0 - q)**2 / SPH_SMOOTHING_H, 0.0)
    
    force_magnitude = masses_b[None, :] * pressure_term * grad_kernel
    force_vec = force_magnitude[..., None] * (physical_diff / dist[..., None])
    p_force_b = jnp.sum(force_vec, axis=1)
    
    vel_diff = vel_b[:, None, :] - vel_b[None, :, :]
    div_v = jnp.sum(vel_diff * physical_diff, axis=-1) / dist_sq
    pdv_heating = (pressure / rho_total) * jnp.sum(masses_b[:, None] * div_v * grad_kernel, axis=1) * active_gas_mask
    du_dt = pdv_heating - cooling_rate + (sn_heating / DT)
    
    return p_force_b, rho_total, du_dt, delta_s_f, delta_z_f, new_bh_mask, new_acc_bits, total_string_bits, pinch_off_trigger

@jit
def step_universe(pos_dm, vel_dm, pos_b, vel_b, alpha, p_force_b, rho_b, a_scale, H_current, bounces, epoch, phi, phi_dot, current_v0_quint, hubble_damping):
    min_alpha = jnp.min(alpha)
    trigger = jnp.where(min_alpha < 0.081, 1.0, 0.0)
    scaling = trigger * (1.0 / (min_alpha**2))
    
    com_dm = jnp.mean(pos_dm, axis=0)
    radial_dir_dm = (pos_dm - com_dm) / jnp.maximum(jnp.linalg.norm(pos_dm - com_dm, axis=1, keepdims=True), 1e-10)
    vel_dm_new = vel_dm + trigger * radial_dir_dm * (150.0 * scaling * DT)
    
    com_b = jnp.mean(pos_b, axis=0)
    radial_dir_b = (pos_b - com_b) / jnp.maximum(jnp.linalg.norm(pos_b - com_b, axis=1, keepdims=True), 1e-10)
    raw_vel_b = vel_b + trigger * radial_dir_b * (150.0 * scaling * DT) + (p_force_b * DT)
    damping_factor = jnp.where(epoch < 250, 0.95, 1.0)
    vel_b_new = raw_vel_b * damping_factor
    
    dV_dphi = -current_v0_quint * jnp.exp(-phi)
    phi_ddot = -3.0 * H_current * phi_dot - dV_dphi
    phi_dot_new = phi_dot + phi_ddot * DT
    phi_new = phi + phi_dot_new * DT
    
    rho_phi = 0.5 * (phi_dot_new**2) + (current_v0_quint * jnp.exp(-phi_new))
    H_damped = H_current * jnp.exp(-hubble_damping * epoch * DT)
    h_active = jnp.where(bounces > 0, 1.0, 0.0)
    new_H = H_damped + (h_active * rho_phi * DT) 
    new_a_scale = a_scale + (new_H * a_scale * DT)
    
    return vel_dm_new, vel_b_new, trigger, new_a_scale, new_H, phi_new, phi_dot_new


# ==============================================================================
# --- MAIN MASTER EXECUTION LOOP ---
# ==============================================================================
if __name__ == '__main__':
    # ---------------------------------------------------------
    # PHASE 1: GENERATION 1 (THE PARENT UNIVERSE)
    # ---------------------------------------------------------
    key = jax.random.PRNGKey(2026)
    key_dm, key_b = jax.random.split(key)

    masses_dm = jnp.ones(N_DM, dtype=jnp.float64) * MASS_DM
    pos_dm = jax.random.uniform(key_dm, (N_DM, 3), minval=0.1*COMOVING_BOX, maxval=0.9*COMOVING_BOX)
    vel_dm = jax.random.normal(key_dm, (N_DM, 3)) * 0.05
    masses_baryon = jnp.ones(N_BARYON, dtype=jnp.float64) * MASS_BARYON
    pos_b = jax.random.uniform(key_b, (N_BARYON, 3), minval=0.45*COMOVING_BOX, maxval=0.55*COMOVING_BOX)
    vel_b = jax.random.normal(key_b, (N_BARYON, 3)) * 0.001

    u_b = jnp.ones(N_BARYON, dtype=jnp.float64) * 150.0 
    s_f = jnp.zeros(N_BARYON, dtype=jnp.float64) 
    z_f = jnp.zeros(N_BARYON, dtype=jnp.float64)       
    bh_mask = jnp.zeros(N_BARYON, dtype=jnp.float64)   
    acc_bits = jnp.zeros(N_BARYON, dtype=jnp.float64) 

    a_scale, H_val, bounces = 1.0, 0.0, 0
    phi, phi_dot = PHI_INIT, PHI_DOT_INIT
    dx_comoving = COMOVING_BOX / N_GRID
    pinch_off_triggered = False 

    print_header("VERSION 33.0 (SEQUENTIAL FRACTAL MANIFOLD)", "Primary Quantum Node (Parent Universe)")

    for epoch in range(1, 1001):
        grid_coords_dm = jnp.clip(jnp.floor(pos_dm/dx_comoving).astype(jnp.int32), 0, N_GRID-1)
        flat_idx_dm = grid_coords_dm[:,0] * N_GRID**2 + grid_coords_dm[:,1] * N_GRID + grid_coords_dm[:,2]
        rho_dm = jax.ops.segment_sum(masses_dm, flat_idx_dm, num_segments=N_GRID**3)
        
        grid_coords_b = jnp.clip(jnp.floor(pos_b/dx_comoving).astype(jnp.int32), 0, N_GRID-1)
        flat_idx_b = grid_coords_b[:,0] * N_GRID**2 + grid_coords_b[:,1] * N_GRID + grid_coords_b[:,2]
        rho_b_grid = jax.ops.segment_sum(masses_baryon, flat_idx_b, num_segments=N_GRID**3)
        
        rho_total = ((rho_dm + rho_b_grid) / dx_comoving**3).reshape((N_GRID, N_GRID, N_GRID))
        alpha = 2.0 / (1.0 + solve_potential_flrw(rho_total, a_scale))
        
        p_force_b, rho_b_particle, du_dt, delta_s_f, delta_z_f, bh_mask, acc_bits, total_bits_val, pinch_off = compute_multiverse_sph(
            pos_b, vel_b, masses_baryon, u_b, s_f, z_f, bh_mask, acc_bits, a_scale, COOLING_COEFF)
        
        vel_dm, vel_b, trig, a_scale, H_val, phi, phi_dot = step_universe(
            pos_dm, vel_dm, pos_b, vel_b, alpha, p_force_b, rho_b_particle, a_scale, H_val, bounces, epoch, phi, phi_dot, V0_QUINTESSENCE, hubble_damping=0.0)
        
        if trig > 0: bounces += 1
        pos_dm = jnp.mod(pos_dm + vel_dm * DT, COMOVING_BOX)
        pos_b = jnp.mod(pos_b + vel_b * DT, COMOVING_BOX)
        
        u_b = jnp.maximum(u_b + (du_dt * DT), MIN_U) 
        s_f = jnp.clip(s_f + delta_s_f, 0.0, 1.0)
        z_f = jnp.clip(z_f + delta_z_f, 0.0, 1.0) 
        
        if float(jnp.max(pinch_off)) > 0:
            pinch_off_triggered = True

        if epoch % 100 == 0 or epoch == 1:
            log_and_print_telemetry(epoch, a_scale, float(jnp.max(rho_b_particle)), float(jnp.max(u_b)), 
                                    float(jnp.max(s_f)), float(jnp.max(z_f)), float(jnp.sum(bh_mask)), 
                                    float(total_bits_val), float(jnp.max(pinch_off)), "GEN 1")

    # ---------------------------------------------------------
    # PHASE 2: FRACTAL EXTRACTION
    # ---------------------------------------------------------
    print("\n\033[95m" + "="*80 + "\033[0m")
    print("\033[1;36m" + "        PHASE 2: FRACTAL MULTIVERSE EXTRACTION " + "\033[0m")
    print("\033[95m" + "="*80 + "\033[0m")

    if pinch_off_triggered:
        bh_mask_np = np.array(bh_mask)
        pos_b_np = np.array(pos_b)
        acc_bits_np = np.array(acc_bits)
        masses_baryon_np = np.array(masses_baryon)
        
        bh_indices = np.where(bh_mask_np == 1.0)[0]
        bh_positions = pos_b_np[bh_indices]
        bh_string_bits = acc_bits_np[bh_indices]
        
        clustering = DBSCAN(eps=SPH_SMOOTHING_H * 1.5, min_samples=1).fit(bh_positions)
        unique_black_holes = set(clustering.labels_)
        
        os.makedirs("Multiverse_Gen2_Seeds", exist_ok=True)
        valid_universes = 0
        
        for bh_id in unique_black_holes:
            particle_mask = (clustering.labels_ == bh_id)
            local_string_bits = float(np.sum(bh_string_bits[particle_mask]))
            local_mass = float(np.sum(masses_baryon_np[bh_indices][particle_mask]))
            
            if local_string_bits < 100.0: continue
            valid_universes += 1
                
            def mutate_constant(): return 1.0 + random.uniform(-0.02, 0.02) 
            gen2_MASS_DM = MASS_DM * mutate_constant()
            gen2_V0_QUINTESSENCE = V0_QUINTESSENCE * mutate_constant()
            gen2_COOLING_COEFF = COOLING_COEFF * mutate_constant()
            
            seed_file = f"Multiverse_Gen2_Seeds/universe_gen2_node_{bh_id}.h5"
            with h5py.File(seed_file, 'w') as f2:
                f2.attrs['Parent_Version'] = "33.0"
                f2.attrs['Generation'] = 2
                f2.attrs['Parent_Node_ID'] = int(bh_id)
                f2.attrs['Inherited_String_Bits'] = local_string_bits
                
                physics = f2.create_group("Physics_Constants")
                physics.attrs['MASS_DM'] = gen2_MASS_DM
                physics.attrs['V0_QUINTESSENCE'] = gen2_V0_QUINTESSENCE
                physics.attrs['COOLING_COEFF'] = gen2_COOLING_COEFF
                
        print(f"  SUCCESS: {valid_universes} distinct Generation 2 Seeds generated.")

        # ---------------------------------------------------------
        # PHASE 3: SEQUENTIAL ORCHESTRATOR
        # ---------------------------------------------------------
        print("\n\033[95m" + "="*80 + "\033[0m")
        print("\033[1;36m" + "        PHASE 3: SEQUENTIAL TPU JIT SUPREMACY " + "\033[0m")
        print("\033[95m" + "="*80 + "\033[0m")
        
        seed_files = glob.glob("Multiverse_Gen2_Seeds/*.h5")
        
        print(f" ► Igniting {len(seed_files)} dimensions sequentially.")
        print(" ► Note: Dimension 1 pays the ~4 minute JIT compiler tax. The rest will cascade instantly.\n")

        output_dir = "Multiverse_Gen2_Outcomes"
        os.makedirs(output_dir, exist_ok=True)

        for i, seed_path in enumerate(seed_files):
            try:
                with h5py.File(seed_path, 'r') as f:
                    node_id = f.attrs['Parent_Node_ID']
                    inherited_bits = f.attrs['Inherited_String_Bits']
                    physics = f["Physics_Constants"]
                    mutated_dm_mass = physics.attrs['MASS_DM']
                    mutated_v0 = physics.attrs['V0_QUINTESSENCE']
                    mutated_cooling = physics.attrs['COOLING_COEFF']
            except Exception as e:
                print(f" \033[1;31m[FAILED]\033[0m Node at {seed_path} corrupted.")
                continue

            print(f"   [IGNITING] Dimension {int(node_id):03d} ({i+1}/{len(seed_files)})... ", end="", flush=True)

            key = jax.random.PRNGKey(int(node_id) + 2026) 
            key_dm, key_b = jax.random.split(key)

            masses_dm = jnp.ones(N_DM, dtype=jnp.float64) * mutated_dm_mass
            pos_dm = jax.random.uniform(key_dm, (N_DM, 3), minval=0.1*COMOVING_BOX, maxval=0.9*COMOVING_BOX)
            vel_dm = jax.random.normal(key_dm, (N_DM, 3)) * 0.05 

            masses_baryon = jnp.ones(N_BARYON, dtype=jnp.float64) * MASS_BARYON
            pos_b = jax.random.uniform(key_b, (N_BARYON, 3), minval=0.45*COMOVING_BOX, maxval=0.55*COMOVING_BOX)
            vel_b = jax.random.normal(key_b, (N_BARYON, 3)) * 0.001

            primordial_heat = inherited_bits * 0.15 
            u_b = jnp.ones(N_BARYON, dtype=jnp.float64) * primordial_heat 
            s_f = jnp.zeros(N_BARYON, dtype=jnp.float64) 
            z_f = jnp.zeros(N_BARYON, dtype=jnp.float64)       
            bh_mask = jnp.zeros(N_BARYON, dtype=jnp.float64)   
            acc_bits = jnp.zeros(N_BARYON, dtype=jnp.float64) 

            a_scale, H_val, bounces = 1.0, 2.5, 0
            phi, phi_dot = PHI_INIT, PHI_DOT_INIT

            for epoch in range(1, 1001):
                grid_coords_dm = jnp.clip(jnp.floor(pos_dm/dx_comoving).astype(jnp.int32), 0, N_GRID-1)
                flat_idx_dm = grid_coords_dm[:,0] * N_GRID**2 + grid_coords_dm[:,1] * N_GRID + grid_coords_dm[:,2]
                rho_dm = jax.ops.segment_sum(masses_dm, flat_idx_dm, num_segments=N_GRID**3)
                
                grid_coords_b = jnp.clip(jnp.floor(pos_b/dx_comoving).astype(jnp.int32), 0, N_GRID-1)
                flat_idx_b = grid_coords_b[:,0] * N_GRID**2 + grid_coords_b[:,1] * N_GRID + grid_coords_b[:,2]
                rho_b_grid = jax.ops.segment_sum(masses_baryon, flat_idx_b, num_segments=N_GRID**3)
                
                rho_total = ((rho_dm + rho_b_grid) / dx_comoving**3).reshape((N_GRID, N_GRID, N_GRID))
                alpha = 2.0 / (1.0 + solve_potential_flrw(rho_total, a_scale))
                
                p_force_b, rho_b_particle, du_dt, delta_s_f, delta_z_f, bh_mask, acc_bits, total_bits_val, pinch_off = compute_multiverse_sph(
                    pos_b, vel_b, masses_baryon, u_b, s_f, z_f, bh_mask, acc_bits, a_scale, mutated_cooling)
                
                vel_dm, vel_b, trig, a_scale, H_val, phi, phi_dot = step_universe(
                    pos_dm, vel_dm, pos_b, vel_b, alpha, p_force_b, rho_b_particle, a_scale, H_val, bounces, epoch, phi, phi_dot, mutated_v0, hubble_damping=0.05)
                
                if trig > 0: bounces += 1
                pos_dm = jnp.mod(pos_dm + vel_dm * DT, COMOVING_BOX)
                pos_b = jnp.mod(pos_b + vel_b * DT, COMOVING_BOX)
                
                u_b = jnp.maximum(u_b + (du_dt * DT), MIN_U) 
                s_f = jnp.clip(s_f + delta_s_f, 0.0, 1.0)
                z_f = jnp.clip(z_f + delta_z_f, 0.0, 1.0) 

            out_file = f"{output_dir}/mature_gen2_node_{int(node_id)}.h5"
            with h5py.File(out_file, 'w') as f_out:
                f_out.attrs['Final_Scale_Factor'] = float(a_scale)
                f_out.attrs['Final_String_Bits'] = float(jnp.sum(acc_bits))
                f_out.attrs['Final_Stellar_Frac'] = float(jnp.max(s_f))
                
            print(f"\033[1;32mSTABLE\033[0m | a(t): {a_scale:.4f} | Stars: {float(jnp.max(s_f))*100:.2f}%")

        print("\n\033[95m" + "="*80 + "\033[0m")
        print("\033[1;36m" + "        ALL GENERATION 2 UNIVERSES HAVE MATURED. " + "\033[0m")
        print("\033[95m" + "="*80 + "\033[0m\n")

    else:
        print("\033[1;31m[SYSTEM] No dimensional pinch-off detected. Universe is genetically dead.\033[0m")
