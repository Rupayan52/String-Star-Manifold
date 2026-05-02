# ==============================================================================
# Bandyopadhyay Cyclic Manifold (v31.2 Final) - Full Execution Suite
# Location: Primary Quantum Node | Kolkata Region, West Bengal
# ==============================================================================

import os
import csv
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from jax import jit, config

# Ensure JAX uses 64-bit and FFmpeg is ready for video encoding
config.update("jax_enable_x64", True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- 1. CORE SIMULATION PARAMETERS ---
N_GRID, COMOVING_BOX, DT = 128, 400.0, 0.0015
N_DM, MASS_DM = 2000, 15.5
N_BARYON, MASS_BARYON = 1000, 5.2
SPH_SMOOTHING_H, SPH_K_PRESSURE = 7.5, 2500.0
LAMBDA_DE = 1.85

# --- 2. PHYSICS KERNELS (v31.2) ---

@jit
def solve_potential_flrw(rho_total_comoving, a_scale):
    dx_physical = (COMOVING_BOX * a_scale) / N_GRID 
    rho_physical = rho_total_comoving / (a_scale ** 3) 
    rho_f32 = rho_physical.astype(jnp.float32)
    rho_k = jnp.fft.rfftn(rho_f32)
    kx, ky, kz = jnp.fft.fftfreq(N_GRID, d=dx_physical), jnp.fft.fftfreq(N_GRID, d=dx_physical), jnp.fft.rfftfreq(N_GRID, d=dx_physical)
    KX, KY, KZ = jnp.meshgrid(kx, ky, kz, indexing='ij')
    k_sq = jnp.where((2 * jnp.pi)**2 * (KX**2 + KY**2 + KZ**2) == 0, 1e-12, (2 * jnp.pi)**2 * (KX**2 + KY**2 + KZ**2)).astype(jnp.float32)
    phi_k = (4.0 * jnp.pi * rho_k * jnp.exp(-0.5 * k_sq * 0.85**2)) / k_sq
    phi = jnp.fft.irfftn(phi_k.at[0,0,0].set(0.0), s=(N_GRID, N_GRID, N_GRID))
    return jnp.maximum(1.0 + phi.astype(jnp.float64), 1.0)

@jit
def compute_baryon_collisions(pos_baryon, masses_baryon, a_scale):
    diff = pos_baryon[:, None, :] - pos_baryon[None, :, :]
    physical_diff = diff * a_scale
    dist_sq = jnp.where(jnp.sum(physical_diff**2, axis=-1) == 0, 1e-10, jnp.sum(physical_diff**2, axis=-1))
    dist = jnp.sqrt(dist_sq)
    q = dist / SPH_SMOOTHING_H
    kernel_val = jnp.where(q < 1.0, (1.0 - q)**3, 0.0)
    rho = jnp.sum(masses_baryon * kernel_val, axis=1)
    pressure = SPH_K_PRESSURE * (rho ** 2)
    pressure_term = (pressure[:, None] / rho[:, None]**2) + (pressure[None, :] / rho[None, :]**2)
    grad_kernel = jnp.where(q < 1.0, -3.0 * (1.0 - q)**2 / SPH_SMOOTHING_H, 0.0)
    force_magnitude = masses_baryon[None, :] * pressure_term * grad_kernel
    force_vec = force_magnitude[..., None] * (physical_diff / dist[..., None])
    return jnp.sum(force_vec, axis=1), jnp.max(rho)

@jit
def step_universe(pos_dm, vel_dm, pos_b, vel_b, alpha, p_force_b, a_scale, H_current, bounces, epoch):
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
    h_active = jnp.where(bounces > 0, 1.0, 0.0)
    new_H = H_current + (h_active * LAMBDA_DE * DT)
    new_a_scale = a_scale + (new_H * a_scale * DT)
    return vel_dm_new, vel_b_new, trigger, new_a_scale, new_H

# --- 3. TELEMETRY & LOGGING ---
telemetry_history = []

def log_and_print_telemetry(epoch, phase, a_t, pressure, alpha_min, bounces):
    color = "\033[1;32m" if "BANG" in phase else "\033[1;31m"
    symbol = "[ >>> EXPANDING >>> ]" if "BANG" in phase else "[ <<< CRUNCHING <<< ]"
    print(f"\033[1mEPOCH {epoch:04d}\033[0m | {color}{symbol}\033[0m")
    print(f"  ├─ Scale Factor a(t)  : {a_t:.4f} x")
    print(f"  ├─ Baryon Pressure    : {pressure:7.2f} bits")
    print(f"  └─ Quantum Bounces    : {bounces}\n")
    telemetry_history.append({"epoch": epoch, "phase": phase, "at": a_t, "pressure": pressure, "alpha": alpha_min, "bounces": bounces})

# --- 4. EXECUTION LOOP ---
key = jax.random.PRNGKey(2026)
pos_dm = jax.random.uniform(key, (N_DM, 3), minval=0.1*COMOVING_BOX, maxval=0.9*COMOVING_BOX)
vel_dm = jax.random.normal(key, (N_DM, 3)) * 0.05
pos_b = jax.random.uniform(key, (N_BARYON, 3), minval=0.45*COMOVING_BOX, maxval=0.55*COMOVING_BOX)
vel_b = jax.random.normal(key, (N_BARYON, 3)) * 0.001

a_scale, H_val, bounces = 1.0, 0.0, 0
dx_comoving = COMOVING_BOX / N_GRID

# INITIALIZE HISTORY LISTS
dm_history, baryon_history, a_history = [], [], []

print("--- [ INITIATING SIMULATION AT QUANTUM NODE ] ---")

for epoch in range(1, 1001):
    grid_coords_dm = jnp.clip(jnp.floor(pos_dm/dx_comoving).astype(jnp.int32), 0, N_GRID-1)
    flat_idx_dm = grid_coords_dm[:,0] * N_GRID**2 + grid_coords_dm[:,1] * N_GRID + grid_coords_dm[:,2]
    rho_dm = jax.ops.segment_sum(jnp.ones(N_DM)*MASS_DM, flat_idx_dm, num_segments=N_GRID**3)
    
    grid_coords_b = jnp.clip(jnp.floor(pos_b/dx_comoving).astype(jnp.int32), 0, N_GRID-1)
    flat_idx_b = grid_coords_b[:,0] * N_GRID**2 + grid_coords_b[:,1] * N_GRID + grid_coords_b[:,2]
    rho_b = jax.ops.segment_sum(jnp.ones(N_BARYON)*MASS_BARYON, flat_idx_b, num_segments=N_GRID**3)
    
    rho_total = ((rho_dm + rho_b) / dx_comoving**3).reshape((N_GRID, N_GRID, N_GRID))
    alpha = 2.0 / (1.0 + solve_potential_flrw(rho_total, a_scale))
    p_force_b, max_rho = compute_baryon_collisions(pos_b, jnp.ones(N_BARYON)*MASS_BARYON, a_scale)
    
    vel_dm, vel_b, trig, a_scale, H_val = step_universe(pos_dm, vel_dm, pos_b, vel_b, alpha, p_force_b, a_scale, H_val, bounces, epoch)
    if trig > 0: bounces += 1
    
    pos_dm = jnp.mod(pos_dm + vel_dm * DT, COMOVING_BOX)
    pos_b = jnp.mod(pos_b + vel_b * DT, COMOVING_BOX)
    
    # DATA COLLECTION FOR VIDEO (Every 5 epochs as per your request)
    if epoch % 5 == 0:
        dm_history.append(np.array(pos_dm))
        baryon_history.append(np.array(pos_b))
        a_history.append(float(a_scale))

    if epoch % 100 == 0 or epoch == 1:
        phase_str = "BIG BANG (HUBBLE FLOW)" if (bounces > 0 and float(jnp.min(alpha)) > 0.15) else "GRAVITATIONAL CRUNCH"
        log_and_print_telemetry(epoch, phase_str, a_scale, max_rho, float(jnp.min(alpha)), bounces)

# --- 5. VISUALIZATION ENGINE ---

def render_manifold_video(dm_h, b_h, a_h, filename="manifold_v31_2.mp4"):
    fig = plt.figure(figsize=(10, 10), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')
    ax.set_xlim(0, 400); ax.set_ylim(0, 400); ax.set_zlim(0, 400); ax.axis('off')

    dm_plot = ax.scatter([], [], [], c='white', s=1, alpha=0.1)
    baryon_plot = ax.scatter([], [], [], c=[], cmap='inferno', s=15, edgecolors='none')
    title = ax.set_title("Quantum Node | Bandyopadhyay Cyclic Manifold v31.2", color='white', fontsize=14, pad=-20)

    def update(frame):
        dm_plot._offsets3d = (dm_h[frame][:, 0], dm_h[frame][:, 1], dm_h[frame][:, 2])
        b_pos = b_h[frame]
        baryon_plot._offsets3d = (b_pos[:, 0], b_pos[:, 1], b_pos[:, 2])
        
        # Fixed center for color mapping
        center = np.array([200, 200, 200])
        dist = np.linalg.norm(b_pos - center, axis=1)
        baryon_plot.set_array(1.0 / (dist + 1e-5)) 
        title.set_text(f"Quantum Node | a(t): {a_h[frame]:.3f}x")
        ax.view_init(elev=20, azim=frame * 0.5)
        return dm_plot, baryon_plot, title

    print(f"--- [ INITIATING VIDEO ENCODE: {filename} ] ---")
    ani = FuncAnimation(fig, update, frames=len(dm_h), blit=False)
    writer = FFMpegWriter(fps=30, metadata=dict(artist='Rupayan Bandyopadhyay'), bitrate=5000)
    ani.save(filename, writer=writer)
    plt.close()
    print(f">>> Encode Complete. File saved at Primary Quantum Node.")

# RUN RENDERER
render_manifold_video(dm_history, baryon_history, a_history)

# --- 6. CSV EXPORT ---
with open("quantum_node_telemetry.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=telemetry_history[0].keys())
    writer.writeheader()
    writer.writerows(telemetry_history)