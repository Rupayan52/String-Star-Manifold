import jax
import jax.numpy as jnp
import numpy as np
import time
import warnings

warnings.filterwarnings("ignore")

# --- 1. HARDWARE HANDSHAKE ---
print(f"System Online: {jax.devices()[0].device_kind.upper()} Calibrated.")

# --- 2. OMEGA-SCALE PRE-ALLOCATION ---
MAX_BODIES = 500 
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

# --- 3. PHYSICS CONSTANTS ---
G, C, LAMBDA = 1.0, 1.0, 0.008
GW_STRENGTH = 0.001   
MIN_EVAP_CONST = 0.5  

print("\n==========================================")
print("  INITIATING TRUE OMEGA PRODUCTION RUN    ")
print("==========================================")

for epoch in range(1, 101):
    events = []
    
    # 0. DARK MATTER RECOMBINATION (State: Pool -> Bulk)
    active = [i for i in range(MAX_BODIES) if (masses[i] + stored_microstates[i]) > 0]
    for node_idx, node_pos in enumerate(dark_matter_nodes):
        local_count = len([idx for idx in active if np.linalg.norm(pos[idx] - node_pos) < NODE_RADIUS])
        if local_count < TARGET_DENSITY and dark_matter_pool >= 2:
            slots = [i for i in range(MAX_BODIES) if (masses[i] + stored_microstates[i]) == 0][:2]
            if len(slots) == 2:
                for s in slots:
                    m_val = float(np.random.choice([1.0, 2.0, 3.0]))
                    masses[s] = m_val # Information becomes Bulk Matter
                    pos[s] = node_pos + np.random.uniform(-15, 15, size=3)
                    vel[s] = np.random.uniform(-0.1, 0.1, size=3)
                    dark_matter_pool -= m_val
                entanglement_map[slots], entanglement_map[slots] = slots, slots
                events.append(f"*** [NODE {node_idx}] Recombined Matter.")

    # 1. KINEMATICS & THE HOLOGRAPHIC SHIFT (State: Bulk -> Surface)
    active = [i for i in range(MAX_BODIES) if (masses[i] + stored_microstates[i]) > 0]
    for i in active:
        # Effective mass for gravity = core bits + horizon bits
        m_eff_i = masses[i] + stored_microstates[i]
        if m_eff_i <= 0: continue
        
        force = np.zeros(3)
        v_mag = np.linalg.norm(vel[i])
        gamma = 1.0 / np.sqrt(1.0 - (min(v_mag, 0.99)**2))
        
        for j in active:
            if i == j: continue
            m_eff_j = masses[j] + stored_microstates[j]
            if m_eff_j <= 0: continue
            
            diff = pos[j] - pos[i]
            dist = np.linalg.norm(diff) + 0.1
            r_s = 2.0 * (m_eff_i + m_eff_j)
            
            if dist < r_s:
                # ACCRETION: All bits from j move to i's horizon
                # Bulk bits of j + Surface bits of j -> Surface bits of i
                stored_microstates[i] += int(masses[j] + stored_microstates[j])
                masses[j], stored_microstates[j], entanglement_map[j] = 0.0, 0, -1
                events.append("!!! [HOLOGRAPHIC SHIFT] Bulk mass converted to Horizon Bits.")
            else:
                # Gravitational Wave Loss
                gw_loss = GW_STRENGTH * (m_eff_i**2 * m_eff_j**2) / (dist**5)
                vel[i] *= (1.0 - gw_loss)
                # Gravity Force
                force += (diff / dist) * (G * m_eff_i * m_eff_j / (dist**2) - LAMBDA * dist)

        vel[i] = (vel[i] + force / (m_eff_i * gamma))
        v_limit = np.linalg.norm(vel[i])
        if v_limit > 0.99: vel[i] = (vel[i] / v_limit) * 0.99
        pos[i] += vel[i]

    # 2. DYNAMIC THERMODYNAMICS (State: Surface -> Pool)
    for i in active:
        m_eff_i = masses[i] + stored_microstates[i]
        if m_eff_i >= 20.0:
            # T = 1/M. Radiation leaks from the stored_microstates first.
            temp_rate = MIN_EVAP_CONST + (30.0 / m_eff_i)
            
            # If the Fuzzball has surface bits, leak them back to the vacuum pool
            if stored_microstates[i] > 0:
                # Calculate capacity S = pi * r_s^2
                capacity = 3.14159 * (2 * m_eff_i)**2
                while capacity < stored_microstates[i] and stored_microstates[i] > 0:
                    stored_microstates[i] -= 1
                    dark_matter_pool += 1
                    events.append("~~~ [RECYCLING] Horizon bits returned to Vacuum.")
            
            # If the Fuzzball is empty of horizon bits, it starts evaporating its core
            elif masses[i] > 0:
                masses[i] -= temp_rate
                if masses[i] < 15.0:
                    events.append("!!! [EVAPORATION] Core dissolved.")
                    masses[i] = 0

    # 3. THE IRONCLAD LEDGER (Non-Redundant Sum)
    p_bulk, s_surf = 0, 0
    for k in range(MAX_BODIES):
        p_bulk += int(masses[k])
        s_surf += int(stored_microstates[k])
    
    total_inf = p_bulk + s_surf + int(dark_matter_pool)
    print(f"\n--- EPOCH {epoch} ---")
    if events:
        for e in list(set(events))[:2]: print(e)
    print(f"Bulk Matter: {p_bulk} | Horizon Bits: {s_surf} | Vacuum Pool: {int(dark_matter_pool)}")
    print(f"System Integrity: {total_inf}/{int(INITIAL_UNIVERSE_BITS)} Bits")
    
    time.sleep(0.3)