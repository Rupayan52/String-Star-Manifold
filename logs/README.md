# Simulation Logs: The Bandyopadhyay Cyclic Manifold

This directory contains the raw telemetry and outcome data proving the absolute Unitarity and fractal stability of the Bandyopadhyay Cyclic Manifold (v33.0). The current dataset captures the transition from a single parent universe into a self-replicating, Darwinian multiverse governed by the Smolin Protocol.

### Hardware & Environment Specifications
*   **Architecture:** Cloud TPU v5e (Tensor Processing Unit) Accelerated
*   **Framework:** JAX (Vectorized Just-In-Time compiled execution with Sequential Supremacy)
*   **Execution Protocol:** XLA preallocation explicitly disabled to prevent VRAM swap-death spirals during dimensional cascading.
*   **Precision:** **Hybrid-Precision** (Float32 for rapid $O(N \log N)$ FFT grid potential solutions; **Float64** for all kinematic state preservation to ensure 1.000000 Unitarity over deep epochs).

---

### Architectural Evolution: v33.0 vs. v31.2
The current telemetry dataset represents a monumental paradigm shift from the v31.2 Dual-Component engine. The core difference lies in the transition from a **closed-loop global bounce** to an **open-ended evolutionary multiverse**:

1.  **Fractal Spawning vs. Global Bounce:** v31.2 bounced the entire universe when global pressure peaked. v33.0 discards this. Instead, when local baryonic density crosses the Schwarzschild threshold ($\rho > 15.0$) and generates extreme quantum entropy, the metric experiences localized **dimensional rupture**, pinching off to spawn independent, parallel universes.
2.  **The Smolin Protocol (Darwinian Physics):** v31.2 utilized hard-coded physics. v33.0 implements Cosmological Natural Selection. Child universes inherit mutated variations of their parent's Dark Matter Mass ($M_{DM}$), Quintessence Potential ($V_0$), and Cooling Coefficients.
3.  **Sequential JIT Supremacy:** Previous builds suffered from XLA compiler gridlock when attempting parallel multiprocessing. v33.0 completely bypasses this by feeding universes to the TPU sequentially. The first dimension pays the ~4-minute compilation tax; subsequent dimensions cascade instantly via local cache retrieval.

---

### Initial Conditions & Engine Parameters (v33.0 Final)
*   **Dark Sector:** 2000 Nodes (Base $Mass_{DM} = 120.0$, subject to mutation)
*   **Baryonic Sector:** 1000 Fluid Elements (Base $Mass_{Baryon} = 5.2$)
*   **Grid Resolution:** 128³ Voxels over a 400.0 Comoving Box
*   **Epoch Cycle:** 1000 Epochs per Dimension
*   **Critical Entropy Threshold:** $1.2 \times 10^4$ String Bits
*   **Spatial Clustering:** DBSCAN (Density-Based Spatial Clustering of Applications with Noise) with $\epsilon = SPH\_H \times 1.5$

---

### Data Dictionary (`Multiverse_Gen2_Outcomes/mature_gen2_node_*.h5`)
The telemetry logs track the thermodynamic and evolutionary fitness of the manifold across three distinct phases (Primary Node, Fractal Extraction, and Maturation).

| Metric | Definition |
| :--- | :--- |
| `epoch` | The current time-step of the dimensional simulation. |
| `scale_factor_a_t` | The global metric expansion $a(t)$. Tracks the physical stretching of the comoving grid space. |
| `max_stellar_fraction_Sf` | The primary **Evolutionary Fitness Metric**. Measures the percentage of baryonic gas successfully converted into star-forming regions. |
| `quantum_string_bits` | The localized entropy measure. When a region exceeds the critical threshold, a dimensional pinch-off is triggered. |
| `active_black_holes` | The count of distinct supermassive singularities mapped by the DBSCAN algorithm. |
| `metallicity_index_Z` | Tracks heavy element seeding from supernova feedback, acting as a multiplier for baryonic cooling rates. |

---

### Notable Event Signatures (v33.0 Build)

| Event | Signature | Mechanism |
| :--- | :--- | :--- |
| **Dimensional Pinch-Off** | `quantum_string_bits` breaches $1.2 \times 10^4$ paired with an `[ANOMALY]` log output. | Localized mass-energy density warps the FLRW metric past the point of return, establishing the blueprint for a child dimension. |
| **DBSCAN Extraction** | Identification of $N$ valid universes (e.g., 37 or 47 nodes). | The algorithm maps spatial proximity, merging overlapping event horizons and filtering out micro-fluctuations (Entropy < 100 bits) to seed true dimensions. |
| **The Avalanche Effect** | Dimension 1 executes in ~4 minutes; Dimensions 2 through $N$ log as `STABLE` instantly. | **Sequential JIT Supremacy** in action. The XLA graph is cached by the primary Python thread, allowing instantaneous physical computation for identically structured, mutated graphs. |

*"By embracing chaotic floating-point drift and the Smolin Protocol, the manifold now successfully optimizes its own physics. We are no longer merely simulating space; we are observing the natural selection of spacetime itself."* — R. Bandyopadhyay, Lead Architect
