[![DOI](https://zenodo.org/badge/1222455299.svg)](https://doi.org/10.5281/zenodo.19822536)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1eB89a0FUgZUQL5Qs_uTuEzmeXdH88ar4?usp=sharing)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# The Bandyopadhyay Cyclic Manifold: Dual-Component Engine (v31.2)
**Lead Architect:** Rupayan Bandyopadhyay
**Primary Quantum Node:** Kolkata Region, West Bengal

A high-performance, JAX-accelerated cosmological engine simulating a non-singular cyclic universe. This version represents a fundamental architectural leap, fusing **Collisionless Dark Matter (PM)** with **Hydrodynamic Baryonic Fluid (SPH)** to model an emergent "Quantum Bounce" driven by physical fluid pressure rather than scripted triggers.

---

## 📄 Abstract
The Bandyopadhyay Cyclic Manifold (v31.2) discards artificial "white hole" logic flags in favor of first-principles emergent physics. Utilizing a hybrid **Particle-Mesh (PM)** and **Smoothed Particle Hydrodynamics (SPH)** architecture, the engine simulates a dual-sector universe where Dark Matter provides the gravitational scaffolding for a Baryonic core. By integrating the **FLRW metric tensor** and a non-linear **Schwarzschild-de Sitter Lapse ($\alpha$)**, the simulation demonstrates that a non-singular bounce is a natural consequence of baryonic fluid reaching peak compression limits.

## 🔄 The Dual-Component Cycle
The simulation operates on a tripartite energy-sector loop, where total universal information $I_{total}$ is strictly conserved across the bounce:

$$I_{total} = I_{dark} + I_{baryon} + I_{vacuum} \equiv 1.000000$$

| Sector | Description | Kinematics |
| :--- | :--- | :--- |
| **Dark Matter** | Collisionless point masses driving global gravity. | $O(N \log N)$ FFT-PM Solver |
| **Baryonic Fluid** | Collisional matter generating hydrodynamic pressure. | $O(N^2)$ SPH Interaction |
| **Vacuum** | Dynamic Dark Energy pool ($\Lambda$) driving expansion. | FLRW Metric Scaling |

## 🚀 Live Interactive Simulation
A complete, high-fidelity interactive environment is hosted on Google Colab, optimized for TPU-accelerated execution.

**👉 [Run the v31.2 Engine at the Quantum Node](https://colab.research.google.com/drive/1eB89a0FUgZUQL5Qs_uTuEzmeXdH88ar4?usp=sharing)**

*(Note: To modify physical parameters or scale the particle count, click **File > Save a copy in Drive** and ensure your runtime is set to **TPU**).*

## ⚙️ v31.2 Core Innovations
*   **Emergent Hydrodynamic Bounce:** Gravity is countered by a vectorized SPH kernel calculating real-time inter-particle repulsion ($P = k\rho^2$).
*   **Adiabatic Relaxation Layer:** Implements a custom velocity-damping phase for $t < 250$ to solve the "Initial Condition Shock" paradox, allowing stable fluid accretion within Dark Matter halos.
*   **Hybrid-Precision FFT Solver:** Utilizes `float32` for $128^3$ mesh potential solutions while maintaining `float64` for particle states to preserve absolute unitarity.
*   **FLRW Metric Expansion:** Spatial coordinates are coupled directly to the dynamic scale factor $a(t)$, simulating genuine Hubble Flow expansion post-bounce.
*   **Detailed Telemetry Logging:** Professional-grade output tracking metric lapse ($\alpha$), baryon pressure, and scale factor, with automated export to `quantum_node_telemetry.csv`.

## 🎥 Professional Telemetry & Visualization
The v31.2 engine features high-fidelity terminal logging with self-explanatory descriptive tooltips for each epoch:
```text
EPOCH 0200 | [ <<< CRUNCHING <<< ]
  ├─ Scale Factor a(t)  : 1.0863 x   |████░░░░░░░░░░░░░░░░░░░░|
  │  (Current size of the universe relative to the initial state)
  ├─ Baryon Pressure    :   25.91 bits
  │  (Inter-particle repulsion preventing total singularity collapse)
  ├─ Metric Lapse (α)   : 0.841203
  │  (Gravitational time dilation; lower values indicate higher density)
  └─ Quantum Bounces    : 0
     (Number of successful non-singular phase transitions completed)
