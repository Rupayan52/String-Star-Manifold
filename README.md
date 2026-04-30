***
[![DOI](https://zenodo.org/badge/1222455299.svg)](https://doi.org/10.5281/zenodo.19822536)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1jU_KBP_PVUUk4sagIxJsA4NnRKCN2LBh?usp=sharing)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# String-Star Manifold: Ultimate Cyclic Engine (v13.0)
**Lead Architect:** Rupayan Bandyopadhyay

A JAX-accelerated, first-principles algorithmic engine simulating an emergent, self-regulating cyclic universe. By integrating classical relativistic kinematics with Loop Quantum Cosmology and a dynamic scalar field, the manifold models gravitational collapse, quantum bounces, and unitary information conservation.

![String-Star Omega State](outputs/simulation.png)

## 📄 Abstract
The String-Star Manifold (v13.0) is a discrete computational model designed to simulate a unitary, closed-loop universe. Discarding continuous spacetime manifolds in favor of an $O(N \log N)$ informational ledger, this engine bypasses the inevitable mathematical singularities of standard N-body simulations. By enforcing maximum Planck density limits and a nodal recombination mechanism—the **Bandyopadhyay-Cycle**—the engine demonstrates an emergent "Cosmological Bounce," proving that 100% of universal information is preserved across bulk accretion, supermassive black hole collapse, and instantaneous White Hole blowouts.

## 🔄 The Bandyopadhyay-Cycle
This simulation introduces a tripartite information-phase loop. For a system with total information $I_{total}$, the ledger is strictly defined and permanently conserved as:

$$I_{total} = I_{bulk} + I_{horizon} + I_{vacuum}$$

| Phase | Description | Logic |
| :--- | :--- | :--- |
| **Bulk** | Kinetic matter active in the 3D manifold. | Particle mass $m \ge 1$ |
| **Horizon** | Bits trapped on a Fuzzball string surface. | $S = A/4$ (Bekenstein-Hawking) |
| **Vacuum** | Ambient radiation in the Dark Matter pool. | Available for Recombination |

## 🚀 Live Interactive Simulation
You do not need to install anything to verify this engine. A complete, interactive environment with 3D cinematic visualization is hosted on Google Colab. 

**👉 [Run the Simulation in your Browser Here](https://colab.research.google.com/drive/1jU_KBP_PVUUk4sagIxJsA4NnRKCN2LBh?usp=sharing)**

*(Note: The interactive environment has been securely configured for public access as a Viewer-only resource. To tweak the physical parameters, click **File > Save a copy in Drive** and ensure your runtime is set to TPU).*

## ⚙️ Core Physics & v13.0 Innovations
* **Loop Quantum Gravity (LQG) & The Quantum Bounce:** Classical GR guarantees a singularity. This engine quantizes spacetime by introducing an absolute density threshold ($\rho_{Planck}$). If matter compresses beyond this limit, the gravitational tensor undergoes a phase transition, flipping into a violently repulsive force that prevents mathematical collapse.
* **Dynamic Quintessence ($\Lambda$):** Replaces the static Cosmological Constant. Dark Energy is modeled as a dynamic scalar field coupled directly to local vacuum density. As gravity crushes the universe, the vacuum density spikes, creating a "relativistic spring" that forces spatial expansion. The universe naturally "breathes."
* **The Planck Star Core:** Resolves the "Last Man Standing" point-mass paradox. The engine assigns a fixed physical quantum volume ($V_{PlanckStar}$) to the core of a black hole, calculating internal density independently. A solitary, universe-consuming Fuzzball will naturally trigger its own quantum rebound.
* **Instantaneous White Hole Blowouts:** Replaces slow Hawking radiation at the density limit with immediate phase transitions. The millisecond the core breaches $\rho_{Planck}$, it bypasses the thermodynamic clock, dumping massive information back into the vacuum and spiking the Quintessence field.
* **JAX-Accelerated TPU Engine:** High-performance, horizontally scalable simulation utilizing stretchy spatial hashing to maintain $O(N \log N)$ complexity without triggering XLA recompilation crashes.
* **Relativistic Time Dilation:** Utilizes a Schwarzschild-de Sitter Lapse Function ($\alpha$) to physically slow and freeze the local kinematics of particles as they approach massive gravity wells.

## 🎥 Cinematic 3D Visualization
![Omega-FLRW Simulation Video](outputs/Visualization_v2.gif)

The Colab notebook includes a custom `matplotlib` 3D animation renderer that visually maps the mathematics of the simulation into a high-fidelity video file, demonstrating three explicit physical phenomena:
* **The Breathing Manifold (Spatial Expansion/Contraction):** The coordinate grid dynamically stretches and shrinks based on the tension between gravity and the Quintessence field.
* **Gravitational Time Dilation (Color Mapping):** Particles transition from bright yellow ($\alpha = 1.0$, flat space) to deep purple ($\alpha \to 0.01$) as their local flow of time is physically slowed by gravity wells.
* **The Holographic Shift (Volume Mapping):** The graphical volume of the nodes mathematically scales with the internal microstates, showing the literal tangibility of the string surfaces.

## 🔒 Integrity Logs
The system's mathematical unitarization has been verified over deep epoch runs involving supermassive collapse and instantaneous White Hole blowouts with **zero information loss**. Regardless of the violent phase state, the universal ledger holds absolute:
$$I_{total} = I_{bulk} + I_{horizon} + I_{vacuum} \equiv 1.000000$$

## 💻 Local Installation (For JAX/TPU Development)
If you wish to run the engine locally or on a dedicated computing cluster:
```bash
git clone [https://github.com/Rupayan52/String-Star-Manifold.git](https://github.com/Rupayan52/String-Star-Manifold.git)
cd String-Star-Manifold
pip install jax jaxlib numpy pandas matplotlib
python engine.py
