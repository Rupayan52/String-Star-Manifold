[![DOI](https://zenodo.org/badge/1222455299.svg)](https://doi.org/10.5281/zenodo.19822536)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zB3BhM96oTJAz2oAeIhe4tZVzuW_2_e6?usp=sharing)
# The Bandyopadhyay Cyclic Manifold: Fractal Multiverse Engine (v33.0)
**Lead Architect:** Rupayan Bandyopadhyay  
**Release:** v33.0 (The Immutable Entropy Update)  
May 2026  

A high-performance, JAX-accelerated cosmological engine simulating a non-singular, cyclic, and **fractal multiverse**. This version (v33.0) represents a monolithic architectural leap, evolving from a single bouncing universe into a self-replicating manifold. It integrates **Smoothed Particle Hydrodynamics (SPH)**, **Collisionless Dark Matter (PM)**, and the **Smolin Protocol** (Cosmological Natural Selection) to model the birth of distinct child dimensions from singular event horizons.

---

## Abstract: The Smolin-Bandyopadhyay Protocol

The Bandyopadhyay Cyclic Manifold (v33.0) fundamentally rewrites the rules of cosmic iteration. Discarding the singular, universe-wide bounce of previous builds, this engine implements localized dimensional rupture. Utilizing a hybrid **Particle-Mesh (PM)** and **Smoothed Particle Hydrodynamics (SPH)** architecture, the engine maps the gravitational collapse of collisionless Dark Matter and the hydrodynamic behavior of baryonic fluid. 

When local baryonic density crosses the Schwarzschild threshold, generating extreme quantum entropy (String Bits), the local Friedmann-Lemaître-Robertson-Walker (FLRW) metric pinches off. The engine dynamically maps these singularities using **DBSCAN spatial clustering** and spawns independent Generation 2 "child" universes. Each child inherits mutated physical constants (Dark Matter Mass, Quintessence Potential, Cooling Rates) in a strict demonstration of Cosmological Natural Selection.

---

## Architectural Evolution: v33.0 vs. Legacy Builds

The v33.0 engine marks a profound shift from a closed-loop bouncing simulation to an open-ended, evolutionary multiverse generator:

*   **From Global Bounce to Fractal Spawning:** The global bounce is replaced by **localized dimensional rupture**. Extreme density pockets ($\rho > 15.0$) trigger localized black hole formations that spawn independent, parallel universes rather than forcing the entire manifold to rebound.
*   **From Static to Darwinian Constants:** Introduces the **Smolin Protocol**. Child universes inherit mutated variations of their parent's foundational physics, allowing the multiverse to actively evolve.
*   **From Single Thread to Sequential JIT Supremacy:** Simulating a multiverse requires extreme hardware optimization. v33.0 introduces a custom orchestrator designed for Cloud TPU v5e nodes. It utilizes **Sequential JIT Supremacy** to completely bypass XLA compiler gridlock, securely mapping and maturing dozens of parallel dimensions without memory swap failure.
*   **From Pressure to Thermodynamic Expansion:** The initial conditions for Generation 2 universes have been corrected from chaotic kinetic bursts to a **Thermodynamic Big Bang**, where expansion is driven natively by inherited primordial thermal pressure ($u_b$).

---

## The Step-by-Step Mechanics of Fractal Multigenesis

To understand the v33.0 runtime, one must follow the exact mathematical and topological progression of matter from the birth of the Primary Node (Gen 1) to the eruption of the Fractal Cascade (Gen 2).

### Step 1: The Thermodynamic Genesis (Epoch 0 - 250)
The Primary Quantum Node boots. The universe begins as an expanding comoving grid containing collisionless Dark Matter and a dense, hot Baryonic fluid. 
*   **Hubble Expansion:** Driven by the FLRW metric, space expands.
*   **Adiabatic Damping:** Initial explosive kinetic energies are smoothly damped to allow the fluid to settle into the expanding Dark Matter scaffolding without artificial detonation.

### Step 2: Dual-Component Collapse (Epoch 250 - 700)
As the universe cools, Dark Matter forms deep gravitational wells solved via a Fast Fourier Transform Particle-Mesh ($O(N \log N)$). 
*   Baryonic matter is dragged into these wells. 
*   The SPH kernel calculates localized pressure ($P = k\rho^\gamma$) and artificial viscosity, creating extreme accretion shockwaves and localized heating as matter compresses.

### Step 3: Dimensional Pinch-Off (The Information Limit)
As baryons cross the critical density threshold ($\rho > 15.0$), the engine tracks the accumulation of Quantum String Bits ($S_{local}$) on the event horizon. 
*   When a localized pocket breaches $S_{local} > 1.2 \times 10^4$ bits, the lapse function ($\alpha$) of the metric collapses to zero.
*   **The Rupture:** The topology of the manifold is severed. The singularity isolates itself from the parent universe, becoming a mature "seed" for a new dimension.

### Step 4: DBSCAN Extraction & Entropy Filtration
At Epoch 1,000, the simulation halts. The orchestrator scans the grid using **Density-Based Spatial Clustering of Applications with Noise (DBSCAN)**.
*   Overlapping event horizons are merged into supermassive singularity seeds.
*   **Entropy Filter:** Any micro-fluctuation containing less than 100 string bits is classified as Hawking radiation noise and deleted. Only viable supermassive seeds remain (yielding dynamically chaotic fertility, typically 35–50 nodes).

### Step 5: The Smolin Protocol (Genetic Drift)
Before a child dimension is booted, its physical constants undergo stochastic genetic mutation via a drift matrix $\mathbf{M}$:
$$ \vec{\Theta}_{Gen2} = \mathbf{M} \cdot \vec{\Theta}_{Gen1} $$
*   The child dimension is assigned a mutated Dark Matter Mass ($M_{DM}$), Quintessence Potential ($V_0$), and Baryonic Cooling Coefficient ($C_{cool}$).

### Step 6: Generation 2 Eruption (Thermodynamic Big Bang)
The child universe boots in complete topological isolation. 
*   **Immutable Entropy:** Its initial thermal energy ($u_b$) is strictly proportional to the exact quantum string bits captured by the parent singularity: $u_{b, Gen2}(0) = \kappa \cdot S_{parent}$.
*   This intense heat drives a localized Thermodynamic Big Bang against the new dimension's FLRW metric, restarting the cycle. 

---

## Evolutionary Fitness & Telemetry

The ultimate goal of the Bandyopadhyay Manifold is to observe Cosmological Natural Selection. The "fitness" of any spawned dimension is objectively measured by its **Stellar Fraction ($S_f$)**.

$$ \mathcal{F}(\mathcal{U}) = S_f = \lim_{t \to t_{end}} \frac{M_{stars}}{M_{baryons}} $$

Universes whose mutated constants allow for the most efficient baryonic cooling and star formation will inherently produce more supermassive black holes, thereby passing their optimal genetic parameters to Generation 3. 

All telemetry for Generation 2 dimensions—including dynamic scale factors, entropy arrays, and final $S_f$ rankings—are securely exported as high-density HDF5 binaries (`.h5`) to the `Multiverse_Gen2_Outcomes` directory.

---

## Hardware Specifications & Execution Warnings

This engine is designed to push the absolute limits of memory allocation on modern tensor hardware. **Attempting to run this on standard consumer hardware will result in an immediate Out-Of-Memory (OOM) fatal crash.**

*   **Target Architecture:** Cloud TPU v5e (or high-VRAM GPU equivalent).
*   **The Swap-Death Spiral:** Simulating 40+ branching 64-bit tensor graphs simultaneously will cause XLA compiler gridlock. Do NOT use Python `multiprocessing`.
*   **Sequential JIT Supremacy:** The engine strictly enforces sequential execution. 
    *   To protect VRAM, the engine explicitly disables aggressive preallocation: `os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'`.
    *   **The Avalanche Effect:** Dimension 1 pays a ~4-minute XLA JIT compilation tax. Because mutated constants are passed as scalars, Dimensions 2 through $N$ bypass compilation entirely via local cache retrieval, executing their 1,000 epochs with near-instantaneous hardware acceleration.

### Dependencies
```bash
pip install jax jaxlib h5py scikit-learn numpy
