[![DOI](https://zenodo.org/badge/1222455299.svg)](https://doi.org/10.5281/zenodo.19822536)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zB3BhM96oTJAz2oAeIhe4tZVzuW_2_e6?usp=sharing)
# The Bandyopadhyay Cyclic Manifold: Fractal Multiverse Engine (v33.0)
**Lead Architect:** Rupayan Bandyopadhyay  
**Release:** v33.0 (The Immutable Entropy Update)  
**Timestamp:** May 2026  

A high-performance, JAX-accelerated cosmological engine simulating a non-singular, cyclic, and **fractal multiverse**. This version (v33.0) represents a monolithic architectural leap, evolving from a single bouncing universe into a self-replicating manifold. It integrates **Smoothed Particle Hydrodynamics (SPH)**, **Collisionless Dark Matter (PM)**, and the **Smolin Protocol** (Cosmological Natural Selection) to model the birth of distinct child dimensions from singular event horizons.

---

## 🌌 Abstract: The Smolin-Bandyopadhyay Protocol

The Bandyopadhyay Cyclic Manifold (v33.0) fundamentally rewrites the rules of cosmic iteration. Discarding the singular, universe-wide bounce of previous builds, this engine implements localized dimensional rupture. Utilizing a hybrid **Particle-Mesh (PM)** and **Smoothed Particle Hydrodynamics (SPH)** architecture, the engine maps the gravitational collapse of collisionless Dark Matter and the hydrodynamic behavior of baryonic fluid. 

When local baryonic density crosses the Schwarzschild threshold, generating extreme quantum entropy (String Bits), the metric pinches off. The engine dynamically maps these singularities using **DBSCAN spatial clustering** and spawns independent Generation 2 "child" universes. Each child inherits mutated physical constants (Dark Matter Mass, Quintessence Potential, Cooling Rates) in a strict demonstration of Cosmological Natural Selection.

## 🚀 Architectural Evolution: v33.0 vs. Legacy Builds

The v33.0 engine marks a profound shift from a closed-loop bouncing simulation to an open-ended, evolutionary multiverse generator:

*   **From Global Bounce to Fractal Spawning:** The global bounce is replaced by **localized dimensional rupture**. Extreme density pockets ($\rho > 15.0$) trigger localized black hole formations that spawn independent, parallel universes.
*   **From Static to Darwinian Constants:** Introduces the **Smolin Protocol**. Child universes inherit mutated variations of their parent's Dark Matter Mass, Quintessence, and Cooling Coefficients.
*   **From Single Thread to Sequential JIT Supremacy:** Simulating a multiverse requires extreme hardware optimization. v33.0 introduces a custom orchestrator designed for Cloud TPU v5e nodes. It utilizes **Sequential JIT Supremacy** to completely bypass XLA compiler gridlock, securely mapping and maturing dozens of parallel dimensions without memory swap failure.
*   **From Pressure to Thermodynamic Expansion:** The initial conditions for Generation 2 universes have been corrected from a chaotic kinetic burst to a **Thermodynamic Big Bang**, where expansion is driven smoothly by inherited primordial thermal pressure ($u_b$).

## 🔄 The Smolin Extraction Cycle

The simulation operates in three distinct, automated phases to execute Cosmological Natural Selection:

| Phase | Description | Key Mechanism |
| :--- | :--- | :--- |
| **I. Primary Quantum Node (Gen 1)** | Simulates the parent universe for 1,000 epochs, tracking Dark Matter halos and baryonic collapse into singularities. | $O(N^2)$ SPH Interaction & Black Hole Masking |
| **II. Fractal Extraction** | Analyzes the final state of Gen 1. Uses spatial clustering to identify distinct supermassive black holes, filtering out quantum micro-fluctuations (Entropy < 100 bits). | DBSCAN Algorithm & Genetic Mutation |
| **III. Sequential JIT Maturation** | Iteratively boots and simulates every generated child universe (Gen 2). Tracks their evolutionary fitness through star formation ($S_f$). | Hardware-Optimized Sequential Execution |

## 💻 Hardware Specifications & Execution

This engine is designed to push the absolute limits of memory allocation on modern tensor hardware.

*   **Target Architecture:** Cloud TPU v5e (or high-VRAM GPU equivalent).
*   **Hardware Safety:** The engine explicitly disables JAX's aggressive VRAM preallocation (`XLA_PYTHON_CLIENT_PREALLOCATE='false'`) to prevent swap-memory death spirals.
*   **Execution Protocol:** Do not attempt arbitrary Python `multiprocessing`. The engine relies on **Sequential Execution** to manage the XLA compiler cache. The first dimension pays a ~4-minute JIT compilation tax; all subsequent dimensions cascade instantly.

**Dependencies:**
```bash
pip install jax jaxlib h5py scikit-learn numpy
