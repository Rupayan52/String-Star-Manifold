# The Bandyopadhyay Cyclic Manifold: Dual-Component Engine (v31.2)

**Lead Architect:** Rupayan Bandyopadhyay  
**Primary Quantum Node:** Kolkata Region, West Bengal  
**Documentation Date:** May 2, 2026  

## 1. Executive Summary & The Paradigm Shift (Legacy vs. v31.2)
The v31.2 engine represents a fundamental architectural leap over the legacy v13.0/v17.0 String-Star models. The core philosophical shift is the transition from **prescriptive logic** (hard-coded thresholds) to **emergent continuum physics**. 

By discarding artificial phase flags, the manifold now proves that a non-singular, cyclic universe arises naturally from the interplay of collisionless gravity and hydrodynamic fluid pressure.

### **Key Upgrades Over the Legacy Model:**
*   **DEPRECATED:** Scripted White Hole Blowouts, explicit Loop Quantum Gravity (LQG) cutoffs, and rigid Fuzzball string-tension boundaries.
*   **IMPLEMENTED:** Dual-Component Universe (PM Gravity + SPH Fluid Dynamics).
*   **IMPLEMENTED:** The "Emergent Bounce," where mathematically rigorous baryonic fluid pressure naturally overcomes total gravitational collapse.
*   **IMPLEMENTED:** Adiabatic Relaxation Layer to prevent initial-condition numerical shocks.

---

## 2. The Dual-Component Architecture
The legacy manifold treated all mass as identical nodes. The v31.2 engine splits the universe into two distinct physical sectors, acting in continuous tension:

1.  **The Dark Sector (Collisionless Scaffolding):** Modeled via an $O(N \log N)$ Fast Fourier Transform Particle-Mesh (FFT-PM) solver. Dark matter acts as point masses that pass freely through one another, providing the massive gravitational wells necessary to initiate the "Crunch."
2.  **The Baryonic Sector (Hydrodynamic Core):** Modeled via a vectorized Smoothed Particle Hydrodynamics (SPH) kernel. Baryons are collisional fluid elements that cannot occupy the same space, generating extreme outward shockwaves when compressed.

## 3. The Emergent Quantum Bounce
In previous models, the simulation artificially halted collapse when density reached $\rho_{Planck}$, triggering a forced "White Hole" expansion. In v31.2, this is solved via first-principles physics. 

As the Dark Sector crushes the Baryonic Fluid into the central manifold, the inter-particle distance decreases, spiking the SPH kernel density. The system generates an exponential pressure gradient based on fluid stiffness ($k$):
$$ P = k\rho^2 $$
When this hydrodynamic pressure mathematically exceeds the gravitational PM tensor, the velocity vectors of the Baryonic fluid violently reverse. The **Quantum Bounce** is no longer a scripted event; it is a native, self-regulating hydrodynamic rebound preventing the mathematical singularity.

## 4. The Information Conservation Mandate (Unitarity)
The absolute preservation of information remains the bedrock of the Bandyopadhyay-Cycle. However, the tripartite loop has been updated to reflect the new Dual-Component reality. Total information complexity is rigorously verified at each epoch:
$$ I_{total} = I_{dark} + I_{baryon} + I_{vacuum} \equiv 1.000000 $$

To guarantee 100% Unitarity over deep epochs, the engine utilizes **Hybrid-Precision computation**: rapid `float32` arrays solve the global FFT mesh potentials, while all discrete kinematic states and ledgers are preserved in strict `float64` to prevent floating-point drift during violent phase transitions.

## 5. FLRW Metric & True Hubble Flow
Replacing the localized "Relativistic Spring" of the legacy quintessence model, v31.2 fully integrates the **Friedmann-Lemaître-Robertson-Walker (FLRW) metric tensor**. 

Expansion and contraction are applied directly to the spatial grid via a dynamic scale factor $a(t)$. Spatial coordinates are coupled to this metric, meaning particles do not just fly apart through static space; the comoving box itself mathematically stretches, generating a genuine **Hubble Flow**:
$$ a(t_{new}) = a(t) + (H \cdot a(t) \cdot \Delta t) $$
Dark Energy ($\Lambda$) is applied as a continuous acceleration to the Hubble parameter $H$ post-bounce.

## 6. Adiabatic Relaxation (New Innovation)
A massive challenge in the v17 model was "Initial Condition Shock"—particles spawning in high-density configurations would instantly detonate due to extreme initial forces. v31.2 introduces an **Adiabatic Damping** phase. 

For the first $t < 250$ epochs, a kinematic damping factor smoothly bleeds off explosive artificial energy:
$$ v_{new} = v_{raw} \cdot (0.95) $$
This allows the SPH fluid to gracefully accrete and settle into the Dark Matter gravity wells, establishing a stable, physical pre-crunch state before the manifold allows total collapse.

## 7. Local Spacetime Warping: The Non-Linear Lapse ($\alpha$)
The engine preserves relativistic time dilation through a modified Lapse Function ($\alpha$). Time flow is tied directly to the continuous FFT-PM density grid rather than isolated point-mass calculations. 

As local density ($\Phi_{FLRW}$) approaches extreme limits, $\alpha$ plummets. This mathematically freezes the kinematics of particles caught deep inside massive gravity wells, respecting relativistic limits without breaking the global solver:
$$ \alpha = \frac{2}{1 + \Phi_{FLRW}} $$

## 8. Deprecated Legacy Mechanics (v17 -> v31.2)
To achieve true physical continuum, several scripted features from the legacy engine have been cleanly phased out:
*   **Holographic Shift & String Tension:** $r_s = 2M + T_0 I_{horizon}$ is removed. Black holes are no longer treated as isolated 2D surfaces but as peak-density SPH fluid clusters.
*   **Entangled Spawning (ER=EPR):** Removed to prioritize thermodynamic fluid accuracy and global FFT gravity mapping.
*   **Instantaneous Mass Dumps:** The 50% core-mass dump has been replaced by the continuous, energy-conserving SPH shockwave.
