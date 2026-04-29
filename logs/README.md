# Simulation Logs: The Bandyopadhyay-Cycle

This directory contains the raw telemetry data proving the **100% Unitary Information Conservation** of the String-Star Manifold. 

## Hardware & Environment Specifications
* **Architecture:** Google TPU V5 Lite
* **Framework:** JAX (Just-In-Time compiled linear algebra)
* **Precision:** Float32 for kinematics, Int32 for discrete microstate tracking

## Initial Conditions (Omega Run)
* **Maximum Bodies (Ghost Buffer):** 500
* **Dark Matter Weavers (Nodes):** 8
* **Initial Vacuum Pool:** 1000 Bits
* **Target Epochs:** 100

## Data Dictionary
The `omega_run_100_epochs.csv` file tracks the state of the universe at the end of each epoch. 

* `Epoch`: The current time-step of the simulation.
* `Bulk_Matter`: Information manifest as kinetic physical mass ($I_{bulk}$).
* `Horizon_Bits`: Information trapped on Holographic Fuzzball surfaces ($I_{horizon}$).
* `Vacuum_Pool`: Ambient radiation waiting for Nodal Recombination ($I_{vacuum}$).
* `Total_Integrity`: The absolute sum of all three phases. **Must equal exactly 1000.**
* `Notable_Events`: Major phase transitions (e.g., Holographic Shifts, Hawking Recycling).
* **Hubble_H:** The instantaneous expansion rate of the universe calculated via the Unified Friedmann Equation.
* **Scale_Factor_a:** The global multiplier ($a(t)$) for the universe's volume, tracking the physical stretching of the spatial hash grid.
* **Avg_Lapse_Alpha:** The average value of the Lapse Function ($\alpha$) across all active particles. Tracks Gravitational Time Dilation (1.0 = flat spacetime, approaches 0.1 = time "freezing" near Fuzzball horizons).
