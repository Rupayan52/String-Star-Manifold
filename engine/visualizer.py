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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def render_manifold_snapshot(pos, masses, stored_microstates, dm_nodes, filename="simulation_output.png"):
    """
    Renders a 2D projection of the 3D String-Star Manifold, matching 
    the exact aesthetic required for the Bandyopadhyay-Cycle telemetry.
    """
    
    # 1. THE VOID AESTHETIC (Dark Mode Setup)
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    # Remove all axes, grids, and borders for that clean "Dashboard" look
    ax.axis('off')
    
    # 2. RENDER DM NODES (The Weavers - Teal 'X')
    # Assuming dm_nodes is an Nx3 array, we plot X and Y
    ax.scatter(dm_nodes[:, 0], dm_nodes[:, 1], 
               c='#00BFFF', # Deep cyan/teal
               marker='x', 
               s=100, 
               linewidths=1.5, 
               alpha=0.8,
               zorder=2)
               
    # 3. RENDER BULK & FUZZBALLS
    active = [i for i in range(len(masses)) if (masses[i] + stored_microstates[i]) > 0]
    
    for i in active:
        x, y = pos[i, 0], pos[i, 1]
        
        # Holographic Phase (Fuzzballs - Purple Glow)
        if stored_microstates[i] > 0:
            # The "Social Ghost" Bloom Effect: Layered transparent circles
            ax.scatter(x, y, c='#FF00FF', marker='o', s=60, alpha=0.2, zorder=3)
            ax.scatter(x, y, c='#FF00FF', marker='o', s=30, alpha=0.4, zorder=3)
            ax.scatter(x, y, c='#E0B0FF', marker='o', s=10, alpha=1.0, zorder=4) # Core
            
        # Kinetic Phase (Bulk Matter - Sharp White)
        elif masses[i] > 0:
            # Size scales slightly with mass
            size = min(30, 5 + (masses[i] * 2))
            ax.scatter(x, y, c='white', marker='o', s=size, alpha=0.9, zorder=3)

    # 4. THE TYPOGRAPHY (Monospace System Font)
    title_text = "THE STRING-STAR MANIFOLD\nCycle: Bandyopadhyay Recombination"
    plt.text(0.5, 0.95, title_text, 
             horizontalalignment='center', 
             verticalalignment='center', 
             transform=ax.transAxes, 
             color='white', 
             fontsize=14, 
             fontname='monospace',
             letterspacing=1.2)

    # 5. THE LEGEND (Bottom Left, Bordered)
    legend_elements = [
        Line2D([0], [0], marker='x', color='black', label='DM Nodes (Weavers)', 
               markerfacecolor='#00BFFF', markeredgecolor='#00BFFF', markersize=8),
        Line2D([0], [0], marker='o', color='black', label='Bulk Matter (Kinetic Phase)', 
               markerfacecolor='white', markersize=6),
        Line2D([0], [0], marker='o', color='black', label='Fuzzball (Holographic Phase)', 
               markerfacecolor='#FF00FF', markersize=6)
    ]
    
    legend = ax.legend(handles=legend_elements, loc='lower left', 
                       frameon=True, facecolor='black', edgecolor='grey',
                       prop={'family': 'monospace', 'size': 8})
    for text in legend.get_texts():
        text.set_color("white")

    # 6. EXPORT
    plt.tight_layout()
    plt.savefig(filename, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    print(f">>> Manifold Snapshot Rendered: {filename}")
    plt.close()

# --- Example Execution Block (For testing without running the full engine) ---
if __name__ == "__main__":
    print("Testing Visualizer Engine...")
    
    # Generate mock data just to test the aesthetic
    mock_dm = np.array([[120, 120, 0], [-120, -120, 0], [120, -120, 0], [-120, 120, 0]])
    mock_pos = np.random.uniform(-150, 150, (100, 3))
    mock_masses = np.random.choice([0, 1, 5], 100)
    mock_microstates = np.random.choice([0, 0, 10], 100) # Sparsely populate Fuzzballs
    
    render_manifold_snapshot(mock_pos, mock_masses, mock_microstates, mock_dm, "test_render.png")
