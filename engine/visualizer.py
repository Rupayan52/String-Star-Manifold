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

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# --- 1. SETUP THE CINEMATIC CANVAS ---
plt.style.use('dark_background')
fig = plt.figure(figsize=(14, 10)) # Expanded canvas for UI elements
ax = fig.add_subplot(111, projection='3d')

# Create a dedicated axis for the colorbar so it doesn't glitch during animation
cax = fig.add_axes([0.88, 0.25, 0.02, 0.5]) 

# We will animate 50 Epochs
num_frames = 50

def update(frame):
    ax.clear()
    
    # Set the dynamic axis limits based on the Scale Factor 'a'
    limit = 200 * (1.0 + frame * 0.01) 
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.set_axis_off() # Keeps the deep space aesthetic
    
    # --- 2. PLOTTING THE PHYSICS ---
    # Generating mock spatial data for the visualization structure
    n_particles = 40
    current_pos = np.random.normal(0, limit/4, (n_particles, 3))
    current_alphas = np.linspace(0.1, 1.0, n_particles) 
    current_sizes = np.random.randint(10, 150, n_particles)
    
    # Upgraded Scatter: Added cyan edgeglow for a more captivating look
    scat = ax.scatter(current_pos[:, 0], current_pos[:, 1], current_pos[:, 2], 
                      c=current_alphas, cmap='plasma', 
                      s=current_sizes, edgecolors='cyan', linewidth=0.5, alpha=0.85)
    
    # --- 3. THE HUD & FOOTNOTES ---
    # Main Title
    ax.set_title(f'OMEGA-FLRW COSMOLOGICAL ENGINE | EPOCH {frame+1}', 
                 color='white', fontsize=16, pad=20, weight='bold', family='monospace')
    
    # Colorbar (Lapse Function) - Rendered only once to attach to the UI
    if frame == 0:
        cb = fig.colorbar(scat, cax=cax)
        cb.set_label('Lapse Function ($\\alpha$)', color='cyan', fontsize=12, weight='bold', family='monospace')
        cb.ax.yaxis.set_tick_params(color='white')
        cb.outline.set_edgecolor('cyan')
        plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white', family='monospace')
        cb.set_ticks([0.1, 0.5, 1.0])
        cb.set_ticklabels(['0.1 (Frozen)', '0.5 (Warped)', '1.0 (Flat)'])

    # The Physics Legend (Footnotes)
    legend_text = (
        "PHYSICS LEGEND:\n"
        "-------------------------------------\n"
        "• Color (Plasma): Local Spacetime Warping.\n"
        "   - Bright Yellow: Flat Space (Time flows normally)\n"
        "   - Deep Purple: Gravity Well (Time freezes)\n\n"
        "• Sphere Volume: Holographic Mass.\n"
        "   - Scales with String Tension ($T_0$) & Horizon Bits\n\n"
        "• Canvas Expansion: The Hubble Flow.\n"
        "   - Vacuum density driving metric expansion ($H$)"
    )
    
    # Position the legend in the bottom left corner
    ax.text2D(0.02, 0.02, legend_text, transform=ax.transAxes, 
              color='lightgray', fontsize=10, family='monospace',
              bbox=dict(facecolor='black', alpha=0.7, edgecolor='cyan', boxstyle='round,pad=0.5'))
    
    # Dynamic Telemetry Readout (Top Left)
    hud_text = (
        f"TELEMETRY DATA\n"
        f"-----------------\n"
        f"Scale Factor ($a$): {(1.0 + frame * 0.01):.2f}\n"
        f"Sys Integrity: 1000/1000"
    )
    ax.text2D(0.02, 0.85, hud_text, transform=ax.transAxes,
              color='cyan', fontsize=11, family='monospace', weight='bold',
              bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

    return scat,

# --- 4. ENCODE THE MOVIE ---
print("Directing Movie: Rendering Cinematic Spacetime HUD with Legends...")
ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=100, blit=False)

from IPython.display import HTML
HTML(ani.to_html5_video())
