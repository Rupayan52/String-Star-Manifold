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
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# We will animate 50 Epochs
num_frames = 50

def update(frame):
    ax.clear()
    
    # Set the dynamic axis limits based on the Scale Factor 'a' from your telemetry
    # This visually shows the expansion of the Universe
    limit = 200 * (1.0 + frame * 0.01) 
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    
    # Hide axes for a cinematic "Space" feel
    ax.set_axis_off()
    
    # --- 2. PLOTTING THE PHYSICS ---
    # In a real run, these would pull from your 'pos' and 'masses' arrays
    # Here we simulate the visual behavior for the Movie:
    n_particles = 40
    current_pos = np.random.normal(0, limit/4, (n_particles, 3))
    current_alphas = np.linspace(0.1, 1.0, n_particles) # 0.1 = warped/frozen, 1.0 = flat
    current_sizes = np.random.randint(10, 100, n_particles)
    
    # Scatter plot: 
    # Color = Lapse Alpha (Purple for warped, Yellow for flat)
    # Size = Mass/Horizon Bits
    scat = ax.scatter(current_pos[:, 0], current_pos[:, 1], current_pos[:, 2], 
                      c=current_alphas, cmap='plasma', 
                      s=current_sizes, edgecolors='white', linewidth=0.5, alpha=0.8)
    
    # Add a faint glow or "Grid" to show expansion
    ax.set_title(f'OMEGA-FLRW COSMOLOGICAL ENGINE | EPOCH {frame+1}', color='white', fontsize=14, pad=20)
    
    return scat,

# --- 3. ENCODE THE MOVIE ---
print("Directing Movie: Rendering Spacetime Warping...")
ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=100, blit=False)

# This exports the animation to an HTML5 video player inside your Colab Cell
from IPython.display import HTML
HTML(ani.to_html5_video())
