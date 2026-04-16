import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

class FERDA_System:
    def __init__(self, init_x, init_y, init_r):
        self.x = np.array([[init_x], [init_y], [init_r]], dtype=float)
        self.P = np.eye(3) * 15.0 
        self.Q = np.diag([0.2, 0.2, 0.5]) 
        self.R = np.diag([5.0, 5.0, 10.0]) 

        self.history_r = [init_r]
        self.timestamps = [0]
        self.current_time = 0

        plt.ion()
        self.fig, (self.ax_map, self.ax_growth) = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={'width_ratios': [2, 1]})
        
        # --- MAP SETUP ---
        self.ax_map.set_xlim(-50, 500)
        self.ax_map.set_ylim(-50, 500)
        
        self.fire_patch = Ellipse((init_x, init_y), init_r*2, init_r*2, color='red', alpha=0.6, label='Current Front', zorder=5)
        self.ax_map.add_patch(self.fire_patch)
        self.meas_marker, = self.ax_map.plot([], [], 'kx', markersize=10, label='Drone Detection')
        self.quiver = self.ax_map.quiver(450, 450, 0, 0, color='blue', scale=50)
        
        # UI REQUEST: Legend to Bottom Right
        self.ax_map.legend(loc='lower right', frameon=True, shadow=True)
        
        # UI REQUEST: Confidence/Time to Top Left (with background box)
        self.conf_text = self.ax_map.text(0.02, 0.95, '', transform=self.ax_map.transAxes, 
                                         fontsize=11, fontweight='bold', va='top',
                                         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

        # --- GROWTH CHART SETUP ---
        self.ax_growth.set_title("Radius Growth History")
        self.growth_line, = self.ax_growth.plot([], [], 'r-', marker='o', markersize=4)
        self.ax_growth.grid(True, alpha=0.3)

    def predict(self, dt, wind_v, wind_dir, slope, terrain_val, base_ros=0.2):
        phi_s = 5.275 * (np.tan(np.radians(slope))**2)
        phi_w = 0.1 * (wind_v**2)
        total_ros = base_ros * (1 + phi_w + phi_s) * terrain_val
        
        theta = np.radians(wind_dir)
        self.x[0,0] += total_ros * np.cos(theta) * dt
        self.x[1,0] += total_ros * np.sin(theta) * dt
        self.x[2,0] += total_ros * dt
        
        self.P = self.P + self.Q
        self.quiver.set_UVC(np.cos(theta)*wind_v, np.sin(theta)*wind_v)
        self.current_time += dt

    def update(self, z):
        z_meas = np.array(z).reshape(3, 1)
        H = np.eye(3)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z_meas - H @ self.x)
        self.P = (np.eye(3) - K @ H) @ self.P

    def render(self):
        self.fire_patch.center = (self.x[0,0], self.x[1,0])
        self.fire_patch.width = self.x[2,0] * 2
        self.fire_patch.height = self.x[2,0] * 2
        
        # Leave a ghost trail
        history_circle = Ellipse((self.x[0,0], self.x[1,0]), self.x[2,0]*2, self.x[2,0]*2, 
                                 edgecolor='gray', facecolor='none', alpha=0.15, linestyle=':')
        self.ax_map.add_patch(history_circle)
        
        self.history_r.append(self.x[2,0])
        self.timestamps.append(self.current_time)
        self.growth_line.set_data(self.timestamps, self.history_r)
        self.ax_growth.relim()
        self.ax_growth.autoscale_view()
        
        # Calculate System Confidence
        trace = np.trace(self.P)
        confidence = max(0, min(100, (20 / (trace + 0.1)) * 100))
        color = 'darkgreen' if confidence > 70 else 'darkorange' if confidence > 40 else 'red'
        
        self.conf_text.set_text(f"Confidence: {confidence:.1f}%\nElapsed: {self.current_time}s")
        self.conf_text.set_color(color)
        
        self.ax_map.set_title(f"State: X:{self.x[0,0]:.1f} Y:{self.x[1,0]:.1f} R:{self.x[2,0]:.1f}")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.1)

def get_valid_float(prompt):
    while True:
        try: return float(input(prompt))
        except ValueError: print("  [!] Numeric value required.")

# --- EXECUTION ---
ferda = FERDA_System(0, 0, 5)

try:
    while True:
        print("\n--- [ENVIRONMENT CONFIG] ---")
        w_v = get_valid_float("  Wind Speed: ")
        w_d = get_valid_float("  Wind Direction: ")
        s_p = get_valid_float("  Slope Angle: ")
        t_c = get_valid_float("  Terrain Multiplier: ")

        print("\n--- [LIVE SIMULATION] ---")
        while True:
            ferda.predict(dt=10, wind_v=w_v, wind_dir=w_d, slope=s_p, terrain_val=t_c)
            ferda.render()
            
            obs_input = input(f" Time {ferda.current_time}s (x,y,r or 'env') -> ").lower().strip()
            if obs_input == 'env': break
            if obs_input:
                try:
                    z = [float(i) for i in obs_input.split(",")]
                    if len(z) == 3:
                        ferda.update(z)
                        ferda.meas_marker.set_data([z[0]], [z[1]])
                        ferda.render()
                    else: print("  [!] Error: Use x,y,r")
                except ValueError: print("  [!] Error: Numeric input only")
except KeyboardInterrupt:
    plt.ioff()
    plt.show()