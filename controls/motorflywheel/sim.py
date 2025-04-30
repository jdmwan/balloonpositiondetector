import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Parameters
Kt = 0.05       # Nm/A
Ke = 0.05       # V·s/rad
R = 1.0         # Ohms
I = 0.0033      # kg·m²

# Desired speed
omega_target = 300  # rad/s

# Controller gains (example: P controller, adjust as you like)
Kp = 0.02
Ki = 0.0  # set non-zero for PI controller

# Time array
t = np.linspace(0, 1.5, 1500)
dt = t[1] - t[0]

# Storage
omega_arr = np.zeros_like(t)
i_arr = np.zeros_like(t)
v_arr = np.zeros_like(t)
error_sum = 0

# Initial state
omega = 0.0

# Simulation loop
for idx in range(1, len(t)):
    error = omega_target - omega
    error_sum += error * dt

    # --- INSERT CONTROLLER HERE ---
    V = Kp * error + Ki * error_sum
    V = np.clip(V, 0, 12)  # clip voltage to [0V, 12V] range
    # ------------------------------

    # Electrical: I = (V - Ke * omega) / R
    current = (V - Ke * omega) / R

    # Mechanical: T = Kt * I → I * domega/dt = T
    domega = (Kt * current) / I

    # Integrate angular velocity
    omega += domega * dt

    # Store data
    omega_arr[idx] = omega
    i_arr[idx] = current
    v_arr[idx] = V

# Plotting results
plt.figure(figsize=(10, 6))
plt.plot(t, omega_arr, label='Angular Velocity ω(t) [rad/s]')
plt.axhline(omega_target, color='r', linestyle='--', label='Target ω')
plt.xlabel('Time [s]')
plt.ylabel('Angular Velocity [rad/s]')
plt.title('Closed-loop Simulation with Placeholder Controller')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
