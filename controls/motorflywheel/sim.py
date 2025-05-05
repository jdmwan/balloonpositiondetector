import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Parameters
Kt = 0.05       # Nm/A
Ke = 0.05       # V·s/rad
R = 1.0         # Ohms
I = 0.0088   # kg·m²

# Desired speed
omega_target = 200  # rad/s old 300

# Controller gains (example: P controller, adjust as you like)
Kp = 40
Ki = 0  # set non-zero for PI controller
Kd = 0

# Time array
t = np.linspace(0, 3, 1500)
dt = t[1] - t[0]

# Storage
omega_arr = np.zeros_like(t)
i_arr = np.zeros_like(t)
v_arr = np.zeros_like(t)
error_sum = 0
error_deriv = 0
prev_error = 0
# Initial state
omega = 0.0

# Simulation loop
for idx in range(1, len(t)):
    error = omega_target - omega
    error_sum += error * dt
    error_deriv = (error - prev_error)/dt


    # --- INSERT CONTROLLER HERE ---
    V = Kp * error + Ki * error_sum + Kd *error_deriv
    # V = np.clip(V, 0, 100)  # clip voltage to [0V, 12V] range
    
    
    # ------------------------------

    # Electrical: I = (V - Ke * omega) / R
    current = (V - Ke * omega) / R

    # Mechanical: T = Kt * I → I * domega/dt = T
    domega = (Kt * current*2) / I

    # Integrate angular velocity
    omega += domega * dt

    # Store data
    omega_arr[idx] = omega
    i_arr[idx] = current
    v_arr[idx] = V
    prev_error = error

# Plotting results
plt.figure(figsize=(10, 6))
plt.plot(t, omega_arr, label='Angular Velocity ω(t) [rad/s]')
# plt.plot(t, v_arr, label='voltage [V]')
print(omega_arr)
plt.axhline(omega_target, color='r', linestyle='--', label='Target ω')
plt.xlabel('Time [s]')
plt.ylabel('Angular Velocity [rad/s]')
plt.title('Closed-loop Simulation with Placeholder Controller')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
