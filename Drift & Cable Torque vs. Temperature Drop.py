import pandas as pd
import numpy as np

# Structural and Thermal Parameters
alpha_al = 23.0e-6  # CTE of Aluminum 6061 (1/K)
alpha_cfrp = 1.5e-6  # CTE of Carbon Fiber Composite (1/K)
L_arm = 0.45  # Structural arm length (m)
R_pivot = 0.12  # Pivot radius (m)

# Cable Rigidity Parameters
E_room = 0.5  # Baseline Young's modulus at +20°C (GPa)
I_area = 1.2e-10  # Cable bundle area moment of inertia (m^4)
kappa = 12.0  # Bending curvature (rad/m)

temperatures = np.arange(25, -45, -5)  # From +25°C down to -40°C

data = []
for T in temperatures:
    delta_T = 25.0 - T  # Relative to room temp (+25°C)

    # Exponential increase of polymer modulus at sub-zero temperatures
    E_cab = E_room * np.exp(0.025 * max(0, -T))
    T_cab = E_cab * 1e9 * I_area * kappa

    # Thermo-elastic angular drift calculation (converted to micro-radians)
    drift_rad = (alpha_al - alpha_cfrp) * delta_T * (L_arm / R_pivot)
    drift_microrad = drift_rad * 1e6
    miss_dist_4km = (drift_microrad * 1e-6) * 4000.0  # miss distance at 4km in meters

    data.append({
        "Ambient Temp (°C)": T,
        "Temperature Drop Delta T (K)": delta_T,
        "Cable Young's Modulus E_cab (GPa)": round(E_cab, 3),
        "Cable Bending Torque T_cable (N·m)": round(T_cab, 4),
        "Thermo-Elastic Angular Drift (µrad)": round(drift_microrad, 2),
        "Target Miss Distance at 4km (m)": round(miss_dist_4km, 2)
    })

df = pd.DataFrame(data)
file_path = "SIH4_Pointing_Drift_and_Cable_Torque_Results.csv"
df.to_csv(file_path, index=False)
print(f"CSV file generated: {file_path}")