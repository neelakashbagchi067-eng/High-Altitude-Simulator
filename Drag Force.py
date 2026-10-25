import numpy as np

# Atmospheric and Geometric Constants
g = 9.80665  # Gravity (m/s^2)
R_spec = 287.058  # Specific gas constant for air (J/kg·K)
L = 0.0065  # Temperature lapse rate (K/m)
T0 = 288.15  # Sea level standard temperature (K)
P0 = 101325.0  # Sea level standard pressure (Pa)

# System Parameters
A = 0.85  # Frontal area (m^2)
Cd = 1.2  # Drag coefficient
V_steady = 20.0  # Steady wind speed (m/s)
V_gust = 30.0  # Gust wind speed (m/s)

altitudes = np.arange(0, 6001, 500)  # 0 to 6000 m in 500 m steps

print(
    f"{'Altitude (m)':<12} | {'Temp (°C)':<10} | {'Press (kPa)':<12} | {'Density (kg/m³)':<16} | {'Steady Drag (N)':<16} | {'Gust Drag (N)':<15} | {'Delta P (kPa)':<12}")
print("-" * 105)

for z in altitudes:
    T_k = T0 - L * z
    T_c = T_k - 273.15
    P = P0 * (1 - (L * z) / T0) ** (g / (R_spec * L))
    rho = P / (R_spec * T_k)

    # Drag Forces
    Fd_steady = 0.5 * rho * Cd * A * (V_steady ** 2)
    Fd_gust = 0.5 * rho * Cd * A * (V_gust ** 2)

    # Differential Pressure vs Sea Level (Outward Enclosure Stress)
    delta_P = (P0 - P) / 1000.0

    print(
        f"{z:<12} | {T_c:<10.1f} | {P / 1000:<12.2f} | {rho:<16.4f} | {Fd_steady:<16.2f} | {Fd_gust:<15.2f} | {delta_P:<12.2f}")