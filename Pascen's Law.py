import numpy as np

# Paschen Parameters for Air
A = 112.5  # (Pa·m)^-1
B = 2737.5  # V/(Pa·m)
gamma_se = 0.01  # Secondary emission coefficient
d_gap = 0.001  # 1 mm gap distance (m)

altitudes = np.arange(0, 6001, 500)
P0 = 101325.0  # Pa

print(
    f"{'Altitude (m)':<12} | {'Pressure (kPa)':<14} | {'1mm Breakdown Volts (V)':<24} | {'Arc Voltage Drop (%)':<22} | {'IEC Clearance Factor (ka)':<25}")
print("-" * 105)

# Baseline Breakdown at Sea Level
P_sea = P0
V_b_sea = (B * P_sea * d_gap) / (np.log(A * P_sea * d_gap) - np.log(np.log(1 + 1 / gamma_se)))

for z in altitudes:
    T_k = 288.15 - 0.0065 * z
    P = P0 * (1 - (0.0065 * z) / 288.15) ** 5.25588

    # Paschen Breakdown Voltage calculation
    Pd = P * d_gap
    V_b = (B * Pd) / (np.log(A * Pd) - np.log(np.log(1 + 1 / gamma_se)))

    voltage_drop_pct = ((V_b_sea - V_b) / V_b_sea) * 100.0
    ka = 101.325 / (P / 1000.0)  # Clearance multiplication factor

    print(f"{z:<12} | {P / 1000:<14.2f} | {V_b:<24.1f} | {voltage_drop_pct:<22.2f}% | {ka:<25.2f}x")