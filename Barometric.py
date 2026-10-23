import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ==============================================================================
# SECTION 1: PHYSICAL CONSTANTS & ALTITUDE BASELINE PARAMETERS
# ==============================================================================
P0 = 101325.0  # Sea-level atmospheric pressure (Pa)
T0_std = 288.15  # Standard sea-level temperature (K)
L_rate = 0.0065  # Temperature lapse rate in troposphere (K/m)
g = 9.81  # Acceleration due to gravity (m/s^2)
M_air = 0.0289644  # Molar mass of dry air (kg/mol)
R_gas = 8.31447  # Universal gas constant (J/(mol*K))
R_spec = 287.05  # Specific gas constant for air (J/(kg*K))
sigma = 5.67037e-8  # Stefan-Boltzmann constant (W/(m^2*K^4))

# Sutherland's Law Constants for Air Dynamic Viscosity
mu0 = 1.716e-5  # Reference viscosity (Pa*s) at T0_suth
T0_suth = 273.15  # Reference temperature (K)
S_suth = 110.4  # Sutherland constant (K)

# Target Operational Altitude Profile
altitudes = np.linspace(0, 6000, 100)  # Altitude array from 0m to 6000m
target_altitude = 4500.0  # Primary deployment altitude (Ladakh)


# ==============================================================================
# SECTION 2: ATMOSPHERIC & FLUID PROPERTY FUNCTIONS
# ==============================================================================
def get_atmospheric_properties(z, T_ambient_C=None):
    """
    Calculates atmospheric pressure, temperature, density, dynamic viscosity,
    and thermal conductivity of air at altitude z (meters).
    """
    # Pressure via Barometric Formula
    P = P0 * (1.0 - (L_rate * z) / T0_std) ** ((g * M_air) / (R_gas * L_rate))

    # Ambient Temperature (Standard Lapse Rate or Custom Cold Ground Temperature)
    if T_ambient_C is None:
        T_K = T0_std - (L_rate * z)
    else:
        T_K = T_ambient_C + 273.15

    # Air Density via Ideal Gas Law
    rho = P / (R_spec * T_K)

    # Dynamic Viscosity via Sutherland's Law
    mu = mu0 * ((T_K / T0_suth) ** 1.5) * ((T0_suth + S_suth) / (T_K + S_suth))

    # Air Thermal Conductivity Empirical Model
    k_f = 0.0242 * ((T_K / 273.15) ** 0.85)

    # Specific Heat Capacity of Air (J/kg*K)
    cp_air = 1005.0

    return P, T_K, rho, mu, k_f, cp_air


# ==============================================================================
# SECTION 3: CONVECTIVE HEAT TRANSFER SCALING (FORCED & NATURAL)
# ==============================================================================
def calculate_convective_coefficients(z, V_air=5.0, L_char=0.15, Delta_T=35.0, T_amb_C=-30.0):
    """
    Calculates forced and natural convective heat transfer coefficients (h)
    and fluid dimensionless numbers at altitude z.
    """
    P, T_K, rho, mu, k_f, cp_air = get_atmospheric_properties(z, T_amb_C)

    # Kinematic Viscosity
    nu = mu / rho

    # Prandtl Number
    Pr = (cp_air * mu) / k_f

    # --- FORCED CONVECTION (External Flow Over Heatsink Fins) ---
    Re = (rho * V_air * L_char) / mu

    # Laminar / Mixed Boundary Layer Correlation for Flat Plate / Heatsink
    if Re < 5e5:
        Nu_forced = 0.664 * (Re ** 0.5) * (Pr ** (1.0 / 3.0))
    else:
        Nu_forced = (0.037 * (Re ** 0.8) - 871.0) * (Pr ** (1.0 / 3.0))

    h_forced = (Nu_forced * k_f) / L_char

    # --- NATURAL CONVECTION (Inside Enclosure Volumetric Air) ---
    beta = 1.0 / T_K  # Thermal expansion coefficient
    Gr = (g * beta * Delta_T * (L_char ** 3) * (rho ** 2)) / (mu ** 2)
    Ra = Gr * Pr

    # Enclosure Natural Convection Correlation
    Nu_natural = 0.54 * (Ra ** 0.25) if Ra > 0 else 1.0
    h_natural = (Nu_natural * k_f) / L_char

    return Re, Ra, h_forced, h_natural, P, rho


# ==============================================================================
# SECTION 4: TRANSIENT SYSTEM ENERGY BALANCE SOLVER
# ==============================================================================
def system_energy_balance(t, T_array, Q_gen, A_surf, m_sys, cp_sys, z, T_amb_C, emissivity):
    """
    Differential equation governing system enclosure transient temperature response:
    m * cp * (dT/dt) = Q_gen - Q_conv - Q_rad
    """
    T_sys = T_array[0]
    P, T_amb_K, rho, mu, k_f, cp_air = get_atmospheric_properties(z, T_amb_C)

    # Effective Sky Temperature for Radiation
    T_dp = T_amb_C - 5.0  # Approximate dew point
    T_sky = T_amb_K * ((0.711 + 0.0056 * T_dp + 0.000073 * (T_dp ** 2)) ** 0.25)

    # Convective Heat Loss (Using calculated altitude forced convection coefficient)
    _, _, h_forced, _, _, _ = calculate_convective_coefficients(z, V_air=3.0, L_char=0.2,
                                                                Delta_T=max(1.0, T_sys - T_amb_K),
                                                                T_amb_C=T_amb_C)

    Q_conv = h_forced * A_surf * (T_sys - T_amb_K)
    Q_rad = emissivity * sigma * A_surf * ((T_sys ** 4) - (T_sky ** 4))

    dT_dt = (Q_gen - Q_conv - Q_rad) / (m_sys * cp_sys)
    return [dT_dt]


# ==============================================================================
# SECTION 5: MAIN EXECUTION, DATA EXPORT & PLOTTING
# ==============================================================================
if __name__ == "__main__":
    print("=================================================================")
    print(" MEMBER 1: HIGH-ALTITUDE THERMAL & BAROMETRIC MODELING SYSTEM ")
    print("=================================================================")

    # 1. Compute Altitude Curves
    results_list = []
    for z in altitudes:
        Re, Ra, h_f, h_n, P_val, rho_val = calculate_convective_coefficients(z, V_air=4.0, T_amb_C=-30.0)
        results_list.append({
            'Altitude_m': z,
            'Pressure_kPa': P_val / 1000.0,
            'Density_kg_m3': rho_val,
            'Re_Number': Re,
            'Ra_Number': Ra,
            'h_forced_W_m2K': h_f,
            'h_natural_W_m2K': h_n
        })

    df_alt = pd.DataFrame(results_list)

    # 2. Extract Key Baseline Comparison (Sea Level vs Ladakh 4500m)
    sl_data = df_alt.iloc[0]
    ladakh_data = df_alt.iloc[(df_alt['Altitude_m'] - target_altitude).abs().idxmin()]

    h_degradation_ratio = (ladakh_data['h_forced_W_m2K'] / sl_data['h_forced_W_m2K']) * 100.0
    rho_degradation_ratio = (ladakh_data['Density_kg_m3'] / sl_data['Density_kg_m3']) * 100.0

    print(f"\n--- ATMOSPHERIC COMPARISON TABLE ---")
    print(f"Sea Level (0m) Pressure       : {sl_data['Pressure_kPa']:.2f} kPa")
    print(f"Ladakh (4500m) Pressure       : {ladakh_data['Pressure_kPa']:.2f} kPa")
    print(f"Air Density Degradation       : {rho_degradation_ratio:.2f}% of Sea Level value")
    print(
        f"Forced Convection Coefficient : Sea Level = {sl_data['h_forced_W_m2K']:.2f} W/m²K | Ladakh = {ladakh_data['h_forced_W_m2K']:.2f} W/m²K")
    print(f"Convective Cooling Capacity   : {h_degradation_ratio:.2f}% retained in thin air\n")

    # 3. Solve Transient Temperature Response at 4500m
    Q_total_gen = 310.0  # Total heat generation: RF (150W) + Motors (80W) + CPU (50W) + Battery (30W)
    A_enclosure = 0.65  # Enclosure surface area (m^2)
    m_chassis = 10.0  # Mass of aluminum alloy chassis (kg)
    cp_chassis = 900.0  # Specific heat capacity of aluminum (J/kg*K)
    T_init_K = 243.15  # Initial cold start temperature (-30 deg C)

    t_span = (0, 3600)  # 1-hour simulation
    t_eval = np.linspace(0, 3600, 500)

    sol_4500m = solve_ivp(system_energy_balance, t_span, [T_init_K], t_eval=t_eval,
                          args=(Q_total_gen, A_enclosure, m_chassis, cp_chassis, 4500.0, -30.0, 0.85))

    sol_0m = solve_ivp(system_energy_balance, t_span, [T_init_K], t_eval=t_eval,
                       args=(Q_total_gen, A_enclosure, m_chassis, cp_chassis, 0.0, -30.0, 0.85))

    # 4. Generate Handoff Data Tables for Team Members
    handoff_df = pd.DataFrame({
        'Parameter': ['Atmospheric Pressure (kPa)', 'Air Density (kg/m³)', 'Dynamic Viscosity (Pa·s)',
                      'Air Thermal Conductivity (W/m·K)', 'Forced Convective h (W/m²K)',
                      'Steady-State Enclosure Temp (°C)'],
        'Sea_Level_0m': [101.33, 1.225, 1.78e-5, 0.0242, sl_data['h_forced_W_m2K'], sol_0m.y[0][-1] - 273.15],
        'Ladakh_4500m': [ladakh_data['Pressure_kPa'], ladakh_data['Density_kg_m3'], 1.55e-5, 0.0218,
                         ladakh_data['h_forced_W_m2K'], sol_4500m.y[0][-1] - 273.15]
    })

    handoff_df.to_csv("Member1_Handoff_Parameters.csv", index=False)
    print("--- HANDOFF DATA TABLE EXPORTED TO 'Member1_Handoff_Parameters.csv' ---")
    print(handoff_df.to_string(index=False))

    # 5. Visualizing Mathematical Results
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Pressure & Density vs Altitude
    axs[0, 0].plot(df_alt['Altitude_m'], df_alt['Pressure_kPa'], 'b-', label='Pressure (kPa)')
    axs[0, 0].set_ylabel('Pressure (kPa)', color='b')
    ax_rho = axs[0, 0].twinx()
    ax_rho.plot(df_alt['Altitude_m'], df_alt['Density_kg_m3'], 'r--', label='Density (kg/m³)')
    ax_rho.set_ylabel('Air Density (kg/m³)', color='r')
    axs[0, 0].set_title('Atmospheric Rarefaction vs. Altitude')
    axs[0, 0].grid(True)

    # Plot 2: Forced Convection Coefficient vs Altitude
    axs[0, 1].plot(df_alt['Altitude_m'], df_alt['h_forced_W_m2K'], 'g-', linewidth=2)
    axs[0, 1].axvline(4500, color='k', linestyle=':', label='Ladakh Deployment (4500m)')
    axs[0, 1].set_xlabel('Altitude (m)')
    axs[0, 1].set_ylabel('Forced Convective h (W/m²K)')
    axs[0, 1].set_title('Convective Cooling Capacity Degradation')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Plot 3: Transient Temperature Curve (Sea Level vs 4500m)
    axs[1, 0].plot(sol_0m.t / 60.0, sol_0m.y[0] - 273.15, 'b--', label='Sea Level (0m)')
    axs[1, 0].plot(sol_4500m.t / 60.0, sol_4500m.y[0] - 273.15, 'r-', linewidth=2, label='Ladakh (4500m)')
    axs[1, 0].set_xlabel('Time (minutes)')
    axs[1, 0].set_ylabel('Enclosure Temperature (°C)')
    axs[1, 0].set_title('Transient Overheating Behavior in Thin Air')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Plot 4: Heat Loss Component Comparison at 4500m
    axs[1, 1].bar(['Forced Conv (0m)', 'Forced Conv (4500m)'],
                  [sl_data['h_forced_W_m2K'], ladakh_data['h_forced_W_m2K']], color=['blue', 'red'])
    axs[1, 1].set_ylabel('Convective Coefficient h (W/m²K)')
    axs[1, 1].set_title('Cooling Capacity Loss Ratio')
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig("Member1_HighAltitude_Thermal_Analysis.png", dpi=300)
    print("\n--- PLOTS GENERATED & SAVED AS 'Member1_HighAltitude_Thermal_Analysis.png' ---")
    plt.show()