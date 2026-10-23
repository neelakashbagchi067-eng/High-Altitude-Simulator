import numpy as np


def calculate_atmosphere_properties(altitudes):
    """
    Calculates atmospheric density, dynamic viscosity, and convective scaling
    across a given array of altitudes (meters).
    """
    # Constants
    T0 = 288.15  # Sea-level standard temperature (K)
    P0 = 101.325  # Sea-level standard pressure (kPa)
    rho0 = 1.225  # Sea-level standard density (kg/m^3)
    g = 9.80665  # Acceleration due to gravity (m/s^2)
    R = 287.05  # Specific gas constant for dry air (J/(kg·K))
    L = 0.0065  # Temperature lapse rate (K/m)

    # Sutherland's Law constants for dynamic viscosity
    mu_ref = 1.716e-5  # Reference viscosity (Pa·s)
    T_ref = 273.15  # Reference temperature (K)
    S = 110.4  # Sutherland's temperature (K)

    results = []

    print(
        f"{'Altitude (m)':<15} {'Temp (K)':<12} {'Pressure (kPa)':<15} {'Density (kg/m^3)':<18} {'Viscosity (Pa·s)':<18} {'Convective Scale':<18}")
    print("-" * 96)

    for z in altitudes:
        # 1. Temperature at altitude z (Troposphere model)
        T = T0 - L * z

        # 2. Pressure at altitude z
        P = P0 * (T / T0) ** (g / (R * L))

        # 3. Density at altitude z (Ideal Gas Law)
        rho = P / (R * T)

        # 4. Dynamic Viscosity via Sutherland's Law
        mu = mu_ref * ((T / T_ref) ** 1.5) * ((T_ref + S) / (T + S))

        # 5. Convective coefficient scaling factor
        # Typically, forced convection scales with (rho^0.5 * P^0.5) or Reynolds/Prandtl combinations.
        # Here we use a standard generalized scaling proportional to sqrt(rho * P) normalized to sea-level.
        h_scale = np.sqrt((rho * P) / (rho0 * P0))

        results.append((z, T, P, rho, mu, h_scale))

        print(f"{z:<15.1f} {T:<12.2f} {P:<15.2f} {rho:<18.4f} {mu:<18.4e} {h_scale:<18.4f}")

    return results


if __name__ == "__main__":
    # Define altitudes from 0 m to 6000 m with 1000 m increments
    altitudes_m = np.linspace(0, 6000, 7)
    calculate_atmosphere_properties(altitudes_m)