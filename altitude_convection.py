"""
Altitude-dependent air properties and convection coefficients
for the high-altitude anti-drone thermal study (SIH 4).

Contents
--------
1. ISA atmosphere (pressure vs altitude) + ideal-gas density at the ACTUAL ambient temperature
2. Air properties (mu, k, cp, Pr, nu, alpha) at film temperature
3. Natural convection  : vertical plate (Churchill-Chu)
4. Forced convection   : (a) wind over a flat plate, (b) fan-driven flow in a duct (Dittus-Boelter / laminar)
5. Surface-temperature solver for a heat-dissipating enclosure (convection + linearised radiation)
6. Sea level vs. altitude comparison, tables and plots

Run:  python altitude_convection.py
"""

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
G = 9.80665          # m/s^2
R_AIR = 287.05       # J/(kg K)
SIGMA = 5.670374e-8  # W/(m^2 K^4)
CP_AIR = 1006.0      # J/(kg K)  (almost constant over -40..+60 C)

# ISA sea-level reference
T0_ISA = 288.15      # K
P0_ISA = 101325.0    # Pa
LAPSE = 0.0065       # K/m  (troposphere)


# ----------------------------------------------------------------------------
# 1. Atmosphere
# ----------------------------------------------------------------------------
def isa_temperature(h):
    """ISA temperature [K] at altitude h [m] (valid up to 11 km)."""
    return T0_ISA - LAPSE * h


def isa_pressure(h):
    """ISA pressure [Pa] at altitude h [m] (valid up to 11 km)."""
    return P0_ISA * (1.0 - LAPSE * h / T0_ISA) ** (G / (R_AIR * LAPSE))


def air_density(h, T_amb=None):
    """
    Air density [kg/m^3] at altitude h [m].
    Pressure comes from ISA; temperature is T_amb [K] if given (e.g. a real
    Ladakh winter night at 4500 m), otherwise the ISA temperature.
    """
    T = isa_temperature(h) if T_amb is None else T_amb
    return isa_pressure(h) / (R_AIR * T)


# ----------------------------------------------------------------------------
# 2. Air properties (pressure-independent except density)
# ----------------------------------------------------------------------------
def air_props(T, h):
    """
    Air properties at temperature T [K] and altitude h [m].
    mu, k via Sutherland's law; rho from ideal gas at ISA pressure for altitude h.
    Returns dict with rho, mu, k, cp, nu, alpha, Pr, beta.
    """
    mu = 1.716e-5 * (T / 273.15) ** 1.5 * (273.15 + 110.4) / (T + 110.4)
    k = 0.0241 * (T / 273.15) ** 1.5 * (273.15 + 194.0) / (T + 194.0)
    rho = isa_pressure(h) / (R_AIR * T)
    nu = mu / rho
    alpha = k / (rho * CP_AIR)
    Pr = mu * CP_AIR / k
    return dict(rho=rho, mu=mu, k=k, cp=CP_AIR, nu=nu, alpha=alpha, Pr=Pr, beta=1.0 / T)


# ----------------------------------------------------------------------------
# 3. Natural convection - vertical plate / enclosure wall (Churchill-Chu, all Ra)
# ----------------------------------------------------------------------------
def h_natural_vertical(T_s, T_amb, L, h_alt):
    """
    h [W/m^2K] for a vertical surface of height L [m].
    T_s, T_amb in K; h_alt = altitude [m].
    """
    dT = abs(T_s - T_amb)
    dT = max(dT, 1e-3)
    p = air_props(0.5 * (T_s + T_amb), h_alt)
    Ra = 9.80665 * p["beta"] * dT * L ** 3 / (p["nu"] * p["alpha"])
    Pr = p["Pr"]
    Nu = (0.825 + 0.387 * Ra ** (1 / 6) / (1 + (0.492 / Pr) ** (9 / 16)) ** (8 / 27)) ** 2
    return Nu * p["k"] / L, Ra


# ----------------------------------------------------------------------------
# 4a. Forced convection - wind over a flat plate (enclosure wall in wind)
# ----------------------------------------------------------------------------
def h_wind_flat_plate(T_s, T_amb, L, U, h_alt):
    """Average h [W/m^2K] over a flat plate of length L [m] in wind speed U [m/s]."""
    p = air_props(0.5 * (T_s + T_amb), h_alt)
    Re = U * L / p["nu"]
    Pr = p["Pr"]
    if Re < 5e5:
        Nu = 0.664 * Re ** 0.5 * Pr ** (1 / 3)
    else:  # mixed laminar-turbulent
        Nu = (0.037 * Re ** 0.8 - 871.0) * Pr ** (1 / 3)
    return Nu * p["k"] / L, Re


# ----------------------------------------------------------------------------
# 4b. Forced convection - fan-driven flow through a duct / between fins
# ----------------------------------------------------------------------------
def h_duct(T_wall, T_in, D_h, Vdot, A_flow, h_alt):
    """
    h [W/m^2K] inside a duct of hydraulic diameter D_h [m], cross-section A_flow [m^2],
    carrying a fan volumetric flow Vdot [m^3/s] (a fan delivers ~constant VOLUME flow,
    so mass flow drops with altitude).
    Returns h, Re, mdot.
    """
    p = air_props(0.5 * (T_wall + T_in), h_alt)
    V = Vdot / A_flow
    Re = V * D_h / p["nu"]
    if Re < 2300:
        Nu = 3.66                                   # fully developed laminar, const T_w
    elif Re < 10000:                                # simple linear blend through transition
        Nu_t = 0.023 * 1e4 ** 0.8 * p["Pr"] ** 0.4
        Nu = 3.66 + (Re - 2300) / (1e4 - 2300) * (Nu_t - 3.66)
    else:
        Nu = 0.023 * Re ** 0.8 * p["Pr"] ** 0.4     # Dittus-Boelter (heating)
    mdot = p["rho"] * Vdot
    return Nu * p["k"] / D_h, Re, mdot


def air_temperature_rise(Q, Vdot, T_in, h_alt):
    """Bulk air temperature rise [K] when Q [W] is picked up by a fan flow Vdot [m^3/s]."""
    rho = air_density(h_alt, T_in)
    return Q / (rho * Vdot * CP_AIR)


# ----------------------------------------------------------------------------
# 5. Surface temperature of a heat-dissipating enclosure
# ----------------------------------------------------------------------------
def surface_temperature(Q, A, L, T_amb, h_alt, mode="natural", U_wind=0.0,
                        emissivity=0.85, T_surr=None, tol=1e-4, itmax=200):
    """
    Steady surface temperature [K] for dissipation Q [W] over area A [m^2], vertical wall
    height L [m], ambient T_amb [K], altitude h_alt [m].
    Heat leaves by convection (natural, or wind-forced) + linearised radiation.
    (Solar gain can be added later as an extra Q.)
    mode : 'natural' or 'wind'
    Returns T_s, h_conv, h_rad.
    """
    T_surr = T_amb if T_surr is None else T_surr
    T_s = T_amb + 10.0
    for _ in range(itmax):
        if mode == "natural":
            h_c, _ = h_natural_vertical(T_s, T_amb, L, h_alt)
        else:
            h_c, _ = h_wind_flat_plate(T_s, T_amb, L, max(U_wind, 0.1), h_alt)
        h_r = emissivity * SIGMA * (T_s + T_surr) * (T_s ** 2 + T_surr ** 2)
        T_new = T_amb + Q / (A * (h_c + h_r))
        if abs(T_new - T_s) < tol:
            T_s = T_new
            break
        T_s = 0.5 * (T_s + T_new)   # under-relaxation
    h_r = emissivity * SIGMA * (T_s + T_surr) * (T_s ** 2 + T_surr ** 2)
    return T_s, h_c, h_r


# ----------------------------------------------------------------------------
# 6. Demo: sea level vs. altitude
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # ---- Example inputs (CHANGE THESE to match your assumed system) --------
    Q_enclosure = 150.0     # W   dissipated by electronics in the enclosure
    A_enclosure = 0.60      # m^2 outer surface area
    L_wall = 0.40           # m   wall height
    U_wind = 5.0            # m/s wind speed
    T_ref_sea = 25.0 + 273.15      # K  sea-level reference ambient (comparison case)
    T_cold_alt = -20.0 + 273.15    # K  ambient held fixed during the altitude sweep (change as needed)
    Vdot_fan = 0.01         # m^3/s (~21 CFM) fan flow
    D_h, A_flow = 0.01, 0.004      # duct hydraulic diameter, flow area [m, m^2]
    # ------------------------------------------------------------------------

    altitudes = np.linspace(0, 6000, 25)

    rho_arr, hn_arr, hw_arr, hd_arr, Ts_n_arr, Ts_w_arr, dTair_arr = ([] for _ in range(7))
    for h in altitudes:
        T_amb = T_cold_alt   # same ambient at every altitude -> isolates the pressure/density effect
        rho_arr.append(air_density(h, T_amb))
        Ts_n, hc_n, _ = surface_temperature(Q_enclosure, A_enclosure, L_wall, T_amb, h, "natural")
        Ts_w, hc_w, _ = surface_temperature(Q_enclosure, A_enclosure, L_wall, T_amb, h, "wind", U_wind)
        hd, Re, mdot = h_duct(T_amb + 30, T_amb, D_h, Vdot_fan, A_flow, h)
        hn_arr.append(hc_n); hw_arr.append(hc_w); hd_arr.append(hd)
        Ts_n_arr.append(Ts_n - T_amb); Ts_w_arr.append(Ts_w - T_amb)
        dTair_arr.append(air_temperature_rise(Q_enclosure, Vdot_fan, T_amb, h))

    # ---- Sea-level reference (25 C) for comparison ------------------------------
    ref_Tn, ref_hn, _ = surface_temperature(Q_enclosure, A_enclosure, L_wall, T_ref_sea, 0, "natural")
    ref_Tw, ref_hw, _ = surface_temperature(Q_enclosure, A_enclosure, L_wall, T_ref_sea, 0, "wind", U_wind)
    ref_hd, _, _ = h_duct(T_ref_sea + 30, T_ref_sea, D_h, Vdot_fan, A_flow, 0)
    print(f"Sea-level reference at {T_ref_sea-273.15:.0f} C: h_nat = {ref_hn:.2f}, h_wind = {ref_hw:.2f}, "
          f"h_duct = {ref_hd:.1f} W/m2K; wall dT = {ref_Tn-T_ref_sea:.1f} K (natural), "
          f"{ref_Tw-T_ref_sea:.1f} K (wind)\n")
    print(f"Altitude sweep at fixed ambient {T_cold_alt-273.15:.0f} C:")

    # ---- Table ---------------------------------------------------------------
    print(f"{'alt[m]':>7} {'p[kPa]':>7} {'rho':>6} {'h_nat':>6} {'h_wind':>7} {'h_duct':>7} "
          f"{'dT_nat[K]':>10} {'dT_wind[K]':>11} {'dT_air_fan[K]':>14}")
    for i, h in enumerate(altitudes):
        if i % 4 == 0:
            print(f"{h:7.0f} {isa_pressure(h)/1e3:7.1f} {rho_arr[i]:6.3f} {hn_arr[i]:6.2f} "
                  f"{hw_arr[i]:7.2f} {hd_arr[i]:7.1f} {Ts_n_arr[i]:10.1f} {Ts_w_arr[i]:11.1f} "
                  f"{dTair_arr[i]:14.1f}")

    # ---- Plots ---------------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    ax[0, 0].plot(altitudes, rho_arr, "o-"); ax[0, 0].set_ylabel("Air density [kg/m$^3$]")
    ax[0, 1].plot(altitudes, hn_arr, label="natural conv."); ax[0, 1].plot(altitudes, hw_arr, label=f"wind {U_wind} m/s")
    ax[0, 1].plot(altitudes, hd_arr, label="fan duct flow"); ax[0, 1].set_ylabel("h [W/m$^2$K]")
    ax[0, 1].axhline(ref_hn, ls="--", c="C0", lw=0.8); ax[0, 1].axhline(ref_hw, ls="--", c="C1", lw=0.8)
    ax[0, 1].axhline(ref_hd, ls="--", c="C2", lw=0.8); ax[0, 1].legend(title="dashed = sea level, 25 C")
    ax[1, 0].plot(altitudes, Ts_n_arr, label="natural conv."); ax[1, 0].plot(altitudes, Ts_w_arr, label=f"wind {U_wind} m/s")
    ax[1, 0].axhline(ref_Tn - T_ref_sea, ls="--", c="C0", lw=0.8); ax[1, 0].axhline(ref_Tw - T_ref_sea, ls="--", c="C1", lw=0.8)
    ax[1, 0].set_ylabel("Wall temp. rise above ambient [K]"); ax[1, 0].legend(title="dashed = sea level, 25 C")
    ax[1, 1].plot(altitudes, dTair_arr, color="C3"); ax[1, 1].set_ylabel("Fan-air temp. rise [K]")
    for a in ax.flat:
        a.set_xlabel("Altitude [m]"); a.grid(alpha=0.3)
    fig.suptitle(f"Effect of altitude on convection (Q = {Q_enclosure:.0f} W, ambient fixed at {T_cold_alt-273.15:.0f} C)")
    fig.tight_layout()
    fig.savefig("altitude_convection_results.png", dpi=150)
    print("\nSaved plot: altitude_convection_results.png")