"""
ENVIRONMENT MODEL (Item 1) - high-altitude anti-drone thermal study (SIH 4)

Provides everything the later models (heat sink, day-night transient, battery/heater,
cable/gimbal) need to know about the surroundings, as a function of altitude and time:

    1. Atmosphere vs altitude    : pressure, density, air properties (table + plots)
    2. Ambient temperature cycle : day-night variation between a chosen Tmin and Tmax
    3. Solar radiation           : Hottel clear-sky model with altitude correction
    4. Sky temperature           : for radiation to the sky (Swinbank clear-sky correlation)
    5. Wind                      : mean + diurnal swing, and dynamic pressure (wind load)

Main interface for the other items:

    env = environment_at(t_solar_hours, scenario)   -> dict with T_amb, T_sky, G_h, rho, p, U, ...

Needs altitude_convection.py (from the previous step) in the same folder.

IMPORTANT: the scenario numbers below (Tmin/Tmax, wind, altitude, latitude) are ILLUSTRATIVE
assumptions. Replace them with IMD / literature data for your chosen site and cite the source
in the report.

Run:  python environment.py
"""

from dataclasses import dataclass
import csv
import numpy as np
import matplotlib.pyplot as plt

from altitude_convection import (isa_pressure, isa_temperature, air_props,
                                 G, R_AIR, SIGMA, CP_AIR)

G_SC = 1367.0   # W/m^2 solar constant


# =============================================================================
# 1. Scenario definition
# =============================================================================
@dataclass
class Scenario:
    name: str
    latitude: float      # deg N
    altitude: float      # m above sea level
    day_of_year: int     # 1..365
    T_min: float         # deg C  daily minimum ambient (near sunrise)
    T_max: float         # deg C  daily maximum ambient (mid-afternoon)
    U_mean: float        # m/s    mean wind speed
    U_swing: float       # m/s    afternoon-peak amplitude of wind (0 = steady wind)
    season: str          # 'winter' or 'summer' (Hottel climate correction factors)
    t_Tmax: float = 15.0 # solar hour of daily maximum temperature


# ILLUSTRATIVE scenarios - replace with real site data (IMD, literature, SIH brief)
SCENARIOS = {
    "Leh winter (typical)":  Scenario("Leh winter (typical)", 34.15, 3500, 15,  -15.0,  0.0, 4.0, 2.0, "winter"),
    "HAA winter (severe)":   Scenario("HAA winter (severe)",  34.15, 4500, 15,  -35.0, -10.0, 8.0, 4.0, "winter"),
    "HAA summer":            Scenario("HAA summer",           34.15, 4500, 180,   0.0,  20.0, 6.0, 3.0, "summer"),
}


# =============================================================================
# 2. Atmosphere vs altitude
# =============================================================================
def atmosphere_table(altitudes, T_amb_C=None, dT_film=0.0):
    """
    Air state vs altitude. If T_amb_C is None the ISA temperature is used at each altitude,
    otherwise the same ambient temperature (deg C) at every altitude (isolates pressure effect).
    dT_film : add this to T to get the film temperature for the property evaluation [K].
    Returns a list of dicts.
    """
    rows = []
    for h in altitudes:
        T = isa_temperature(h) if T_amb_C is None else T_amb_C + 273.15
        pr = air_props(T + dT_film, h)
        rows.append(dict(h=h, T=T, p=isa_pressure(h), **pr))
    return rows


def rayleigh_scale(h, T_amb, dT=30.0, L=0.4):
    """Rayleigh number Ra = g*beta*dT*L^3/(nu*alpha) at altitude h (natural convection driver)."""
    p = air_props(T_amb + dT / 2, h)
    return G * p["beta"] * dT * L ** 3 / (p["nu"] * p["alpha"])


# =============================================================================
# 3. Ambient temperature (day-night)
# =============================================================================
def solar_declination(n):
    """Declination [deg], Cooper's equation, n = day of year."""
    return 23.45 * np.sin(np.radians(360.0 * (284 + n) / 365.0))


def sunrise_sunset(lat, n):
    """Solar-time sunrise and sunset [h]."""
    dec = np.radians(solar_declination(n))
    x = -np.tan(np.radians(lat)) * np.tan(dec)
    x = np.clip(x, -1.0, 1.0)
    ws = np.degrees(np.arccos(x))      # sunset hour angle
    return 12.0 - ws / 15.0, 12.0 + ws / 15.0


def ambient_temperature(t, sc):
    """
    Ambient temperature [deg C] at solar time t [h].
    Minimum at sunrise, maximum at sc.t_Tmax; half-cosine rise and fall (continuous, 24 h periodic).
    """
    t_sr, _ = sunrise_sunset(sc.latitude, sc.day_of_year)
    t_min, t_max = t_sr, sc.t_Tmax
    t = t % 24.0
    if t < t_min:
        t += 24.0
    rise = t_max - t_min
    fall = 24.0 - t_max + t_min
    if t <= t_max:
        f = 0.5 * (1.0 - np.cos(np.pi * (t - t_min) / rise))
    else:
        f = 0.5 * (1.0 + np.cos(np.pi * (t - t_max) / fall))
    return sc.T_min + (sc.T_max - sc.T_min) * f


# =============================================================================
# 4. Solar radiation: Hottel clear-sky model (Duffie & Beckman) with altitude
# =============================================================================
def extraterrestrial_normal(n):
    return G_SC * (1.0 + 0.033 * np.cos(np.radians(360.0 * n / 365.0)))


def cos_zenith(lat, n, t):
    dec = np.radians(solar_declination(n))
    phi = np.radians(lat)
    omega = np.radians(15.0 * (t - 12.0))
    return np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.cos(omega)


def hottel_clear_sky(cosz, altitude_m, n, season):
    """
    Clear-sky beam (normal), beam on horizontal, diffuse (horizontal), global horizontal [W/m^2].
    Hottel's coefficients use altitude A in km. The correlation is formally fitted for A < 2.5 km;
    it is extrapolated here to 4-5 km. This is a stated assumption: say so in the report.
    """
    if cosz <= 0.0:
        return dict(Gbn=0.0, Gbh=0.0, Gdh=0.0, Gh=0.0)
    A = altitude_m / 1000.0
    a0 = 0.4237 - 0.00821 * (6.0 - A) ** 2
    a1 = 0.5055 + 0.00595 * (6.5 - A) ** 2
    k = 0.2711 + 0.01858 * (2.5 - A) ** 2
    r0, r1, rk = (1.03, 1.01, 1.00) if season == "winter" else (0.97, 0.99, 1.02)   # mid-latitude
    tau_b = r0 * a0 + r1 * a1 * np.exp(-rk * k / cosz)
    tau_d = 0.271 - 0.294 * tau_b
    Gon = extraterrestrial_normal(n)
    Gbn = Gon * tau_b
    Gbh = Gbn * cosz
    Gdh = Gon * cosz * tau_d
    return dict(Gbn=Gbn, Gbh=Gbh, Gdh=Gdh, Gh=Gbh + Gdh)


# =============================================================================
# 5. Sky temperature and wind
# =============================================================================
def sky_temperature(T_amb_C):
    """
    Effective clear-sky temperature [deg C], Swinbank: T_sky = 0.0552*T_amb^1.5 (K).
    Dry, thin high-altitude air is usually colder than this, so it is a conservative choice
    for heat-loss estimates (state as an assumption).
    """
    T_K = T_amb_C + 273.15
    return 0.0552 * T_K ** 1.5 - 273.15


def wind_speed(t, sc):
    """Wind speed [m/s]: mean plus afternoon peak (illustrative valley-wind pattern)."""
    return max(sc.U_mean + sc.U_swing * np.cos(2 * np.pi * (t - 15.0) / 24.0), 0.0)


# =============================================================================
# 6. Single interface used by the other items
# =============================================================================
def environment_at(t, sc):
    """All environmental inputs at solar time t [h] for scenario sc."""
    T_amb_C = ambient_temperature(t, sc)
    T_amb = T_amb_C + 273.15
    p = isa_pressure(sc.altitude)
    rho = p / (R_AIR * T_amb)
    cz = cos_zenith(sc.latitude, sc.day_of_year, t)
    sol = hottel_clear_sky(cz, sc.altitude, sc.day_of_year, sc.season)
    U = wind_speed(t, sc)
    props = air_props(T_amb, sc.altitude)
    return dict(t=t, T_amb_C=T_amb_C, T_amb=T_amb, T_sky_C=sky_temperature(T_amb_C),
                p=p, rho=rho, mu=props["mu"], k=props["k"], nu=props["nu"], Pr=props["Pr"],
                cos_zenith=cz, sun_elev_deg=np.degrees(np.arcsin(max(cz, 0.0))),
                U=U, q_wind=0.5 * rho * U ** 2, **sol)


def daily_profile(sc, dt=0.25):
    ts = np.arange(0.0, 24.0 + 1e-9, dt)
    return [environment_at(t, sc) for t in ts]


def export_csv(sc, fname, dt=0.25):
    prof = daily_profile(sc, dt)
    cols = ["t", "T_amb_C", "T_sky_C", "Gbn", "Gbh", "Gdh", "Gh", "sun_elev_deg", "p", "rho", "U", "q_wind"]
    with open(fname, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in prof:
            w.writerow([f"{r[c]:.4f}" for c in cols])


# =============================================================================
# 7. Demo: tables and figures
# =============================================================================
if __name__ == "__main__":
    # ---------- Table 1: atmosphere vs altitude (ISA temperature) ----------------
    alts = np.arange(0, 6001, 1000)
    rows = atmosphere_table(alts)
    r0 = rows[0]
    print("TABLE 1  Standard atmosphere and air properties (ISA temperature at each altitude)")
    print(f"{'h[m]':>6} {'T[C]':>6} {'p[kPa]':>7} {'rho':>6} {'rho/rho0':>8} {'mu[1e-5]':>9} "
          f"{'k[mW/mK]':>9} {'nu[1e-5]':>9} {'Pr':>6} {'Ra/Ra0':>7}")
    for r in rows:
        Ra_ratio = rayleigh_scale(r["h"], r["T"]) / rayleigh_scale(0, r0["T"])
        print(f"{r['h']:6.0f} {r['T']-273.15:6.1f} {r['p']/1e3:7.1f} {r['rho']:6.3f} {r['rho']/r0['rho']:8.3f} "
              f"{r['mu']*1e5:9.3f} {r['k']*1e3:9.2f} {r['nu']*1e5:9.2f} {r['Pr']:6.3f} {Ra_ratio:7.3f}")

    # ---------- Table 2: same ambient temperature, altitude varies ---------------
    print("\nTABLE 2  Pressure effect alone: ambient fixed at -20 C")
    rows2 = atmosphere_table(alts, T_amb_C=-20.0)
    for r in rows2:
        print(f"{r['h']:6.0f} m   p = {r['p']/1e3:6.1f} kPa   rho = {r['rho']:.3f} kg/m3   "
              f"nu = {r['nu']*1e5:.2f}e-5 m2/s   (k = {r['k']*1e3:.2f} mW/mK, unchanged)")

    # ---------- Table 3: scenario summaries --------------------------------------
    print("\nTABLE 3  Scenario summary (solar time)")
    for sc in SCENARIOS.values():
        prof = daily_profile(sc, 0.1)
        sr, ss = sunrise_sunset(sc.latitude, sc.day_of_year)
        Gh = np.array([r["Gh"] for r in prof]); T = np.array([r["T_amb_C"] for r in prof])
        rho = np.array([r["rho"] for r in prof]); q = np.array([r["q_wind"] for r in prof])
        Tsky = np.array([r["T_sky_C"] for r in prof])
        daily_kWh = np.trapezoid(Gh, dx=0.1) / 1000.0
        print(f"\n{sc.name}: altitude {sc.altitude:.0f} m, p = {isa_pressure(sc.altitude)/1e3:.1f} kPa, "
              f"day {sc.day_of_year}")
        print(f"   sunrise {sr:4.1f} h, sunset {ss:4.1f} h; T_amb {T.min():.1f} to {T.max():.1f} C; "
              f"T_sky {Tsky.min():.1f} to {Tsky.max():.1f} C")
        print(f"   peak global horizontal = {Gh.max():.0f} W/m2, clear-sky daily total = {daily_kWh:.2f} kWh/m2/day")
        print(f"   air density {rho.min():.3f} to {rho.max():.3f} kg/m3; peak wind dynamic pressure = {q.max():.0f} Pa")
        fn = "environment_profile_" + sc.name.split(" (")[0].replace(" ", "_") + ".csv"
        export_csv(sc, fn)
        print(f"   saved {fn}")

    # ---------- Figure 1: atmosphere vs altitude ---------------------------------
    alt_f = np.linspace(0, 6000, 61)
    isa_rows = atmosphere_table(alt_f)
    fix_rows = atmosphere_table(alt_f, T_amb_C=-20.0)
    fig, ax = plt.subplots(2, 3, figsize=(13, 7.5))
    ax[0, 0].plot(alt_f, [r["p"] / 1e3 for r in isa_rows]); ax[0, 0].set_ylabel("Pressure [kPa]")
    ax[0, 1].plot(alt_f, [r["rho"] for r in isa_rows], label="ISA temperature")
    ax[0, 1].plot(alt_f, [r["rho"] for r in fix_rows], "--", label="fixed -20 C")
    ax[0, 1].set_ylabel("Density [kg/m$^3$]"); ax[0, 1].legend()
    ax[0, 2].plot(alt_f, [r["nu"] * 1e5 for r in isa_rows], label="ISA temperature")
    ax[0, 2].plot(alt_f, [r["nu"] * 1e5 for r in fix_rows], "--", label="fixed -20 C")
    ax[0, 2].set_ylabel(r"Kinematic viscosity $\nu$ [$10^{-5}$ m$^2$/s]"); ax[0, 2].legend()
    ax[1, 0].plot(alt_f, [r["k"] * 1e3 for r in isa_rows], label="ISA temperature")
    ax[1, 0].plot(alt_f, [r["k"] * 1e3 for r in fix_rows], "--", label="fixed -20 C")
    ax[1, 0].set_ylabel("Conductivity k [mW/m K]"); ax[1, 0].legend()
    ax[1, 1].plot(alt_f, [r["Pr"] for r in isa_rows]); ax[1, 1].set_ylabel("Prandtl number")
    ax[1, 1].set_ylim(0.6, 0.8)
    Ra0 = rayleigh_scale(0, isa_rows[0]["T"])
    ax[1, 2].plot(alt_f, [rayleigh_scale(r["h"], r["T"]) / Ra0 for r in isa_rows], label="ISA temperature")
    ax[1, 2].plot(alt_f, [rayleigh_scale(r["h"], -20 + 273.15) / rayleigh_scale(0, -20 + 273.15) for r in isa_rows],
                  "--", label="fixed -20 C")
    ax[1, 2].set_ylabel(r"Ra / Ra$_{sea\ level}$ ($\Delta T$=30 K, L=0.4 m)"); ax[1, 2].legend()
    for a in ax.flat:
        a.set_xlabel("Altitude [m]"); a.grid(alpha=0.3)
    fig.suptitle("Atmosphere and air properties vs altitude")
    fig.tight_layout(); fig.savefig("environment_atmosphere.png", dpi=150)

    # ---------- Figure 2: day-night cycle for the scenarios ----------------------
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    for i, sc in enumerate(SCENARIOS.values()):
        prof = daily_profile(sc, 0.1)
        t = [r["t"] for r in prof]
        c = f"C{i}"
        ax[0, 0].plot(t, [r["T_amb_C"] for r in prof], c, label=sc.name)
        ax[0, 0].plot(t, [r["T_sky_C"] for r in prof], c + "--", lw=0.9)
        ax[0, 1].plot(t, [r["Gh"] for r in prof], c, label=sc.name)
        ax[1, 0].plot(t, [r["rho"] for r in prof], c, label=sc.name)
        ax[1, 1].plot(t, [SIGMA * ((r["T_amb"]) ** 4 - (r["T_sky_C"] + 273.15) ** 4) for r in prof], c, label=sc.name)
    ax[0, 0].set_ylabel("Temperature [C]\n(solid = ambient, dashed = clear-sky)")
    ax[0, 1].set_ylabel("Clear-sky global horizontal irradiance [W/m$^2$]")
    ax[1, 0].set_ylabel("Air density at site [kg/m$^3$]")
    ax[1, 1].set_ylabel(r"$\sigma(T_{amb}^4 - T_{sky}^4)$ [W/m$^2$]" + "\n(black-surface radiative loss to sky)")
    for a in ax.flat:
        a.set_xlabel("Solar time [h]"); a.set_xlim(0, 24); a.set_xticks(range(0, 25, 4)); a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.suptitle("Day-night environment (illustrative scenarios)")
    fig.tight_layout(); fig.savefig("environment_daily_cycle.png", dpi=150)
    print("\nSaved: environment_atmosphere.png, environment_daily_cycle.png")