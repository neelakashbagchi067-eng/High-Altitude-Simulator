"""
ITEM 4 - Transient (lumped-capacitance) day-night model of the anti-drone electronics enclosure

Same thermal network as item 3, now with thermal mass, a 24 h environment, and time-varying power.

States (6):  T_c,i  (4 components, each a lumped mass)   -- R_jc + R_tim --> base plate
             T_b    (base plate + heat sink + internal structure)
             T_w    (enclosure skin)

    C_i  dT_c,i/dt = Q_i(t) - (T_c,i - T_b)/R_i
    C_b  dT_b/dt   = sum_i (T_c,i - T_b)/R_i - Q_fin(T_b) - Q_int(T_b, T_w)
    C_w  dT_w/dt   = Q_int + Q_sun(t) - Q_ext(T_w)

Environment (from environment.py, clear-sky):  T_amb(t), T_sky(t), wind U(t), solar on each face
of the box (beam + diffuse + ground-reflected, with snow albedo option), solar absorptivity alpha.
Wall convection = natural/wind blended (n=3 mixed-convection rule, continuous), plus radiation to
a sky/ground mixture.  Heat sink: natural convection, fixed-speed fan, or proportional fan.

Checks: with constant boundary conditions the transient solution must settle at the item 3
steady-state result (printed at the start).

Needs altitude_convection.py, environment.py, thermal_network.py in the same folder.
ALL masses, powers, limits and areas are ILLUSTRATIVE ASSUMPTIONS ("CHANGE THESE" block).

Run:  python transient_daynight.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from altitude_convection import (h_natural_vertical, h_wind_flat_plate, R_AIR, SIGMA, isa_pressure)
from environment import (SCENARIOS, environment_at, solar_declination, sky_temperature)
from thermal_network import (COMPONENTS, HS, ENC, heat_sink_heat, solve_system)

# =============================================================================
# CHANGE THESE: thermal masses, limits, geometry
# =============================================================================
RHO_AL, CP_AL = 2700.0, 900.0
CAP_COMP = {"Processor / SoC": 90.0, "RF power amplifier": 120.0,           # J/K per component
            "Power electronics": 150.0, "Gimbal drive / motor": 400.0}
T_MIN_OP = {"Processor / SoC": -40.0, "RF power amplifier": -40.0,          # deg C, lowest operating T
            "Power electronics": -40.0, "Gimbal drive / motor": -20.0}      # (grease / bearings)
C_INTERNAL = 1500.0       # J/K  PCBs, brackets, chassis tied to the base plate
WALL_THK = 0.002          # m    aluminium skin
ALBEDO_DEFAULT = 0.2      # ground reflectance (snow: 0.6 - 0.8)
FACES = {                 # name: (area m2, tilt beta deg, surface azimuth gamma deg; 0 = south, -90 = east)
    "top":   (0.12, 0.0, 0.0),
    "south": (0.12, 90.0, 0.0),
    "east":  (0.12, 90.0, -90.0),
    "west":  (0.12, 90.0, 90.0),
    "north": (0.12, 90.0, 180.0),
}                         # the heat-sink face is the 6th face, assumed shaded
FAN_RATED_POWER = 4.0     # W at sea-level density and Vmax
FAN_VMAX = 0.012          # m^3/s
FAN_T_ON, FAN_T_FULL = 45.0, 65.0   # deg C base-plate temperatures for proportional fan control

C_BASE = (RHO_AL * (HS.W * HS.L * HS.t_base + HS.N * HS.t * HS.H * HS.L) * CP_AL) + C_INTERNAL
C_WALL = RHO_AL * WALL_THK * ENC.A_wall * CP_AL

DUTY = {  # (hour breakpoints, power scale), periodic over 24 h
    "continuous": ([0, 12], [1.0, 1.0]),
    "standby_night": ([0, 6.8333, 7.0, 19.0, 19.1667], [0.3, 0.3, 1.0, 1.0, 0.3]),
    "off_night": ([0, 6.8333, 7.0, 19.0, 19.1667], [0.0, 0.0, 1.0, 1.0, 0.0]),
}


# =============================================================================
# Solar on an oriented surface
# =============================================================================
def cos_incidence(lat, n, t, beta, gamma):
    phi, dec = np.radians(lat), np.radians(solar_declination(n))
    om, b, g = np.radians(15.0 * (t - 12.0)), np.radians(beta), np.radians(gamma)
    return (np.sin(dec) * np.sin(phi) * np.cos(b) - np.sin(dec) * np.cos(phi) * np.sin(b) * np.cos(g)
            + np.cos(dec) * np.cos(phi) * np.cos(b) * np.cos(om)
            + np.cos(dec) * np.sin(phi) * np.sin(b) * np.cos(g) * np.cos(om)
            + np.cos(dec) * np.sin(b) * np.sin(g) * np.sin(om))


def absorbed_solar(env, sc, alpha, albedo, t):
    """Solar power [W] absorbed by the box skin (beam + isotropic diffuse + ground reflected)."""
    if env["cos_zenith"] <= 0:
        return 0.0
    Q = 0.0
    for A, beta, gamma in FACES.values():
        cb = np.cos(np.radians(beta))
        cth = max(cos_incidence(sc.latitude, sc.day_of_year, t, beta, gamma), 0.0)
        Q += alpha * A * (env["Gbn"] * cth + env["Gdh"] * (1 + cb) / 2 + albedo * env["Gh"] * (1 - cb) / 2)
    return Q


# =============================================================================
# Simulation
# =============================================================================
def simulate(sc, mode="natural", duty="continuous", alpha=None, albedo=ALBEDO_DEFAULT, wind_scale=1.0,
             F_sky=0.5, days=3, const_env=None, comps=COMPONENTS, hs=HS, enc=ENC):
    """
    mode : 'natural' | 'fan' (fixed speed) | 'fan_prop' (speed vs base-plate temperature)
    duty : key of DUTY (power profile of the electronics)
    const_env : dict(T_amb_C, G_sun, U_wind) -> constant boundary conditions (verification run)
    F_sky : fraction of the wall's radiative exchange that is with the sky (rest: air-temperature surroundings)
    """
    alpha = enc.alpha_sun if alpha is None else alpha
    alt = sc.altitude
    n = len(comps)
    dt_env = 300.0
    ts = np.arange(0.0, days * 86400.0 + dt_env, dt_env)

    A_amb, A_sky, A_sun, A_U = (np.zeros_like(ts) for _ in range(4))
    for i, t in enumerate(ts):
        if const_env is None:
            e = environment_at(t / 3600.0, sc)
            A_amb[i], A_sky[i] = e["T_amb"], e["T_sky_C"] + 273.15
            A_sun[i] = absorbed_solar(e, sc, alpha, albedo, t / 3600.0)
            A_U[i] = e["U"] * wind_scale
        else:
            A_amb[i] = const_env["T_amb_C"] + 273.15
            A_sky[i] = sky_temperature(const_env["T_amb_C"]) + 273.15
            A_sun[i] = alpha * const_env["G_sun"] * enc.A_sun
            A_U[i] = const_env["U_wind"]

    xp, fp = DUTY[duty]
    Rc = np.array([c.R_to_base for c in comps])
    Cc = np.array([CAP_COMP[c.name] for c in comps])
    Qc0 = np.array([c.Q for c in comps])
    alt_int = alt if enc.alt_internal is None else enc.alt_internal

    def terms(t, y):
        Tc, Tb, Tw = y[:n], y[n], y[n + 1]
        Tamb, Tsky = np.interp(t, ts, A_amb), np.interp(t, ts, A_sky)
        Qsun, U = np.interp(t, ts, A_sun), np.interp(t, ts, A_U)
        scale = np.interp((t / 3600.0) % 24.0, xp, fp, period=24.0)
        Qc = Qc0 * scale
        # heat sink
        Vd = 0.0
        if mode == "fan":
            Vd = FAN_VMAX
        elif mode == "fan_prop":
            Vd = FAN_VMAX * np.clip(((Tb - 273.15) - FAN_T_ON) / (FAN_T_FULL - FAN_T_ON), 0.0, 1.0)
        Q_fin = heat_sink_heat(Tb, Tamb, hs, alt, "natural")["Q"]
        if Vd > 1e-5:
            Q_fin = max(Q_fin, heat_sink_heat(Tb, Tamb, hs, alt, "fan", Vd)["Q"])
        rho = isa_pressure(alt) / (R_AIR * Tamb)
        P_fan = FAN_RATED_POWER * (Vd / FAN_VMAX) ** 3 * rho / 1.225 if Vd > 1e-5 else 0.0
        # internal exchange base plate <-> wall
        hci, _ = h_natural_vertical(Tb, Tw, enc.L_int, alt_int)
        hri = enc.eps_int * SIGMA * (Tb + Tw) * (Tb ** 2 + Tw ** 2)
        Q_int = (Tb - Tw) * enc.A_int * (hci + hri)
        # external exchange wall <-> ambient / sky (mixed convection + radiation)
        hn, _ = h_natural_vertical(Tw, Tamb, enc.L_wall, alt)
        hf = h_wind_flat_plate(Tw, Tamb, enc.L_wall, U, alt)[0] if U > 0.1 else 0.0
        hc = (hn ** 3 + hf ** 3) ** (1.0 / 3.0)
        Trs = F_sky * Tsky + (1.0 - F_sky) * Tamb
        hr = enc.eps_ext * SIGMA * (Tw + Trs) * (Tw ** 2 + Trs ** 2)
        Q_ext = enc.A_wall * (hc * (Tw - Tamb) + hr * (Tw - Trs))   # radiation driven by T_w - T_rs
        Q_to_base = (Tc - Tb) / Rc
        dy = np.empty(n + 2)
        dy[:n] = (Qc - Q_to_base) / Cc
        dy[n] = (Q_to_base.sum() - Q_fin - Q_int) / C_BASE
        dy[n + 1] = (Q_int + Qsun - Q_ext) / C_WALL
        diag = dict(Q_fin=Q_fin, Q_int=Q_int, Q_ext=Q_ext, Q_sun=Qsun, Q_el=Qc.sum(), V_fan=Vd, P_fan=P_fan,
                    T_amb=Tamb, T_sky=Tsky, U=U, scale=scale)
        return dy, diag

    y0 = np.full(n + 2, A_amb[0])
    t_eval = np.arange(0.0, days * 86400.0 + 1.0, 300.0)
    sol = solve_ivp(lambda t, y: terms(t, y)[0], (0.0, days * 86400.0), y0, method="LSODA",
                    t_eval=t_eval, rtol=1e-6, atol=1e-4, max_step=300.0)
    if not sol.success:
        raise RuntimeError(sol.message)
    diag = [terms(t, sol.y[:, k])[1] for k, t in enumerate(sol.t)]
    out = {key: np.array([d[key] for d in diag]) for key in diag[0]}
    out.update(t=sol.t / 3600.0, Tc={c.name: sol.y[i] for i, c in enumerate(comps)}, Tb=sol.y[n], Tw=sol.y[n + 1],
               comps=comps, sc=sc, mode=mode, duty=duty)
    return out


def last_day(r):
    """Boolean mask of the last 24 h and the time (h) within that day."""
    m = r["t"] >= r["t"][-1] - 24.0 - 1e-9
    return m, r["t"][m] - r["t"][m][0]


def summarize(r, label):
    m, th = last_day(r)
    print(f"\n{label}")
    min_margin = np.inf
    for c in r["comps"]:
        T = r["Tc"][c.name][m] - 273.15
        i = int(np.argmax(T))
        marg = c.T_lim_C - T.max()
        min_margin = min(min_margin, marg)
        print(f"   {c.name:<22s} max {T.max():6.1f} C at {th[i]:4.1f} h (limit {c.T_lim_C:.0f}, margin {marg:+5.1f} K), "
              f"min {T.min():6.1f} C (limit {T_MIN_OP[c.name]:.0f})")
    Tw = r["Tw"][m] - 273.15
    E_sun = np.trapezoid(r["Q_sun"][m], r["t"][m]) / 1000.0
    E_fan = np.trapezoid(r["P_fan"][m], r["t"][m])
    print(f"   skin {Tw.min():.1f} to {Tw.max():.1f} C (ambient {r['T_amb'][m].min()-273.15:.1f} to "
          f"{r['T_amb'][m].max()-273.15:.1f} C); absorbed solar peak {r['Q_sun'][m].max():.0f} W, "
          f"daily {E_sun:.2f} kWh; fan energy {E_fan:.1f} Wh/day")
    return min_margin


def metrics(r):
    m, _ = last_day(r)
    marg = min(c.T_lim_C - (r["Tc"][c.name][m].max() - 273.15) for c in r["comps"])
    Tmin = min(min(r["Tc"][c.name][m] - 273.15) for c in r["comps"])
    return dict(margin=marg, Tw_max=(r["Tw"][m].max() - 273.15), Tw_min=(r["Tw"][m].min() - 273.15), Tc_min=Tmin)


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    summer = SCENARIOS["HAA summer"]
    winter = SCENARIOS["HAA winter (severe)"]

    print(f"Thermal masses: components {sum(CAP_COMP.values()):.0f} J/K, base plate + heat sink + internals "
          f"{C_BASE:.0f} J/K, skin {C_WALL:.0f} J/K")

    # ---------- verification against the item-3 steady state ---------------------------
    cond = dict(T_amb_C=20.0, altitude=4500.0, G_sun=1003.0, U_wind=0.0)
    ss = solve_system(cond)
    r = simulate(summer, const_env=dict(T_amb_C=20.0, G_sun=1003.0, U_wind=0.0), days=1)
    print("\nVERIFICATION (constant 20 C, 1003 W/m2 on 0.3 m2, no wind, 4500 m): transient end state vs item-3 steady state")
    print(f"   base plate: {r['Tb'][-1]-273.15:6.2f} C vs {ss['Tb']-273.15:6.2f} C;  skin: {r['Tw'][-1]-273.15:6.2f} C vs "
          f"{ss['Tw']-273.15:6.2f} C")
    for c in COMPONENTS:
        print(f"   {c.name:<22s}: {r['Tc'][c.name][-1]-273.15:6.2f} C vs {ss['Tc'][c.name]-273.15:6.2f} C")
    G_tot = sum(c.Q for c in COMPONENTS) / (ss["Tb"] - ss["T_amb"])
    C_tot = sum(CAP_COMP.values()) + C_BASE + C_WALL
    print(f"   overall time constant approx. C_total/G = {C_tot:.0f}/{G_tot:.2f} = {C_tot/G_tot/60:.0f} min")

    # ---------- day-night runs ----------------------------------------------------------
    runs = {}
    runs["summer_nat"] = simulate(summer, "natural", "continuous")
    runs["summer_fan"] = simulate(summer, "fan_prop", "continuous")
    runs["winter_cont"] = simulate(winter, "natural", "continuous")
    runs["winter_off"] = simulate(winter, "natural", "off_night")
    runs["summer_standby"] = simulate(summer, "natural", "standby_night")
    summarize(runs["summer_nat"], "HAA summer (4500 m, 0 to 20 C), continuous power, natural convection")
    summarize(runs["summer_fan"], "HAA summer, continuous power, proportional fan")
    summarize(runs["summer_standby"], "HAA summer, natural convection, standby (30 %) at night")
    summarize(runs["winter_cont"], "HAA winter (-35 to -10 C), continuous power, natural convection")
    summarize(runs["winter_off"], "HAA winter, powered off at night (cold-soak start-up at 07:00)")

    # warm-up after cold soak
    r = runs["winter_off"]
    m, th = last_day(r)
    print("\nCold-soak start-up (winter, power switched on at 06:50, full power by 07:00):")
    for c in COMPONENTS:
        T = r["Tc"][c.name][m] - 273.15
        i_on = int(np.argmin(np.abs(th - 6.8333)))
        if T[i_on] >= T_MIN_OP[c.name]:
            print(f"   {c.name:<22s}: {T[i_on]:6.1f} C at power-on (above its {T_MIN_OP[c.name]:.0f} C minimum)")
        else:
            j = i_on + int(np.argmax(T[i_on:] >= T_MIN_OP[c.name]))
            print(f"   {c.name:<22s}: {T[i_on]:6.1f} C at power-on, reaches {T_MIN_OP[c.name]:.0f} C after "
                  f"{(th[j]-th[i_on])*60:.0f} min")

    # ---------- sensitivities -------------------------------------------------------------
    sens = [("baseline", {}), ("solar absorptivity 0.25 (white)", dict(alpha=0.25)),
            ("solar absorptivity 0.90", dict(alpha=0.90)), ("no wind", dict(wind_scale=0.0)),
            ("wind x2", dict(wind_scale=2.0)), ("no sky radiation (F_sky = 0)", dict(F_sky=0.0)),
            ("full sky view (F_sky = 1)", dict(F_sky=1.0)),
            ("calm night, no sky radiation", dict(wind_scale=0.0, F_sky=0.0)),
            ("calm night, full sky view", dict(wind_scale=0.0, F_sky=1.0))]
    print("\nSENSITIVITY  (natural convection; summer = continuous power, winter = powered off at night)")
    print(f"{'case':<34s} {'summer: min margin':>19s} {'skin max':>9s} | {'winter: skin min':>17s} {'comp min':>9s}")
    sens_res = []
    for lab, kw in sens:
        ms = metrics(simulate(summer, "natural", "continuous", **kw))
        mw = metrics(simulate(winter, "natural", "off_night", **kw))
        sens_res.append((lab, ms, mw))
        print(f"{lab:<34s} {ms['margin']:+16.1f} K {ms['Tw_max']:8.1f}C | {mw['Tw_min']:15.1f} C {mw['Tc_min']:8.1f}C")

    # ---------- figures -------------------------------------------------------------------
    lim_color = "0.3"
    # Figure 1: summer
    r1, r2 = runs["summer_nat"], runs["summer_fan"]
    m1, th = last_day(r1)
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))
    for c in COMPONENTS:
        ax[0, 0].plot(th, r1["Tc"][c.name][m1] - 273.15, label=c.name)
    ax[0, 0].plot(th, r1["Tb"][m1] - 273.15, "k", lw=2, label="base plate / heat sink")
    ax[0, 0].plot(th, r1["Tw"][m1] - 273.15, "brown", lw=2, label="enclosure skin")
    ax[0, 0].plot(th, r1["T_amb"][m1] - 273.15, "b--", label="ambient")
    ax[0, 0].plot(th, r1["T_sky"][m1] - 273.15, "c:", label="clear sky")
    ax[0, 0].set_ylabel("Temperature [C]"); ax[0, 0].legend(fontsize=7, ncol=2)
    ax[0, 0].set_title("HAA summer, natural convection, continuous power", fontsize=9)
    ax[0, 1].plot(th, r1["Q_sun"][m1], "orange", label="absorbed solar")
    ax[0, 1].plot(th, r1["Q_el"][m1], "r", label="electronics dissipation")
    ax[0, 1].plot(th, r1["Q_fin"][m1], "k", label="heat sink to air")
    ax[0, 1].plot(th, r1["Q_ext"][m1], "brown", label="skin to air + sky (net)")
    ax[0, 1].set_ylabel("Heat flow [W]"); ax[0, 1].legend(fontsize=8)
    rf = [c for c in COMPONENTS if c.name == "RF power amplifier"][0]
    ax[1, 0].plot(th, r1["Tc"][rf.name][m1] - 273.15, label="natural convection")
    ax[1, 0].plot(th, r2["Tc"][rf.name][m1] - 273.15, label="proportional fan")
    ax[1, 0].axhline(rf.T_lim_C, color=lim_color, ls="--", label="limit")
    ax[1, 0].set_ylabel("RF amplifier temperature [C]"); ax[1, 0].legend(fontsize=8)
    ax[1, 1].plot(th, r2["V_fan"][m1] / FAN_VMAX * 100, "C2"); ax[1, 1].set_ylabel("Fan speed [% of max]", color="C2")
    ax3 = ax[1, 1].twinx(); ax3.plot(th, r2["P_fan"][m1], "C3--"); ax3.set_ylabel("Fan power [W] (dashed)", color="C3")
    ax[1, 1].set_title(f"Fan energy {np.trapezoid(r2['P_fan'][m1], r2['t'][m1]):.0f} Wh/day", fontsize=9)
    for a in ax.flat:
        a.set_xlabel("Solar time [h]"); a.set_xlim(0, 24); a.set_xticks(range(0, 25, 4)); a.grid(alpha=0.3)
    fig.suptitle("Item 4: day-night transient, HAA summer (4500 m)")
    fig.tight_layout(); fig.savefig("item4_summer.png", dpi=150)

    # Figure 2: winter + sensitivity
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))
    rw, ro = runs["winter_cont"], runs["winter_off"]
    mw_, thw = last_day(rw)
    for c in COMPONENTS:
        ax[0, 0].plot(thw, rw["Tc"][c.name][mw_] - 273.15, label=c.name)
    ax[0, 0].plot(thw, rw["Tw"][mw_] - 273.15, "brown", lw=2, label="enclosure skin")
    ax[0, 0].plot(thw, rw["T_amb"][mw_] - 273.15, "b--", label="ambient")
    ax[0, 0].set_title("HAA winter, continuous power", fontsize=9); ax[0, 0].legend(fontsize=7)
    mo, tho = last_day(ro)
    for c in COMPONENTS:
        ax[0, 1].plot(tho, ro["Tc"][c.name][mo] - 273.15, label=c.name)
    ax[0, 1].plot(tho, ro["T_amb"][mo] - 273.15, "b--", label="ambient")
    ax[0, 1].axhline(T_MIN_OP["Gimbal drive / motor"], color=lim_color, ls="--", label="gimbal minimum operating T")
    ax[0, 1].set_title("HAA winter, powered off 19:10 - 06:50 (cold soak)", fontsize=9); ax[0, 1].legend(fontsize=7)
    z = (tho >= 6.0) & (tho <= 10.0)
    for c in COMPONENTS:
        ax[1, 0].plot(tho[z], ro["Tc"][c.name][mo][z] - 273.15, label=c.name)
    ax[1, 0].axhline(T_MIN_OP["Gimbal drive / motor"], color=lim_color, ls="--")
    ax[1, 0].set_title("Start-up zoom, power switched on at 06:50", fontsize=9); ax[1, 0].legend(fontsize=7)
    lbl = [s[0].replace(" (white)", "").replace("solar absorptivity", "alpha") for s in sens_res]
    ax[1, 1].barh(range(len(sens_res)), [s[1]["margin"] for s in sens_res], color="C1")
    ax[1, 1].set_yticks(range(len(sens_res))); ax[1, 1].set_yticklabels(lbl, fontsize=8); ax[1, 1].invert_yaxis()
    ax[1, 1].axvline(0, color="k", lw=1); ax[1, 1].set_xlabel("Minimum temperature margin over the day [K]")
    ax[1, 1].set_title("Summer sensitivity (natural convection)", fontsize=9)
    for a in ax.flat[:3]:
        a.set_xlabel("Solar time [h]"); a.set_ylabel("Temperature [C]"); a.grid(alpha=0.3)
    ax[0, 0].set_xlim(0, 24); ax[0, 1].set_xlim(0, 24)
    ax[1, 1].grid(alpha=0.3, axis="x")
    fig.suptitle("Item 4: day-night transient, HAA winter and sensitivities")
    fig.tight_layout(); fig.savefig("item4_winter_sensitivity.png", dpi=150)
    print("\nSaved: item4_summer.png, item4_winter_sensitivity.png")