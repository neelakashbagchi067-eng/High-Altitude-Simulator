"""
ITEM 2 + ITEM 3 - high-altitude anti-drone thermal study (SIH 4)

Item 2 : heat transfer coefficient vs altitude, compared with sea level
         - enclosure wall      : natural convection (Churchill-Chu) and wind (flat plate)
         - heat-sink channels  : natural convection (Bar-Cohen & Rohsenow) and fan-driven flow
Item 3 : steady-state resistance network of the electronics enclosure with a finned heat sink

Thermal network (all components sit on one base plate)

    T_j,i --[R_jc + R_tim]--> T_b --[R_base]--> T_fb --[fins: eta_o h A]--> ambient (air)
                               |
                               +--[R_int: internal conv + rad]--> T_w --[ext conv + rad]--> ambient/sky
                                                                    ^
                                                                    +-- absorbed solar Q_sun

Unknowns T_b (base plate), T_fb (fin base), T_w (enclosure wall) are solved together because
every h depends on temperature.

Needs altitude_convection.py and environment.py in the same folder.

ALL component powers, limits and dimensions below are ILLUSTRATIVE ASSUMPTIONS.
Replace them with your own design numbers (see the 'CHANGE THESE' blocks).

Run:  python thermal_network.py
"""

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, brentq

from altitude_convection import (air_props, air_density, isa_pressure, h_natural_vertical,
                                 h_wind_flat_plate, G, R_AIR, SIGMA, CP_AIR)
from environment import sky_temperature, environment_at, SCENARIOS


# =============================================================================
# 1. System description  (CHANGE THESE)
# =============================================================================
@dataclass
class Component:
    name: str
    Q: float            # W dissipated
    R_jc: float         # K/W junction-to-case (datasheet)
    tim_area: float     # m^2 contact area
    tim_thk: float      # m   thermal-interface-material thickness
    tim_k: float        # W/mK TIM conductivity
    T_lim_C: float      # deg C allowable (already derated) temperature

    @property
    def R_tim(self):
        return self.tim_thk / (self.tim_k * self.tim_area)

    @property
    def R_to_base(self):
        return self.R_jc + self.R_tim


COMPONENTS = [
    Component("Processor / SoC",       30.0, 0.5, 20e-3 * 20e-3, 0.2e-3, 3.0,  95.0),
    Component("RF power amplifier",    35.0, 0.8, 25e-3 * 25e-3, 0.2e-3, 3.0, 100.0),
    Component("Power electronics",     20.0, 1.0, 30e-3 * 30e-3, 0.3e-3, 3.0, 100.0),
    Component("Gimbal drive / motor",  15.0, 1.2, 30e-3 * 30e-3, 0.3e-3, 3.0,  90.0),
]


@dataclass
class HeatSink:
    W: float = 0.30        # m   width across the fins
    L: float = 0.20        # m   length along the (vertical) flow direction
    H: float = 0.040       # m   fin height
    t: float = 0.0025      # m   fin thickness
    S: float = 0.009       # m   fin spacing (gap)
    t_base: float = 0.008  # m   base plate thickness
    k: float = 167.0       # W/mK aluminium (6063/6061-class)

    @property
    def N(self):
        return int((self.W + self.S) / (self.t + self.S))

    @property
    def Lc(self):          # corrected fin length (convecting tip)
        return self.H + self.t / 2

    @property
    def A_fin(self):       # area of one fin (both faces + tip)
        return 2 * self.L * self.Lc

    @property
    def A_total(self):     # all fins + exposed base between fins
        return self.N * self.A_fin + (self.W - self.N * self.t) * self.L

    @property
    def R_base(self):      # 1-D conduction through base plate
        return self.t_base / (self.k * self.W * self.L)


@dataclass
class Enclosure:
    A_wall: float = 0.60       # m^2 outer wall area (excluding heat sink)
    L_wall: float = 0.25       # m   characteristic height/length for external convection
    A_int: float = 0.40        # m^2 internal exchange area (base plate/air/walls)
    L_int: float = 0.20        # m   internal characteristic height
    eps_ext: float = 0.90      # external emissivity (paint)
    alpha_sun: float = 0.60    # solar absorptivity of paint (olive-drab ~0.6; white ~0.25)
    A_sun: float = 0.30        # m^2 projected area facing the sun
    eps_int: float = 0.50      # effective internal emissivity (base plate <-> walls)
    alt_internal: float = None # altitude [m] of the internal air pressure (None = pressure-equalised)


HS = HeatSink()
ENC = Enclosure()


# =============================================================================
# 2. Fin analysis
# =============================================================================
def fin_efficiency(h, hs):
    """Straight rectangular fin with convecting tip (corrected length)."""
    P = 2 * (hs.L + hs.t)
    Ac = hs.L * hs.t
    m = np.sqrt(h * P / (hs.k * Ac))
    x = m * hs.Lc
    return np.tanh(x) / x if x > 1e-9 else 1.0


def overall_surface_efficiency(h, hs):
    eta_f = fin_efficiency(h, hs)
    return 1.0 - hs.N * hs.A_fin / hs.A_total * (1.0 - eta_f), eta_f


def h_channel_natural(T_fb, T_amb, hs, alt):
    """
    Natural convection between vertical isothermal fins, symmetric channel (Bar-Cohen & Rohsenow).
    Nu_S = [576/X^2 + 2.873/X^0.5]^-0.5,  X = Ra_S * S/L  ->  h = Nu_S k / S.
    The large-spacing limit is the single-plate value, the small-spacing limit is fully developed flow.
    """
    dT = max(abs(T_fb - T_amb), 0.5)
    p = air_props(0.5 * (T_fb + T_amb), alt)
    RaS = G * p["beta"] * dT * hs.S ** 3 / (p["nu"] * p["alpha"])
    X = RaS * hs.S / hs.L
    Nu = (576.0 / X ** 2 + 2.873 / X ** 0.5) ** -0.5
    return Nu * p["k"] / hs.S


def h_channel_forced(T_fb, T_amb, hs, alt, Vdot):
    """
    Fan-driven flow through the fin channels (all fan flow assumed to pass through the channels).
    Laminar: parallel-plate fully developed Nu = 7.54 plus a Hausen-type developing-flow term.
    Turbulent (Re > 1e4): Dittus-Boelter. Linear blend in between.
    Returns h, Re, mdot [kg/s].
    """
    p = air_props(0.5 * (T_fb + T_amb), alt)
    Dh = 2 * hs.S * hs.H / (hs.S + hs.H)
    V = Vdot / (hs.N * hs.S * hs.H)
    Re = V * Dh / p["nu"]
    Gz = max(Re * p["Pr"] * Dh / hs.L, 1e-9)
    Nu_lam = 7.54 + 0.0668 * Gz / (1 + 0.04 * Gz ** (2 / 3))
    if Re <= 2300:
        Nu = Nu_lam
    elif Re >= 1e4:
        Nu = 0.023 * Re ** 0.8 * p["Pr"] ** 0.4
    else:
        Gz_t = 2300 * p["Pr"] * Dh / hs.L
        Nu_a = 7.54 + 0.0668 * Gz_t / (1 + 0.04 * Gz_t ** (2 / 3))
        Nu_b = 0.023 * 1e4 ** 0.8 * p["Pr"] ** 0.4
        Nu = Nu_a + (Re - 2300) / (1e4 - 2300) * (Nu_b - Nu_a)
    rho_in = isa_pressure(alt) / (R_AIR * T_amb)
    return Nu * p["k"] / Dh, Re, rho_in * Vdot


def heat_sink_heat(T_fb, T_amb, hs, alt, mode, Vdot=0.0):
    """Heat [W] removed by the fins when the fin-base temperature is T_fb. Returns dict."""
    if mode == "natural":
        h = h_channel_natural(T_fb, T_amb, hs, alt)
        eta_o, eta_f = overall_surface_efficiency(h, hs)
        Q = eta_o * h * hs.A_total * (T_fb - T_amb)
        return dict(Q=Q, h=h, eta_o=eta_o, eta_f=eta_f, Re=np.nan, dT_air=np.nan, mdot=np.nan)
    h, Re, mdot = h_channel_forced(T_fb, T_amb, hs, alt, Vdot)
    eta_o, eta_f = overall_surface_efficiency(h, hs)
    NTU = eta_o * h * hs.A_total / (mdot * CP_AIR)
    eps = 1.0 - np.exp(-NTU)                      # fin-base ~ isothermal, air heats up
    Q = eps * mdot * CP_AIR * (T_fb - T_amb)
    return dict(Q=Q, h=h, eta_o=eta_o, eta_f=eta_f, Re=Re, dT_air=Q / (mdot * CP_AIR), mdot=mdot)


# =============================================================================
# 3. Steady-state network solution
# =============================================================================
def solve_system(cond, comps=COMPONENTS, hs=HS, enc=ENC, mode="natural", Vdot=0.0):
    """
    cond = dict(T_amb_C, altitude, G_sun [W/m2], U_wind [m/s])
    mode : heat sink 'natural' or 'fan' (volumetric flow Vdot [m^3/s], taken as altitude-independent)
    """
    T_amb = cond["T_amb_C"] + 273.15
    alt = cond["altitude"]
    Qtot = sum(c.Q for c in comps)
    T_sky = sky_temperature(cond["T_amb_C"]) + 273.15
    T_rs = 0.5 * (T_amb + T_sky)                   # 50 % sky, 50 % surroundings (assumption)
    alt_int = alt if enc.alt_internal is None else enc.alt_internal
    Qsun = enc.alpha_sun * cond["G_sun"] * enc.A_sun
    fan_mode = "natural" if mode == "natural" else "fan"

    def wall_UA(Tw):
        """Returns total wall heat loss [W] (convection to air + radiation to sky/surroundings), h_conv, h_rad."""
        if cond["U_wind"] > 0.5:
            hc, _ = h_wind_flat_plate(Tw, T_amb, enc.L_wall, cond["U_wind"], alt)
        else:
            hc, _ = h_natural_vertical(Tw, T_amb, enc.L_wall, alt)
        hr = enc.eps_ext * SIGMA * (Tw + T_rs) * (Tw ** 2 + T_rs ** 2)
        Q_loss = enc.A_wall * (hc * (Tw - T_amb) + hr * (Tw - T_rs))   # radiation is driven by T_w - T_rs
        return Q_loss, hc, hr

    def int_UA(Tb, Tw):
        hc, _ = h_natural_vertical(Tb, Tw, enc.L_int, alt_int)
        hr = enc.eps_int * SIGMA * (Tb + Tw) * (Tb ** 2 + Tw ** 2)
        return enc.A_int * (hc + hr), hc, hr

    def residuals(x):
        Tb, Tfb, Tw = x
        Q_base = (Tb - Tfb) / hs.R_base
        Q_fin = heat_sink_heat(Tfb, T_amb, hs, alt, fan_mode, Vdot)["Q"]
        UAi, _, _ = int_UA(Tb, Tw)
        Q_w = (Tb - Tw) * UAi
        Q_loss, _, _ = wall_UA(Tw)
        return [Q_base - Q_fin, Qtot - Q_fin - Q_w, Q_w + Qsun - Q_loss]

    sol, ok = None, False
    for dT0 in (30.0, 60.0, 15.0, 90.0):
        x0 = [T_amb + dT0, T_amb + 0.95 * dT0, T_amb + 0.4 * dT0]
        s, info, ier, _ = fsolve(residuals, x0, full_output=True, xtol=1e-9)
        if ier == 1 and np.max(np.abs(residuals(s))) < 1e-4:
            sol, ok = s, True
            break
    if not ok:
        raise RuntimeError("steady-state solution did not converge")

    Tb, Tfb, Tw = sol
    fins = heat_sink_heat(Tfb, T_amb, hs, alt, fan_mode, Vdot)
    UAi, hci, hri = int_UA(Tb, Tw)
    _, hce, hre = wall_UA(Tw)
    Tc = {c.name: Tb + c.Q * c.R_to_base for c in comps}
    margin = {c.name: c.T_lim_C - (Tc[c.name] - 273.15) for c in comps}
    return dict(Tb=Tb, Tfb=Tfb, Tw=Tw, Tc=Tc, margin=margin, Q_fin=fins["Q"],
                Q_wall_path=(Tb - Tw) * UAi, Q_sun=Qsun, Qtot=Qtot,
                h_fin=fins["h"], eta_o=fins["eta_o"], eta_f=fins["eta_f"], Re=fins["Re"],
                dT_air=fins["dT_air"], mdot=fins["mdot"], h_wall_conv=hce, h_wall_rad=hre,
                h_int_conv=hci, h_int_rad=hri,
                R_hs_eff=(Tfb - T_amb) / fins["Q"] if fins["Q"] > 0 else np.inf,
                T_amb=T_amb, min_margin=min(margin.values()),
                T_max_C=max(Tc.values()) - 273.15)


# =============================================================================
# 4. ITEM 2 - h vs altitude
# =============================================================================
def h_vs_altitude(alts, T_amb_C=20.0, dT_wall=30.0, dT_fin=40.0, U=5.0, Vdot=0.012):
    T_amb = T_amb_C + 273.15
    out = dict(wall_nat=[], wall_wind=[], fin_nat=[], fin_fan=[], Re_fan=[], dTair=[], mdot=[], Rhs_nat=[], Rhs_fan=[])
    for a in alts:
        out["wall_nat"].append(h_natural_vertical(T_amb + dT_wall, T_amb, ENC.L_wall, a)[0])
        out["wall_wind"].append(h_wind_flat_plate(T_amb + dT_wall, T_amb, ENC.L_wall, U, a)[0])
        n = heat_sink_heat(T_amb + dT_fin, T_amb, HS, a, "natural")
        f = heat_sink_heat(T_amb + dT_fin, T_amb, HS, a, "fan", Vdot)
        out["fin_nat"].append(n["h"]); out["fin_fan"].append(f["h"])
        out["Re_fan"].append(f["Re"]); out["mdot"].append(f["mdot"])
        out["Rhs_nat"].append(dT_fin / n["Q"]); out["Rhs_fan"].append(dT_fin / f["Q"])
        out["dTair"].append(f["dT_air"])
    return {k: np.array(v) for k, v in out.items()}


# =============================================================================
# 5. Main
# =============================================================================
if __name__ == "__main__":
    Qtot = sum(c.Q for c in COMPONENTS)
    print(f"Total dissipation = {Qtot:.0f} W;  heat sink: {HS.N} fins, H = {HS.H*1e3:.0f} mm, "
          f"t = {HS.t*1e3:.1f} mm, S = {HS.S*1e3:.1f} mm, A_total = {HS.A_total:.3f} m2, "
          f"R_base = {HS.R_base*1e3:.1f} mK/W")
    print("Component resistances (junction to base plate):")
    for c in COMPONENTS:
        print(f"   {c.name:<22s} Q={c.Q:4.0f} W  R_jc={c.R_jc:.2f}  R_tim={c.R_tim:.3f}  "
              f"-> rise above base = {c.Q*c.R_to_base:5.1f} K, limit {c.T_lim_C:.0f} C")

    # ---------------- Item 2 --------------------------------------------------------
    alts = np.linspace(0, 6000, 25)
    hv = h_vs_altitude(alts)
    print("\nITEM 2: h vs altitude at T_amb = 20 C (isolates the pressure/density effect)")
    print(f"{'alt[m]':>7} {'wall nat':>9} {'wall wind5':>11} {'fin nat':>8} {'fin fan':>8} {'Re_fan':>7} "
          f"{'air rise[K]':>12} {'R_hs nat':>9} {'R_hs fan':>9}")
    for i in range(0, len(alts), 4):
        print(f"{alts[i]:7.0f} {hv['wall_nat'][i]:9.2f} {hv['wall_wind'][i]:11.2f} {hv['fin_nat'][i]:8.2f} "
              f"{hv['fin_fan'][i]:8.1f} {hv['Re_fan'][i]:7.0f} {hv['dTair'][i]:12.1f} "
              f"{hv['Rhs_nat'][i]:9.3f} {hv['Rhs_fan'][i]:9.3f}")
    i4 = int(np.argmin(np.abs(alts - 4500)))
    print(f"\nRatio at ~4500 m to sea level:  wall natural {hv['wall_nat'][i4]/hv['wall_nat'][0]:.2f}, "
          f"wall wind {hv['wall_wind'][i4]/hv['wall_wind'][0]:.2f}, fin natural {hv['fin_nat'][i4]/hv['fin_nat'][0]:.2f}, "
          f"fin fan {hv['fin_fan'][i4]/hv['fin_fan'][0]:.2f}; heat-sink R: natural x{hv['Rhs_nat'][i4]/hv['Rhs_nat'][0]:.2f}, "
          f"fan x{hv['Rhs_fan'][i4]/hv['Rhs_fan'][0]:.2f}")

    # ---------------- Item 3: design cases -----------------------------------------
    leh_s = SCENARIOS["HAA summer"]
    leh_w = SCENARIOS["HAA winter (severe)"]
    noon_s, noon_w = environment_at(12.0, leh_s), environment_at(12.0, leh_w)
    cases = [
        ("Sea level 35C, no sun",       dict(T_amb_C=35.0, altitude=0.0, G_sun=0.0, U_wind=0.0), "natural", 0.0),
        ("4500 m summer noon, sun",     dict(T_amb_C=20.0, altitude=4500.0, G_sun=noon_s["Gbn"], U_wind=0.0), "natural", 0.0),
        ("  same + fan 12 L/s",         dict(T_amb_C=20.0, altitude=4500.0, G_sun=noon_s["Gbn"], U_wind=0.0), "fan", 0.012),
        ("4500 m summer noon + wind",   dict(T_amb_C=20.0, altitude=4500.0, G_sun=noon_s["Gbn"], U_wind=6.0), "natural", 0.0),
        ("4500 m winter night -35C",    dict(T_amb_C=-35.0, altitude=4500.0, G_sun=0.0, U_wind=0.0), "natural", 0.0),
        ("4500 m winter noon, sun",     dict(T_amb_C=-10.0, altitude=4500.0, G_sun=noon_w["Gbn"], U_wind=0.0), "natural", 0.0),
    ]
    print("\nITEM 3: steady-state results   (margin = limit - temperature; negative = over limit)")
    res_cases = []
    for name, cond, mode, V in cases:
        r = solve_system(cond, mode=mode, Vdot=V)
        res_cases.append((name, r))
        print(f"\n{name}: T_amb = {cond['T_amb_C']:.0f} C, sun on wall = {cond['G_sun']:.0f} W/m2, wind {cond['U_wind']:.0f} m/s")
        print(f"   T_base = {r['Tb']-273.15:6.1f} C, T_fin-base = {r['Tfb']-273.15:6.1f} C, T_wall = {r['Tw']-273.15:6.1f} C")
        print(f"   heat sink takes {r['Q_fin']:5.1f} W (h = {r['h_fin']:.1f}, eta_o = {r['eta_o']:.2f}, "
              f"R_hs = {r['R_hs_eff']:.3f} K/W), wall path {r['Q_wall_path']:5.1f} W, absorbed sun {r['Q_sun']:.0f} W"
              + (f", air rise {r['dT_air']:.1f} K" if mode == "fan" else ""))
        print("   margins [K]: " + ", ".join(f"{k.split(' /')[0].split(' (')[0]} {v:+.0f}" for k, v in r["margin"].items()))
    # energy check on last case (heat in = heat out)
    r = res_cases[1][1]
    print("\nEnergy-balance check (4500 m summer noon): internal Q = "
          f"{r['Qtot']:.1f} W, fins = {r['Q_fin']:.1f} W, wall path = {r['Q_wall_path']:.1f} W -> "
          f"residual = {r['Qtot']-r['Q_fin']-r['Q_wall_path']:.2e} W")

    # ---------------- altitude sweep of peak component temperature ------------------
    alt_sw = np.linspace(0, 6000, 13)
    Tmax_nat, Tmax_fan = [], []
    for a in alt_sw:
        c = dict(T_amb_C=20.0, altitude=a, G_sun=0.0, U_wind=0.0)
        Tmax_nat.append(solve_system(c)["T_max_C"])
        Tmax_fan.append(solve_system(c, mode="fan", Vdot=0.012)["T_max_C"])

    # ---------------- fin-spacing sweep ---------------------------------------------
    S_list = np.arange(0.003, 0.0301, 0.001)
    sweep = {}
    for a in (0.0, 4500.0):
        vals = []
        for S in S_list:
            hs = HeatSink(S=S)
            vals.append(solve_system(dict(T_amb_C=20.0, altitude=a, G_sun=0.0, U_wind=0.0), hs=hs)["T_max_C"])
        sweep[a] = np.array(vals)
    print("\nFin-spacing optimum for natural convection (T_amb = 20 C, no sun):")
    for a, v in sweep.items():
        i = int(np.argmin(v))
        print(f"   {a:5.0f} m: best S = {S_list[i]*1e3:.0f} mm  (peak component {v[i]:.1f} C);  "
              f"at S = 9 mm: {v[int(round((0.009-0.003)/0.001))]:.1f} C")

    # ---------------- fan sizing: volumetric flow needed to hold the limit ----------
    def fan_needed(a, T_amb_C=20.0, Gs=noon_s["Gbn"]):
        c = dict(T_amb_C=T_amb_C, altitude=a, G_sun=Gs, U_wind=0.0)
        if solve_system(c)["min_margin"] >= 0:
            return 0.0
        f = lambda V: solve_system(c, mode="fan", Vdot=V)["min_margin"]
        if f(0.2) < 0:
            return np.nan
        return brentq(f, 1e-4, 0.2, xtol=1e-6)
    fan_alts = np.linspace(0, 6000, 7)
    fan_req = np.array([fan_needed(a) for a in fan_alts])
    print("\nFan volumetric flow needed to keep every component within its limit (20 C ambient, noon sun):")
    for a, v in zip(fan_alts, fan_req):
        txt = "natural convection is enough" if v == 0 else (f"{v*1e3:5.1f} L/s ({v*2118.9:4.0f} CFM)" if np.isfinite(v) else "not achievable with this heat sink")
        print(f"   {a:5.0f} m: {txt}")

    # ---------------- Figure 1: item 2 ----------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    labels = [("wall_nat", "wall, natural"), ("wall_wind", "wall, wind 5 m/s"),
              ("fin_nat", "fin channels, natural"), ("fin_fan", "fin channels, fan 12 L/s")]
    for key, lab in labels:
        ax[0, 0].plot(alts, hv[key], label=lab)
        ax[0, 1].plot(alts, hv[key] / hv[key][0], label=lab)
    ax[0, 0].set_ylabel("h [W/m$^2$K]"); ax[0, 0].set_yscale("log"); ax[0, 0].legend(fontsize=8)
    ax[0, 1].set_ylabel("h / h(sea level)"); ax[0, 1].set_ylim(0.35, 1.05); ax[0, 1].legend(fontsize=8)
    ax[1, 0].plot(alts, hv["dTair"], "C3"); ax[1, 0].set_ylabel("Air temperature rise through fan-cooled fins [K]")
    ax2 = ax[1, 0].twinx(); ax2.plot(alts, hv["mdot"] * 1e3, "C2--"); ax2.set_ylabel("Air mass flow [g/s] (dashed)")
    ax[1, 1].plot(alts, hv["Rhs_nat"], label="natural"); ax[1, 1].plot(alts, hv["Rhs_fan"], label="fan 12 L/s")
    ax[1, 1].set_ylabel("Heat-sink resistance to ambient [K/W]"); ax[1, 1].legend()
    for a in ax.flat:
        a.set_xlabel("Altitude [m]"); a.grid(alpha=0.3)
    fig.suptitle("Item 2: convection coefficients vs altitude (T_amb = 20 C)")
    fig.tight_layout(); fig.savefig("item2_h_vs_altitude.png", dpi=150)

    # ---------------- Figure 2: item 3 ----------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))
    names = [c.name for c in COMPONENTS]
    w = 0.8 / len(names)
    for j, nm in enumerate(names):
        ax[0, 0].bar(np.arange(len(res_cases)) + j * w, [r["margin"][nm] for _, r in res_cases], w, label=nm)
    ax[0, 0].axhline(0, c="k", lw=1)
    ax[0, 0].set_xticks(np.arange(len(res_cases)) + 0.4 - w / 2)
    short = ["Sea level\n35 C", "4500 m\nsummer noon", "summer noon\n+ fan", "summer noon\n+ 6 m/s wind",
             "winter night\n-35 C", "winter noon\nwith sun"]
    ax[0, 0].set_xticklabels(short[:len(res_cases)], fontsize=7)
    ax[0, 0].set_ylabel("Temperature margin to limit [K]"); ax[0, 0].legend(fontsize=7)
    ax[0, 1].plot(alt_sw, Tmax_nat, "o-", label="natural convection")
    ax[0, 1].plot(alt_sw, Tmax_fan, "s-", label="fan 12 L/s")
    ax[0, 1].set_ylabel("Peak component temperature [C]"); ax[0, 1].set_xlabel("Altitude [m]")
    ax[0, 1].set_title("20 C ambient, no sun", fontsize=9); ax[0, 1].legend()
    for a, v in sweep.items():
        ax[1, 0].plot(S_list * 1e3, v, label=f"{a:.0f} m")
        i = int(np.argmin(v)); ax[1, 0].plot(S_list[i] * 1e3, v[i], "k*", ms=10)
    ax[1, 0].set_xlabel("Fin spacing S [mm]"); ax[1, 0].set_ylabel("Peak component temperature [C]")
    ax[1, 0].set_title("Natural convection, fixed envelope (star = optimum)", fontsize=9); ax[1, 0].legend()
    ok = np.isfinite(fan_req)
    ax[1, 1].plot(fan_alts[ok], fan_req[ok] * 1e3, "o-"); ax[1, 1].set_xlabel("Altitude [m]")
    ax[1, 1].set_ylabel("Fan volumetric flow needed [L/s]")
    ax[1, 1].set_title("Noon sun, 20 C ambient (0 = natural convection suffices)", fontsize=9)
    for a in (ax[0, 1], ax[1, 1]):
        a.grid(alpha=0.3); a.set_xlabel("Altitude [m]")
    ax[0, 0].grid(alpha=0.3, axis="y"); ax[1, 0].grid(alpha=0.3)
    fig.suptitle("Item 3: steady-state enclosure and heat-sink results")
    fig.tight_layout(); fig.savefig("item3_enclosure_network.png", dpi=150)
    print("\nSaved: item2_h_vs_altitude.png, item3_enclosure_network.png")