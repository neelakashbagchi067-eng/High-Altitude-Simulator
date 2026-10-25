"""
High-Altitude Anti-Drone System - Conceptual Simulation
--------------------------------------------------------
Scenario
  * A high-altitude platform (HAP, e.g. stratospheric balloon/airship) hovers above
    a protected asset and provides wide-area radar surveillance.
  * Hostile drones approach the asset from random bearings.
  * The HAP detects drones probabilistically (detection falls off with range, noisy
    measurements), tracks them with an alpha-beta filter, and cues interceptor
    drones launched from a ground base.
  * Interceptors fly proportional navigation (PN) toward the tracked target.

Everything is a simplified, educational model (point masses, constant speeds).
Run:  python anti_drone_sim.py            -> animated 3D view + summary plots
      python anti_drone_sim.py --save     -> also saves anti_drone.gif
Requires: numpy, matplotlib  (pillow for --save)
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ----------------------------- PARAMETERS ---------------------------------
SEED            = 7
DT              = 0.05         # s, simulation time step (small enough to avoid tunnelling)
T_MAX           = 160.0        # s

# High-altitude platform + radar
HAP_POS         = np.array([0.0, 0.0, 18000.0])   # m (change altitude here)
RADAR_RANGE     = 30000.0      # m, max slant range
RADAR_PD0       = 0.90         # detection probability at close range
RADAR_SIGMA     = 12.0         # m, measurement noise (1-sigma per axis)
RADAR_PERIOD    = 0.5          # s between radar scans

# Tracker (alpha-beta filter)
ALPHA, BETA     = 0.6, 0.25
CONFIRM_HITS    = 3            # detections needed to confirm a track

# Protected asset / defence
ASSET_POS       = np.array([0.0, 0.0, 0.0])
ENGAGE_RADIUS   = 3500.0       # m from asset: start engaging confirmed tracks
BREACH_RADIUS   = 100.0        # m: a drone this close = "leaker"
N_INTERCEPTORS  = 8
LAUNCH_DELAY    = 1.0          # s between launches (one launcher)

# Hostile drones
N_DRONES        = 5
DRONE_SPEED     = 35.0         # m/s
SPAWN_RANGE     = 4300.0       # m horizontal distance from asset

# Interceptors
INT_SPEED       = 95.0         # m/s (constant speed model)
INT_MAX_ACCEL   = 120.0        # m/s^2 lateral
PN_GAIN         = 4.0
KILL_RADIUS     = 10.0         # m
INT_LIFETIME    = 45.0         # s
INT_BASE        = np.array([60.0, 0.0, 0.0])
# --------------------------------------------------------------------------

rng = np.random.default_rng(SEED)


class Drone:
    def __init__(self, idx):
        ang = rng.uniform(0, 2 * np.pi)
        alt = rng.uniform(250, 700)
        self.id = idx
        self.pos = np.array([SPAWN_RANGE * np.cos(ang), SPAWN_RANGE * np.sin(ang), alt])
        aim = ASSET_POS + np.array([0, 0, 20.0])
        d = aim - self.pos
        self.vel = DRONE_SPEED * d / np.linalg.norm(d)
        self.alive = True
        self.breached = False
        # track state
        self.hits = 0
        self.confirmed = False
        self.est_pos = None
        self.est_vel = np.zeros(3)
        self.last_update = 0.0
        self.assigned = False
        self.first_detect_t = None

    def step(self, dt):
        if self.alive and not self.breached:
            self.pos = self.pos + self.vel * dt


class Interceptor:
    def __init__(self, target, t):
        self.pos = INT_BASE.copy()
        self.target = target
        d = target.est_pos - self.pos
        d[2] = max(d[2], 50)
        self.vel = INT_SPEED * d / np.linalg.norm(d)
        self.t0 = t
        self.active = True
        self.result = None  # 'kill' / 'timeout'

    def step(self, dt, t):
        if not self.active:
            return
        tgt = self.target
        # Use tracker estimate (extrapolated to now) as the cue
        est = tgt.est_pos + tgt.est_vel * (t - tgt.last_update)
        r = est - self.pos
        v_rel = tgt.est_vel - self.vel
        rng_ = np.linalg.norm(r) + 1e-9
        omega = np.cross(r, v_rel) / rng_**2          # LOS rotation vector
        acc = PN_GAIN * np.cross(omega, self.vel)     # pure proportional nav
        a = np.linalg.norm(acc)
        if a > INT_MAX_ACCEL:
            acc *= INT_MAX_ACCEL / a
        self.vel = self.vel + acc * dt
        self.vel *= INT_SPEED / np.linalg.norm(self.vel)
        self.pos = self.pos + self.vel * dt

        # Terminal kill check against TRUE target position
        if tgt.alive and np.linalg.norm(tgt.pos - self.pos) < KILL_RADIUS:
            tgt.alive = False
            self.active = False
            self.result = "kill"
        elif t - self.t0 > INT_LIFETIME or not tgt.alive:
            self.active = False
            self.result = self.result or "timeout"


def radar_scan(drones, t):
    """HAP radar: probabilistic detection + noisy measurement + alpha-beta update."""
    detections = 0
    for d in drones:
        if not d.alive or d.breached:
            continue
        slant = np.linalg.norm(d.pos - HAP_POS)
        if slant > RADAR_RANGE:
            continue
        pd = RADAR_PD0 * (1 - (slant / RADAR_RANGE) ** 4)
        if rng.random() > pd:
            continue
        detections += 1
        z = d.pos + rng.normal(0, RADAR_SIGMA, 3)
        if d.est_pos is None:
            d.est_pos, d.est_vel, d.last_update = z, np.zeros(3), t
            d.first_detect_t = t
        else:
            dt = max(t - d.last_update, 1e-3)
            pred = d.est_pos + d.est_vel * dt
            resid = z - pred
            d.est_pos = pred + ALPHA * resid
            d.est_vel = d.est_vel + BETA * resid / dt
            d.last_update = t
        d.hits += 1
        if d.hits >= CONFIRM_HITS:
            d.confirmed = True
    return detections


def run():
    drones = [Drone(i) for i in range(N_DRONES)]
    interceptors, log = [], []
    next_scan, last_launch, t = 0.0, -99.0, 0.0
    stats = {"detections": 0, "launched": 0}
    history = {"t": [], "drones": [], "alive": [], "ints": [], "int_active": [], "est": []}

    while t < T_MAX:
        # Radar + tracking
        if t >= next_scan:
            stats["detections"] += radar_scan(drones, t)
            next_scan += RADAR_PERIOD

        # Threat assessment: engage closest confirmed, unassigned tracks first
        threats = [d for d in drones if d.alive and d.confirmed and not d.assigned
                   and np.linalg.norm(d.est_pos[:2] - ASSET_POS[:2]) < ENGAGE_RADIUS]
        threats.sort(key=lambda d: np.linalg.norm(d.est_pos - ASSET_POS))
        if threats and stats["launched"] < N_INTERCEPTORS and t - last_launch >= LAUNCH_DELAY:
            tgt = threats[0]
            tgt.assigned = True
            interceptors.append(Interceptor(tgt, t))
            stats["launched"] += 1
            last_launch = t
            log.append(f"t={t:6.1f}s  Launch interceptor #{stats['launched']} -> drone {tgt.id}")

        # Re-assign if an interceptor timed out against a live target
        for it in interceptors:
            if not it.active and it.result == "timeout" and it.target.alive:
                it.target.assigned = False

        # Dynamics
        for d in drones:
            d.step(DT)
            if d.alive and not d.breached and np.linalg.norm(d.pos - ASSET_POS) < BREACH_RADIUS:
                d.breached = True
                log.append(f"t={t:6.1f}s  !! Drone {d.id} BREACHED the perimeter")
        for it in interceptors:
            was = it.active
            it.step(DT, t)
            if was and it.result == "kill":
                log.append(f"t={t:6.1f}s  Drone {it.target.id} NEUTRALISED")

        # Record frame
        history["t"].append(t)
        history["drones"].append(np.array([d.pos for d in drones]))
        history["alive"].append(np.array([d.alive for d in drones]))
        history["ints"].append(np.array([i.pos for i in interceptors]).reshape(-1, 3))
        history["int_active"].append(np.array([i.active for i in interceptors]))
        history["est"].append(np.array([d.est_pos if d.est_pos is not None else [np.nan] * 3
                                        for d in drones]))
        t += DT
        if all((not d.alive) or d.breached for d in drones):
            break

    stats["kills"] = sum(not d.alive for d in drones)
    stats["leakers"] = sum(d.breached for d in drones)
    stats["first_detect"] = [d.first_detect_t for d in drones]
    return drones, interceptors, history, stats, log


def animate(history, stats, save=False):
    fig = plt.figure(figsize=(12, 6))
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    axt = fig.add_subplot(1, 2, 2)

    lim = SPAWN_RANGE * 1.1
    ax3.set_xlim(-lim, lim); ax3.set_ylim(-lim, lim); ax3.set_zlim(0, 1500)
    ax3.set_xlabel("x (m)"); ax3.set_ylabel("y (m)"); ax3.set_zlabel("z (m)")
    ax3.set_title("Engagement zone (HAP at %.0f km altitude, not to scale)" % (HAP_POS[2] / 1000))
    ax3.scatter(*ASSET_POS, c="green", s=120, marker="s", label="Protected asset")
    ax3.scatter(*INT_BASE, c="blue", s=60, marker="^", label="Interceptor base")
    th = np.linspace(0, 2 * np.pi, 100)
    ax3.plot(ENGAGE_RADIUS * np.cos(th), ENGAGE_RADIUS * np.sin(th), 0, "k--", lw=0.8,
             label="Engagement radius")

    dr_sc = ax3.scatter([], [], [], c="red", s=30, label="Hostile drone")
    int_sc = ax3.scatter([], [], [], c="blue", s=20, label="Interceptor")
    trail_d = [ax3.plot([], [], [], "r-", lw=0.6, alpha=0.5)[0] for _ in range(N_DRONES)]
    ax3.legend(loc="upper left", fontsize=7)
    txt = ax3.text2D(0.02, 0.02, "", transform=ax3.transAxes, fontsize=8)

    # Right panel: range-to-asset vs time
    T = np.array(history["t"])
    dpos = np.array(history["drones"])
    rng_asset = np.linalg.norm(dpos - ASSET_POS, axis=2)
    for i in range(N_DRONES):
        axt.plot(T, rng_asset[:, i], label=f"Drone {i}")
    axt.axhline(ENGAGE_RADIUS, color="k", ls="--", lw=0.8)
    axt.axhline(BREACH_RADIUS, color="r", ls=":", lw=0.8)
    axt.set_xlabel("time (s)"); axt.set_ylabel("distance to asset (m)")
    axt.set_title(f"Kills: {stats['kills']}/{N_DRONES} | Leakers: {stats['leakers']} | "
                  f"Interceptors used: {stats['launched']}")
    axt.legend(fontsize=7)
    cursor = axt.axvline(0, color="gray")

    step = 4  # frames skipped for speed

    def update(k):
        k = min(k * step, len(T) - 1)
        alive = history["alive"][k]
        p = history["drones"][k]
        dr_sc._offsets3d = (p[alive, 0], p[alive, 1], p[alive, 2])
        ip = history["ints"][k]
        ia = history["int_active"][k]
        if len(ip):
            int_sc._offsets3d = (ip[ia, 0], ip[ia, 1], ip[ia, 2])
        else:
            int_sc._offsets3d = ([], [], [])
        for i, ln in enumerate(trail_d):
            tr = dpos[:k + 1, i]
            ln.set_data(tr[:, 0], tr[:, 1]); ln.set_3d_properties(tr[:, 2])
        cursor.set_xdata([T[k], T[k]])
        txt.set_text(f"t = {T[k]:.1f} s | drones alive: {int(alive.sum())}")
        return dr_sc, int_sc, cursor

    ani = FuncAnimation(fig, update, frames=len(T) // step + 1, interval=50, blit=False)
    if save:
        ani.save("anti_drone.gif", writer="pillow", fps=20)
        print("Saved anti_drone.gif")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    drones, ints, hist, stats, log = run()
    print("\n".join(log))
    print("\n--- SUMMARY ---")
    print(f"Drones: {N_DRONES} | Detections (radar hits): {stats['detections']}")
    print(f"Interceptors launched: {stats['launched']} | Kills: {stats['kills']} | "
          f"Leakers: {stats['leakers']}")
    for d in drones:
        fd = stats["first_detect"][d.id]
        print(f"  Drone {d.id}: first detected at "
              f"{'never' if fd is None else f'{fd:.1f} s'}"
              f" -> {'NEUTRALISED' if not d.alive else 'BREACHED' if d.breached else 'survived'}")
    animate(hist, stats, save="--save" in sys.argv)
