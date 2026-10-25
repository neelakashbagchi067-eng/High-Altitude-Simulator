
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# =========================================================
# PAGE SETUP
# =========================================================

st.set_page_config(
    page_title="High-Altitude Performance Simulator",
    page_icon="🌡️",
    layout="wide"
)

st.title("High-Altitude Environmental Performance Simulator")

st.write(
    "An educational simulator for exploring thermal behaviour "
    "and standard atmospheric conditions."
)

st.warning(
    "This is an illustrative software model. Its parameters and "
    "outputs are not validated predictions of real equipment."
)

# =========================================================
# USER INPUTS
# All configurable inputs are entered manually.
# =========================================================

st.header("1. Enter Simulation Parameters")

st.write(
    "Enter the values below. The displayed initial values are "
    "examples and can be changed."
)

with st.form("simulation_input_form"):

    st.subheader("Thermal simulation inputs")

    col1, col2 = st.columns(2)

    with col1:
        ambient_temp = st.number_input(
            "Ambient temperature (°C)",
            min_value=-100.0,
            max_value=60.0,
            value=-20.0,
            step=1.0
        )

        initial_temp = st.number_input(
            "Initial component temperature (°C)",
            min_value=-100.0,
            max_value=150.0,
            value=20.0,
            step=1.0
        )

        heat_generated = st.number_input(
            "Heat generation (W)",
            min_value=0.0,
            max_value=10000.0,
            value=8.0,
            step=1.0
        )

        duration_minutes = st.number_input(
            "Simulation duration (minutes)",
            min_value=1.0,
            max_value=1440.0,
            value=60.0,
            step=5.0
        )

    with col2:
        thermal_capacity = st.number_input(
            "Thermal capacity (J/°C)",
            min_value=1.0,
            max_value=1000000.0,
            value=500.0,
            step=50.0
        )

        heat_transfer = st.number_input(
            "Heat-transfer coefficient (W/°C)",
            min_value=0.001,
            max_value=10000.0,
            value=0.8,
            step=0.1,
            format="%.3f"
        )

        time_step = st.number_input(
            "Calculation time step (seconds)",
            min_value=0.1,
            max_value=60.0,
            value=10.0,
            step=1.0
        )

        scenario_offset = st.number_input(
            "Scenario temperature offset (°C)",
            min_value=0.0,
            max_value=30.0,
            value=10.0,
            step=1.0
        )

    st.subheader("Atmospheric inputs")

    altitude_m = st.number_input(
        "Altitude above sea level (m)",
        min_value=0.0,
        max_value=11000.0,
        value=3000.0,
        step=250.0
    )

    st.subheader("Sensitivity analysis inputs")

    sensitivity_percent = st.number_input(
        "Parameter variation (%)",
        min_value=1.0,
        max_value=50.0,
        value=20.0,
        step=1.0
    )

    run_simulation = st.form_submit_button(
        "Run Simulation",
        type="primary"
    )

# =========================================================
# SIMULATION FUNCTIONS
# =========================================================

def simulate(
    ambient,
    initial,
    power,
    duration_seconds,
    capacity,
    transfer,
    requested_dt
):
    """Simulate a single thermal mass using an explicit time step."""

    if capacity <= 0 or transfer <= 0:
        raise ValueError(
            "Thermal capacity and heat transfer must be positive."
        )

    if duration_seconds <= 0 or requested_dt <= 0:
        raise ValueError(
            "Duration and time step must be positive."
        )

    steps = max(1, math.ceil(duration_seconds / requested_dt))
    actual_dt = duration_seconds / steps

    time = np.linspace(0, duration_seconds, steps + 1)
    temp = np.zeros(steps + 1)
    temp[0] = initial

    for i in range(1, len(time)):
        rate = (
            power - transfer * (temp[i - 1] - ambient)
        ) / capacity

        temp[i] = temp[i - 1] + rate * actual_dt

    equilibrium = ambient + power / transfer

    return time, temp, equilibrium


def standard_atmosphere(altitude):
    """Approximate standard atmosphere from 0 to 11,000 metres."""

    temperature_k = 288.15 - 0.0065 * altitude

    pressure_pa = 101325.0 * (
        temperature_k / 288.15
    ) ** 5.25588

    density = pressure_pa / (287.05 * temperature_k)

    temperature_c = temperature_k - 273.15
    pressure_kpa = pressure_pa / 1000.0

    return temperature_c, pressure_kpa, density


# =========================================================
# RUN THE SIMULATION
# =========================================================

if run_simulation:

    duration_seconds = duration_minutes * 60.0
    variation = sensitivity_percent / 100.0

    # -----------------------------------------------------
    # THERMAL SCENARIO COMPARISON
    # -----------------------------------------------------

    st.divider()
    st.header("2. Thermal Scenario Comparison")

    scenarios = [
        ("Colder scenario", ambient_temp - scenario_offset),
        ("Selected scenario", ambient_temp),
        ("Warmer scenario", ambient_temp + scenario_offset)
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    results = []

    for scenario_name, scenario_ambient in scenarios:

        time, temp, equilibrium = simulate(
            ambient=scenario_ambient,
            initial=initial_temp,
            power=heat_generated,
            duration_seconds=duration_seconds,
            capacity=thermal_capacity,
            transfer=heat_transfer,
            requested_dt=time_step
        )

        ax.plot(
            time / 60.0,
            temp,
            label=f"{scenario_name} ({scenario_ambient:.1f} °C)"
        )

        results.append({
            "Scenario": scenario_name,
            "Ambient temperature (°C)": round(scenario_ambient, 2),
            "Initial temperature (°C)": initial_temp,
            "Final temperature (°C)": round(float(temp[-1]), 2),
            "Equilibrium estimate (°C)": round(equilibrium, 2)
        })

    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Component temperature (°C)")
    ax.set_title("Thermal Response Under Different Ambient Temperatures")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    results_df = pd.DataFrame(results)

    st.subheader("Thermal results")
    st.dataframe(results_df, use_container_width=True)

    st.download_button(
        "Download thermal results (CSV)",
        data=results_df.to_csv(index=False).encode("utf-8"),
        file_name="thermal_results.csv",
        mime="text/csv"
    )

    # -----------------------------------------------------
    # SENSITIVITY ANALYSIS
    # -----------------------------------------------------

    st.divider()
    st.header("3. Sensitivity Analysis")

    _, baseline_temp, _ = simulate(
        ambient=ambient_temp,
        initial=initial_temp,
        power=heat_generated,
        duration_seconds=duration_seconds,
        capacity=thermal_capacity,
        transfer=heat_transfer,
        requested_dt=time_step
    )

    baseline_final = float(baseline_temp[-1])

    sensitivity_tests = [
        (
            "Ambient temperature (-)",
            ambient_temp - 5,
            heat_generated,
            thermal_capacity,
            heat_transfer
        ),
        (
            "Ambient temperature (+)",
            ambient_temp + 5,
            heat_generated,
            thermal_capacity,
            heat_transfer
        ),
        (
            "Heat generation (-)",
            ambient_temp,
            heat_generated * (1 - variation),
            thermal_capacity,
            heat_transfer
        ),
        (
            "Heat generation (+)",
            ambient_temp,
            heat_generated * (1 + variation),
            thermal_capacity,
            heat_transfer
        ),
        (
            "Thermal capacity (-)",
            ambient_temp,
            heat_generated,
            thermal_capacity * (1 - variation),
            heat_transfer
        ),
        (
            "Thermal capacity (+)",
            ambient_temp,
            heat_generated,
            thermal_capacity * (1 + variation),
            heat_transfer
        ),
        (
            "Heat transfer (-)",
            ambient_temp,
            heat_generated,
            thermal_capacity,
            heat_transfer * (1 - variation)
        ),
        (
            "Heat transfer (+)",
            ambient_temp,
            heat_generated,
            thermal_capacity,
            heat_transfer * (1 + variation)
        )
    ]

    sensitivity_rows = []

    for name, amb, power, capacity, transfer in sensitivity_tests:

        _, temp, _ = simulate(
            ambient=amb,
            initial=initial_temp,
            power=power,
            duration_seconds=duration_seconds,
            capacity=capacity,
            transfer=transfer,
            requested_dt=time_step
        )

        final_temp = float(temp[-1])

        sensitivity_rows.append({
            "Test": name,
            "Final temperature (°C)": round(final_temp, 2),
            "Change from baseline (°C)": round(
                final_temp - baseline_final, 2
            )
        })

    sensitivity_df = pd.DataFrame(sensitivity_rows)

    st.write(f"Baseline final temperature: {baseline_final:.2f} °C")
    st.dataframe(sensitivity_df, use_container_width=True)

    fig_sens, ax_sens = plt.subplots(figsize=(11, 5))

    ax_sens.bar(
        sensitivity_df["Test"],
        sensitivity_df["Change from baseline (°C)"]
    )

    ax_sens.axhline(0, linewidth=1)
    ax_sens.set_ylabel("Temperature change (°C)")
    ax_sens.set_title("One-Parameter-at-a-Time Sensitivity")
    ax_sens.tick_params(axis="x", rotation=35, labelsize=8)
    ax_sens.grid(axis="y", alpha=0.3)
    fig_sens.tight_layout()

    st.pyplot(fig_sens)
    plt.close(fig_sens)

    st.download_button(
        "Download sensitivity results (CSV)",
        data=sensitivity_df.to_csv(index=False).encode("utf-8"),
        file_name="sensitivity_results.csv",
        mime="text/csv"
    )

    # -----------------------------------------------------
    # ATMOSPHERIC MODEL
    # -----------------------------------------------------

    st.divider()
    st.header("4. Atmospheric Conditions")

    atm_temp, pressure_kpa, density = standard_atmosphere(altitude_m)
    sea_temp, sea_pressure, sea_density = standard_atmosphere(0)

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Standard air temperature",
        f"{atm_temp:.2f} °C"
    )

    col2.metric(
        "Atmospheric pressure",
        f"{pressure_kpa:.2f} kPa",
        delta=f"{(pressure_kpa / sea_pressure - 1) * 100:.1f}% vs sea level"
    )

    col3.metric(
        "Air density",
        f"{density:.4f} kg/m³",
        delta=f"{(density / sea_density - 1) * 100:.1f}% vs sea level"
    )

    altitudes = np.linspace(0, 11000, 100)
    pressures = [
        standard_atmosphere(h)[1] for h in altitudes
    ]
    densities = [
        standard_atmosphere(h)[2] for h in altitudes
    ]

    fig_atm, ax_atm = plt.subplots(figsize=(9, 4))
    ax_atm.plot(altitudes, pressures)
    ax_atm.set_xlabel("Altitude (m)")
    ax_atm.set_ylabel("Pressure (kPa)")
    ax_atm.set_title("Standard Atmospheric Pressure vs Altitude")
    ax_atm.grid(True, alpha=0.3)
    fig_atm.tight_layout()

    st.pyplot(fig_atm)
    plt.close(fig_atm)

    fig_density, ax_density = plt.subplots(figsize=(9, 4))
    ax_density.plot(altitudes, densities)
    ax_density.set_xlabel("Altitude (m)")
    ax_density.set_ylabel("Density (kg/m³)")
    ax_density.set_title("Standard Air Density vs Altitude")
    ax_density.grid(True, alpha=0.3)
    fig_density.tight_layout()

    st.pyplot(fig_density)
    plt.close(fig_density)

    # -----------------------------------------------------
    # COMBINED ENVIRONMENTAL SUMMARY
    # -----------------------------------------------------

    st.divider()
    st.header("5. Combined Environmental Summary")

    environment_df = pd.DataFrame([
        {
            "Parameter": "Altitude",
            "Value": altitude_m,
            "Unit": "m"
        },
        {
            "Parameter": "User-selected ambient temperature",
            "Value": ambient_temp,
            "Unit": "°C"
        },
        {
            "Parameter": "Standard-atmosphere temperature",
            "Value": round(atm_temp, 2),
            "Unit": "°C"
        },
        {
            "Parameter": "Atmospheric pressure",
            "Value": round(pressure_kpa, 3),
            "Unit": "kPa"
        },
        {
            "Parameter": "Air density",
            "Value": round(density, 4),
            "Unit": "kg/m³"
        },
        {
            "Parameter": "Thermal capacity",
            "Value": thermal_capacity,
            "Unit": "J/°C"
        },
        {
            "Parameter": "Heat-transfer coefficient",
            "Value": heat_transfer,
            "Unit": "W/°C"
        }
    ])

    st.dataframe(environment_df, use_container_width=True)

    st.download_button(
        "Download environmental summary (CSV)",
        data=environment_df.to_csv(index=False).encode("utf-8"),
        file_name="environmental_summary.csv",
        mime="text/csv"
    )

    # -----------------------------------------------------
    # ENVIRONMENTAL IMPACT ASSESSMENT
    # -----------------------------------------------------

    st.divider()
    st.header("6. Environmental Impact Assessment")

    assessment_df = pd.DataFrame([
        {
            "Factor": "Ambient temperature",
            "Coverage": "Modelled",
            "Limitation": "Uses a simplified thermal equation."
        },
        {
            "Factor": "Altitude",
            "Coverage": "Atmosphere estimated",
            "Limitation": "Does not directly predict equipment performance."
        },
        {
            "Factor": "Pressure",
            "Coverage": "Estimated",
            "Limitation": "Standard atmosphere, not local weather."
        },
        {
            "Factor": "Air density",
            "Coverage": "Estimated",
            "Limitation": "No validated component-impact model."
        },
        {
            "Factor": "Dust and snow",
            "Coverage": "Not modelled",
            "Limitation": "Requires a separate evidence-based model."
        },
        {
            "Factor": "Wind and vibration",
            "Coverage": "Not modelled",
            "Limitation": "No structural or vibration simulation."
        },
        {
            "Factor": "Component reliability",
            "Coverage": "Not validated",
            "Limitation": "No failure-rate prediction."
        }
    ])

    st.dataframe(assessment_df, use_container_width=True)

    st.download_button(
        "Download impact assessment (CSV)",
        data=assessment_df.to_csv(index=False).encode("utf-8"),
        file_name="impact_assessment.csv",
        mime="text/csv"
    )

    # -----------------------------------------------------
    # SCIENTIFIC REFERENCES
    # -----------------------------------------------------

    st.divider()
    st.header("7. Scientific References")

    st.markdown(
        """
        - [NASA Glenn: Earth Atmosphere Equation](https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/earth-atmosphere-equation-metric/)
        - [NASA: Passive Thermal Control Engineering Guidebook](https://ntrs.nasa.gov/citations/20230013900)
        - [U.S. DLA: MIL-STD-810 information](https://quicksearch.dla.mil/qaDocDetails.aspx?ident_number=35978)
        """
    )

    st.caption(
        "These references provide background. They do not validate "
        "this app's assumed thermal parameters."
    )

    # -----------------------------------------------------
    # VERIFICATION TESTS
    # -----------------------------------------------------

    st.divider()
    st.header("8. Mathematical Verification Tests")

    st.write(
        "These tests use controlled test inputs to check the "
        "calculation function. They are separate from the values "
        "entered for your simulation."
    )

    test_results = []

    # Test 1: equal starting and ambient temperatures, no heat
    _, temp1, _ = simulate(
        ambient=20,
        initial=20,
        power=0,
        duration_seconds=600,
        capacity=500,
        transfer=0.8,
        requested_dt=10
    )

    pass1 = np.allclose(temp1, 20.0)

    test_results.append({
        "Test": "Zero heat at ambient equilibrium",
        "Expected": "Temperature stays at 20°C",
        "Actual": f"{temp1[-1]:.2f}°C",
        "Status": "PASS" if pass1 else "FAIL"
    })

    # Test 2: zero heat; temperature should move toward ambient
    _, temp2, _ = simulate(
        ambient=-20,
        initial=20,
        power=0,
        duration_seconds=3600,
        capacity=500,
        transfer=0.8,
        requested_dt=10
    )

    pass2 = abs(temp2[-1] + 20) < abs(20 + 20)

    test_results.append({
        "Test": "Zero heat moves temperature toward ambient",
        "Expected": "Closer to -20°C",
        "Actual": f"{temp2[-1]:.2f}°C",
        "Status": "PASS" if pass2 else "FAIL"
    })

    # Test 3: equilibrium equation
    _, _, eq3 = simulate(
        ambient=-20,
        initial=-20,
        power=8,
        duration_seconds=3600,
        capacity=500,
        transfer=0.8,
        requested_dt=10
    )

    expected_eq = -20 + 8 / 0.8
    pass3 = np.isclose(eq3, expected_eq)

    test_results.append({
        "Test": "Equilibrium calculation",
        "Expected": f"{expected_eq:.2f}°C",
        "Actual": f"{eq3:.2f}°C",
        "Status": "PASS" if pass3 else "FAIL"
    })

    test_df = pd.DataFrame(test_results)
    st.dataframe(test_df, use_container_width=True)

    if all([pass1, pass2, pass3]):
        st.success("All three mathematical verification tests passed.")
    else:
        st.error("One or more mathematical verification tests failed.")

    st.caption(
        "Passing these tests checks selected mathematical behaviours. "
        "It does not establish real-world accuracy or equipment reliability."
    )

else:
    st.info(
        "Enter your parameters above and click Run Simulation "
        "to generate the graphs, tables, and reports."
    )

# =========================================================
# LIMITATIONS
# =========================================================

st.divider()
st.header("Model Limitations")

st.markdown(
    """
    - The thermal model represents a component as one simplified thermal mass.
    - Thermal capacity and heat transfer must be supplied by the user;
      their example values are not measured equipment data.
    - The standard-atmosphere model does not represent actual weather.
    - The thermal and atmospheric calculations are separate models.
    - Dust, snow, wind, vibration, sensor behaviour, and failure rates
      are not currently simulated.
    - Results are educational estimates, not validated engineering predictions.
    """
)

st.caption(
    "Educational prototype | User-configurable inputs | "
    "Not validated for operational use"
)
