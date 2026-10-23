
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------
# Page setup
# -----------------------------------

st.set_page_config(
    page_title="High-Altitude Thermal Simulator",
    page_icon="🌡️",
    layout="wide"
)

st.title("High-Altitude Environmental Performance Simulator")

st.write(
    "An educational simulation for studying how ambient "
    "temperature and thermal parameters affect a component."
)

st.warning(
    "This is an illustrative software model, not a validated "
    "prediction of real equipment performance."
)

# -----------------------------------
# Model function
# -----------------------------------

def simulate(
    ambient,
    initial,
    power,
    duration,
    capacity=500.0,
    transfer=0.8
):
    # Time step: 10 seconds
    dt = 10.0

    steps = max(1, int(duration / dt))
    time = np.linspace(0, duration, steps + 1)
    actual_dt = time[1] - time[0]

    temp = np.zeros(steps + 1)
    temp[0] = initial

    for i in range(1, len(time)):
        rate = (
            power - transfer * (temp[i - 1] - ambient)
        ) / capacity

        temp[i] = temp[i - 1] + rate * actual_dt

    equilibrium = ambient + power / transfer

    return time, temp, equilibrium


# -----------------------------------
# Sidebar: user inputs
# -----------------------------------

st.sidebar.header("Simulation settings")

ambient_temp = st.sidebar.slider(
    "Ambient temperature (°C)",
    min_value=-60,
    max_value=30,
    value=-20,
    step=5
)

initial_temp = st.sidebar.slider(
    "Initial component temperature (°C)",
    min_value=-40,
    max_value=60,
    value=20,
    step=5
)

heat_generated = st.sidebar.slider(
    "Illustrative heat generation (W)",
    min_value=0.0,
    max_value=20.0,
    value=8.0,
    step=1.0
)

duration_minutes = st.sidebar.slider(
    "Simulation duration (minutes)",
    min_value=10,
    max_value=240,
    value=60,
    step=10
)

st.sidebar.caption(
    "The parameters are illustrative and are not measured "
    "specifications for a real component."
)

# -----------------------------------
# Model assumptions
# -----------------------------------

with st.expander("Model assumptions and limitations"):
    st.markdown(
        """
        - The component is represented as a single thermal mass.
        - Ambient temperature remains constant during each run.
        - Heat generation is constant during each run.
        - Heat transfer is represented by a simplified linear model.
        - Thermal capacity and heat-transfer coefficient are assumed
          values, not measured equipment data.
        - The model does not represent airflow, altitude pressure,
          dust, snow, vibration, or sensor errors.
        - Results must not be interpreted as proof of real-world
          equipment reliability or safety.
        """
    )

st.latex(r"C\frac{dT}{dt}=P-K(T-T_a)")

st.write(
    "Here, C is thermal capacity, P is heat generation, "
    "K is the heat-transfer coefficient, T is component "
    "temperature, and Tₐ is ambient temperature."
)

# -----------------------------------
# Run simulations
# -----------------------------------

if st.button("Run comparison", type="primary"):

    duration_seconds = duration_minutes * 60

    # Compare three ambient temperatures
    scenarios = [
        ("Cold scenario", ambient_temp - 10),
        ("Selected scenario", ambient_temp),
        ("Warmer scenario", min(30, ambient_temp + 10))
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    results = []

    for scenario_name, scenario_ambient in scenarios:
        time, temp, equilibrium = simulate(
            ambient=scenario_ambient,
            initial=initial_temp,
            power=heat_generated,
            duration=duration_seconds
        )

        time_minutes = time / 60

        ax.plot(
            time_minutes,
            temp,
            label=f"{scenario_name}: {scenario_ambient}°C"
        )

        results.append({
            "Scenario": scenario_name,
            "Ambient temperature (°C)": scenario_ambient,
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

    st.subheader("Temperature comparison")
    st.pyplot(fig)
    plt.close(fig)

    results_df = pd.DataFrame(results)

    st.subheader("Simulation results")
    st.dataframe(results_df, use_container_width=True)

    st.download_button(
        label="Download comparison results as CSV",
        data=results_df.to_csv(index=False).encode("utf-8"),
        file_name="thermal_simulation_results.csv",
        mime="text/csv"
    )

    # -----------------------------------
    # Sensitivity analysis
    # -----------------------------------

    st.subheader("Sensitivity Analysis")

    st.write(
        "This section changes one model parameter at a time "
        "to show how it affects the final simulated temperature."
    )

    baseline_ambient = float(ambient_temp)
    baseline_power = float(heat_generated)
    baseline_capacity = 500.0
    baseline_transfer = 0.8

    _, baseline_temp, _ = simulate(
        ambient=baseline_ambient,
        initial=initial_temp,
        power=baseline_power,
        duration=duration_seconds,
        capacity=baseline_capacity,
        transfer=baseline_transfer
    )

    baseline_final = float(baseline_temp[-1])

    tests = [
        (
            "Ambient temperature",
            baseline_ambient - 5,
            baseline_power,
            baseline_capacity,
            baseline_transfer,
            baseline_ambient - 5
        ),
        (
            "Ambient temperature",
            baseline_ambient + 5,
            baseline_power,
            baseline_capacity,
            baseline_transfer,
            baseline_ambient + 5
        ),
        (
            "Heat generation",
            baseline_ambient,
            baseline_power * 0.8,
            baseline_capacity,
            baseline_transfer,
            baseline_power * 0.8
        ),
        (
            "Heat generation",
            baseline_ambient,
            baseline_power * 1.2,
            baseline_capacity,
            baseline_transfer,
            baseline_power * 1.2
        ),
        (
            "Thermal capacity",
            baseline_ambient,
            baseline_power,
            baseline_capacity * 0.8,
            baseline_transfer,
            baseline_capacity * 0.8
        ),
        (
            "Thermal capacity",
            baseline_ambient,
            baseline_power,
            baseline_capacity * 1.2,
            baseline_transfer,
            baseline_capacity * 1.2
        ),
        (
            "Heat transfer",
            baseline_ambient,
            baseline_power,
            baseline_capacity,
            baseline_transfer * 0.8,
            baseline_transfer * 0.8
        ),
        (
            "Heat transfer",
            baseline_ambient,
            baseline_power,
            baseline_capacity,
            baseline_transfer * 1.2,
            baseline_transfer * 1.2
        )
    ]

    sensitivity_rows = []

    for name, ambient, power, capacity, transfer, tested_value in tests:
        _, temp, _ = simulate(
            ambient=ambient,
            initial=initial_temp,
            power=power,
            duration=duration_seconds,
            capacity=capacity,
            transfer=transfer
        )

        final_temp = float(temp[-1])

        sensitivity_rows.append({
            "Parameter": name,
            "Tested value": round(tested_value, 3),
            "Final temperature (°C)": round(final_temp, 2),
            "Change from baseline (°C)": round(
                final_temp - baseline_final, 2
            )
        })

    sensitivity_df = pd.DataFrame(sensitivity_rows)

    st.write(
        f"Baseline final temperature: {baseline_final:.2f} °C"
    )

    st.dataframe(
        sensitivity_df,
        use_container_width=True
    )

    fig_sens, ax_sens = plt.subplots(figsize=(11, 5))

    labels = [
        f"{row['Parameter']}\n{row['Tested value']}"
        for row in sensitivity_rows
    ]

    changes = [
        row["Change from baseline (°C)"]
        for row in sensitivity_rows
    ]

    ax_sens.bar(labels, changes)
    ax_sens.axhline(0, linewidth=1)
    ax_sens.set_ylabel("Change in final temperature (°C)")
    ax_sens.set_title("One-Parameter-at-a-Time Sensitivity Analysis")
    ax_sens.tick_params(axis="x", labelsize=8)
    ax_sens.grid(axis="y", alpha=0.3)
    fig_sens.tight_layout()

    st.pyplot(fig_sens)
    plt.close(fig_sens)

    st.caption(
        "Sensitivity results depend on the assumed parameters and "
        "simulation duration. They are not validated operating "
        "conditions or measurements of real equipment."
    )

else:
    st.info(
        "Adjust the settings in the sidebar, then click "
        "'Run comparison' to view the simulation and sensitivity analysis."
    )

# -----------------------------------
# Project footer
# -----------------------------------

st.divider()

st.caption(
    "Educational prototype | Simplified thermal model | "
    "Not validated for real-world equipment"
)

# -----------------------------------
# Reference and validation section
# -----------------------------------

st.divider()

st.header("Scientific References and Model Verification")

st.subheader("Scientific references")

st.markdown(
    """
    **1. NASA Glenn Research Center — Earth Atmosphere Equation**

    Explains how atmospheric temperature, pressure, and density
    vary with altitude.

    [Open NASA atmosphere reference](https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/earth-atmosphere-equation-metric/)

    **2. NASA — Passive Thermal Control Engineering Guidebook**

    Provides engineering guidance on thermal analysis, thermal
    control, modelling, and testing.

    [Open NASA thermal engineering guide](https://ntrs.nasa.gov/citations/20230013900)

    **3. U.S. Department of Defense — MIL-STD-810**

    Describes an approach to considering environmental stresses
    and tailoring tests to system requirements.

    [Open official standard information](https://quicksearch.dla.mil/qaDocDetails.aspx?ident_number=35978)
    """
)

st.caption(
    "These sources provide background and engineering guidance. "
    "They do not certify this educational simulator or its parameters."
)

st.subheader("Automatic model verification")

st.write(
    "Run these checks to verify selected mathematical behaviours "
    "of the simulation function."
)

if st.button("Run verification tests"):

    test_results = []

    # Test 1: No temperature change when there is no heat flow
    test_time, test_temp, test_eq = simulate(
        ambient=20,
        initial=20,
        power=0,
        duration=600,
        capacity=500,
        transfer=0.8
    )

    passed_1 = np.allclose(test_temp, 20.0)

    test_results.append({
        "Test": "1. Equal initial and ambient temperature, zero heat",
        "Expected": "Temperature remains 20°C",
        "Actual": f"{test_temp[-1]:.2f}°C",
        "Status": "PASS" if passed_1 else "FAIL"
    })

    # Test 2: With zero heat generation, temperature approaches ambient
    test_time, test_temp, test_eq = simulate(
        ambient=-20,
        initial=20,
        power=0,
        duration=3600,
        capacity=500,
        transfer=0.8
    )

    passed_2 = (
        abs(test_temp[-1] - (-20)) <
        abs(20 - (-20))
    )

    test_results.append({
        "Test": "2. Zero heat generation",
        "Expected": "Temperature moves toward -20°C",
        "Actual": f"{test_temp[-1]:.2f}°C",
        "Status": "PASS" if passed_2 else "FAIL"
    })

    # Test 3: Check the equilibrium equation
    test_time, test_temp, test_eq = simulate(
        ambient=-20,
        initial=-20,
        power=8,
        duration=3600,
        capacity=500,
        transfer=0.8
    )

    expected_eq = -20 + (8 / 0.8)
    passed_3 = np.isclose(test_eq, expected_eq)

    test_results.append({
        "Test": "3. Equilibrium calculation",
        "Expected": f"{expected_eq:.2f}°C",
        "Actual": f"{test_eq:.2f}°C",
        "Status": "PASS" if passed_3 else "FAIL"
    })

    test_df = pd.DataFrame(test_results)

    st.dataframe(test_df, use_container_width=True)

    all_passed = all([
        passed_1,
        passed_2,
        passed_3
    ])

    if all_passed:
        st.success(
            "All three mathematical verification tests passed."
        )
    else:
        st.error(
            "One or more tests failed. Review the simulation function."
        )

    st.info(
        "Passing these tests checks selected code behaviours only. "
        "It does not establish real-world accuracy, equipment reliability, "
        "or suitability for operational use."
    )

st.subheader("Current limitations")

st.markdown(
    """
    - The thermal parameters are illustrative assumptions.
    - No measured component data has been used for calibration.
    - The model does not simulate pressure, airflow, snow, dust,
      vibration, or sensor performance.
    - Further validation would require suitable experimental data
      and review by qualified engineers.
    """
)



# -----------------------------------
# Altitude and Atmospheric Conditions
# -----------------------------------

st.divider()

st.header("Altitude and Atmospheric Conditions")

st.write(
    "Explore how standard atmospheric temperature, pressure, "
    "and air density vary with altitude."
)

st.caption(
    "This is a standard-atmosphere estimate, not a live weather "
    "forecast or a measurement at a particular location."
)

altitude_m = st.slider(
    "Select altitude above sea level (metres)",
    min_value=0,
    max_value=11000,
    value=3000,
    step=250
)

def standard_atmosphere(altitude):
    """
    Approximate standard atmosphere from 0 to 11,000 metres.
    Returns temperature in °C, pressure in kPa,
    and density in kg/m³.
    """
    temperature_c = 15.04 - 0.00649 * altitude

    pressure_kpa = 101.29 * (
        (temperature_c + 273.1) / 288.08
    ) ** 5.256

    density = pressure_kpa / (
        0.2869 * (temperature_c + 273.1)
    )

    return temperature_c, pressure_kpa, density


temperature_c, pressure_kpa, density = standard_atmosphere(
    altitude_m
)

sea_temp, sea_pressure, sea_density = standard_atmosphere(0)

pressure_change = (
    (pressure_kpa / sea_pressure) - 1
) * 100

density_change = (
    (density / sea_density) - 1
) * 100

col1, col2, col3 = st.columns(3)

col1.metric(
    "Standard air temperature",
    f"{temperature_c:.1f} °C"
)

col2.metric(
    "Atmospheric pressure",
    f"{pressure_kpa:.2f} kPa",
    delta=f"{pressure_change:.1f}% vs sea level"
)

col3.metric(
    "Air density",
    f"{density:.3f} kg/m³",
    delta=f"{density_change:.1f}% vs sea level"
)

# Plot standard atmospheric conditions over altitude

altitudes = np.linspace(0, 11000, 100)

temperatures = []
pressures = []
densities = []

for h in altitudes:
    t, p, rho = standard_atmosphere(h)
    temperatures.append(t)
    pressures.append(p)
    densities.append(rho)

fig_atm, ax_atm = plt.subplots(figsize=(9, 5))

ax_atm.plot(
    altitudes,
    pressures,
    label="Pressure (kPa)"
)

ax_atm.set_xlabel("Altitude (m)")
ax_atm.set_ylabel("Pressure (kPa)")
ax_atm.set_title("Standard Atmospheric Pressure vs Altitude")
ax_atm.grid(True, alpha=0.3)
ax_atm.legend()

st.pyplot(fig_atm)
plt.close(fig_atm)

fig_density, ax_density = plt.subplots(figsize=(9, 5))

ax_density.plot(
    altitudes,
    densities,
    label="Air density (kg/m³)"
)

ax_density.set_xlabel("Altitude (m)")
ax_density.set_ylabel("Air density (kg/m³)")
ax_density.set_title("Standard Air Density vs Altitude")
ax_density.grid(True, alpha=0.3)
ax_density.legend()

st.pyplot(fig_density)
plt.close(fig_density)

st.info(
    "Model boundary: these equations are intended here for "
    "altitudes from 0 to 11,000 metres. They assume a standard "
    "atmosphere and do not account for actual weather conditions."
)



# -----------------------------------
# Combined Environmental Summary
# -----------------------------------

st.divider()

st.header("Combined Environmental Summary")

st.write(
    "Review the selected altitude alongside the ambient "
    "temperature and the standard-atmosphere estimates."
)

# Use the altitude selected in the previous section.
summary_temp, summary_pressure, summary_density = (
    standard_atmosphere(altitude_m)
)

summary_col1, summary_col2 = st.columns(2)

with summary_col1:
    st.metric(
        "Selected altitude",
        f"{altitude_m:,} m"
    )

    st.metric(
        "Selected ambient temperature",
        f"{ambient_temp} °C"
    )

with summary_col2:
    st.metric(
        "Standard atmospheric pressure",
        f"{summary_pressure:.2f} kPa"
    )

    st.metric(
        "Standard air density",
        f"{summary_density:.3f} kg/m³"
    )

st.caption(
    "The selected ambient temperature is an independent user input. "
    "It may differ from the standard-atmosphere temperature shown "
    "by the altitude calculator."
)

environment_summary = pd.DataFrame([
    {
        "Environmental parameter": "Altitude",
        "Value": altitude_m,
        "Unit": "m"
    },
    {
        "Environmental parameter": "User-selected ambient temperature",
        "Value": ambient_temp,
        "Unit": "°C"
    },
    {
        "Environmental parameter": "Standard-atmosphere temperature",
        "Value": round(summary_temp, 2),
        "Unit": "°C"
    },
    {
        "Environmental parameter": "Standard atmospheric pressure",
        "Value": round(summary_pressure, 2),
        "Unit": "kPa"
    },
    {
        "Environmental parameter": "Standard air density",
        "Value": round(summary_density, 4),
        "Unit": "kg/m³"
    }
])

st.subheader("Environmental conditions table")

st.dataframe(
    environment_summary,
    use_container_width=True,
    hide_index=True
)

st.download_button(
    "Download environmental summary as CSV",
    data=environment_summary.to_csv(index=False).encode("utf-8"),
    file_name="environmental_summary.csv",
    mime="text/csv"
)



# -----------------------------------
# Environmental Impact Assessment
# -----------------------------------

st.divider()

st.header("Environmental Impact Assessment")

st.write(
    "This section summarizes the environmental variables "
    "represented by the current prototype and its limitations."
)

assessment_rows = [
    {
        "Environmental factor": "Ambient temperature",
        "Current coverage": "Modelled",
        "What the app does": (
            "Uses the selected temperature in the thermal simulation."
        ),
        "Next step": (
            "Compare assumptions with suitable component data."
        )
    },
    {
        "Environmental factor": "Altitude",
        "Current coverage": "Partially modelled",
        "What the app does": (
            "Estimates standard-atmosphere conditions from altitude."
        ),
        "Next step": (
            "Keep atmospheric estimates separate from component effects "
            "until a supported relationship is established."
        )
    },
    {
        "Environmental factor": "Air pressure",
        "Current coverage": "Estimated",
        "What the app does": (
            "Calculates standard atmospheric pressure."
        ),
        "Next step": (
            "Identify which component behaviours require pressure data."
        )
    },
    {
        "Environmental factor": "Air density",
        "Current coverage": "Estimated",
        "What the app does": (
            "Calculates standard-atmosphere air density."
        ),
        "Next step": (
            "Use documented engineering relationships before modelling "
            "any performance impact."
        )
    },
    {
        "Environmental factor": "Dust and snow",
        "Current coverage": "Not modelled",
        "What the app does": (
            "Does not simulate contamination or accumulation."
        ),
        "Next step": (
            "Document the limitation and research suitable test methods."
        )
    },
    {
        "Environmental factor": "Wind and vibration",
        "Current coverage": "Not modelled",
        "What the app does": (
            "Does not calculate wind loads or vibration response."
        ),
        "Next step": (
            "Define a separate, evidence-based model if time permits."
        )
    },
    {
        "Environmental factor": "Component reliability",
        "Current coverage": "Not validated",
        "What the app does": (
            "Does not predict actual failure rates or certify reliability."
        ),
        "Next step": (
            "Find appropriate published data before making predictions."
        )
    }
]

assessment_df = pd.DataFrame(assessment_rows)

st.dataframe(
    assessment_df,
    use_container_width=True,
    hide_index=True
)

st.subheader("Assessment summary")

modelled_count = sum(
    row["Current coverage"] in ["Modelled", "Estimated", "Partially modelled"]
    for row in assessment_rows
)

not_modelled_count = len(assessment_rows) - modelled_count

col_a, col_b = st.columns(2)

col_a.metric(
    "Factors represented or estimated",
    modelled_count
)

col_b.metric(
    "Factors not yet represented or validated",
    not_modelled_count
)

st.info(
    "These counts describe feature coverage only. They are not "
    "scores of equipment performance, reliability, or safety."
)

st.download_button(
    "Download environmental impact assessment as CSV",
    data=assessment_df.to_csv(index=False).encode("utf-8"),
    file_name="environmental_impact_assessment.csv",
    mime="text/csv"
)



# -----------------------------------
# Software Test Plan and Results
# -----------------------------------

st.divider()

st.header("Software Test Plan and Results")

st.write(
    "Use this checklist to record manual tests of the simulator. "
    "Mark a test as passed only after you have performed it."
)

test_cases = [
    {
        "Test ID": "T01",
        "Test": "App opens successfully",
        "Procedure": "Open the Streamlit app in the browser.",
        "Expected result": "The app loads without an error."
    },
    {
        "Test ID": "T02",
        "Test": "Thermal simulation",
        "Procedure": "Run the thermal comparison.",
        "Expected result": "The temperature graph and results table appear."
    },
    {
        "Test ID": "T03",
        "Test": "Sensitivity analysis",
        "Procedure": "Run the comparison and inspect sensitivity results.",
        "Expected result": "The sensitivity table and chart appear."
    },
    {
        "Test ID": "T04",
        "Test": "Verification tests",
        "Procedure": "Click Run verification tests.",
        "Expected result": "All three mathematical checks pass."
    },
    {
        "Test ID": "T05",
        "Test": "Altitude model",
        "Procedure": "Change the altitude slider.",
        "Expected result": "Pressure and density estimates update."
    },
    {
        "Test ID": "T06",
        "Test": "Environmental summary",
        "Procedure": "Inspect the combined summary and download its CSV.",
        "Expected result": "The table appears and the CSV downloads."
    },
    {
        "Test ID": "T07",
        "Test": "Impact assessment",
        "Procedure": "Inspect the assessment table and download its CSV.",
        "Expected result": "Coverage and limitations are displayed."
    }
]

test_df = pd.DataFrame(test_cases)

st.dataframe(
    test_df,
    use_container_width=True,
    hide_index=True
)

st.subheader("Record your test results")

test_statuses = {}

for test in test_cases:
    test_statuses[test["Test ID"]] = st.selectbox(
        f'{test["Test ID"]}: {test["Test"]}',
        options=["Not tested", "Pass", "Fail"],
        index=0,
        key=f'test_status_{test["Test ID"]}'
    )

test_notes = st.text_area(
    "Notes, errors, or observations",
    placeholder="Record what happened when you ran the tests."
)

if st.button("Prepare test report"):
    report_rows = []

    for test in test_cases:
        report_rows.append({
            "Test ID": test["Test ID"],
            "Test": test["Test"],
            "Expected result": test["Expected result"],
            "Recorded status": test_statuses[test["Test ID"]],
            "Notes": test_notes
        })

    report_df = pd.DataFrame(report_rows)

    st.subheader("Test report preview")
    st.dataframe(
        report_df,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download test report as CSV",
        data=report_df.to_csv(index=False).encode("utf-8"),
        file_name="software_test_report.csv",
        mime="text/csv"
    )

    passed = sum(
        status == "Pass"
        for status in test_statuses.values()
    )

    failed = sum(
        status == "Fail"
        for status in test_statuses.values()
    )

    st.write(
        f"Recorded: {passed} passed, {failed} failed, "
        f"{len(test_cases) - passed - failed} not tested."
    )

    st.caption(
        "This report records the statuses you selected. "
        "It does not automatically verify the tests or prove "
        "real-world equipment performance."
    )
