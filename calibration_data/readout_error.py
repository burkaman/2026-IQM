import json

with open("calibration_data/emerald.json") as f:
    data = json.load(f)

    cz_observations = [
        obs
        for obs in data["observations"]
        if obs["dut_field"].startswith("metrics.ssro.measure_fidelity")
        and obs["dut_field"].endswith("fidelity")
    ]

    qubits = []
    for obs in cz_observations:
        qubit = obs["dut_field"].split(".")[4].split(".")[0]
        qubits.append((int(qubit[2:]) - 1, 100 * (1 - obs["value"])))
    print(qubits)
