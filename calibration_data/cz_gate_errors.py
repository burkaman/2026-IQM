import json

with open("calibration_data/garnet.json") as f:
    data = json.load(f)

    cz_observations = [
        obs
        for obs in data["observations"]
        if obs["dut_field"].startswith("metrics.irb.cz")
    ]

    pairs = []
    for obs in cz_observations:
        qubit_pair = obs["dut_field"].split(".")[4].split(".")[0].split("__")
        pairs.append(
            (
                [int(qubit_pair[0][2:]) - 1, int(qubit_pair[1][2:]) - 1],
                100 * (1 - obs["value"]),
            )
        )
    print(pairs)
