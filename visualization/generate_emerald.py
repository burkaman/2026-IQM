#!/usr/bin/env python3
"""
Generate emerald.svg with error-based coloring for nodes and edges.

The color scale goes from #7ad151 (minimum error) to #482475 (maximum error).
"""

import colorsys
import os
from pathlib import Path
from typing import Optional, Sequence, Tuple


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """Convert RGB tuple to hex color."""
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def interpolate_color(
    value: float,
    min_val: float,
    max_val: float,
    min_color: str = "#7ad151",
    max_color: str = "#482475",
) -> str:
    """
    Interpolate color using viridis-like color scheme.

    Args:
        value: The error percentage value
        min_val: Minimum error percentage in dataset
        max_val: Maximum error percentage in dataset
        min_color: Color for minimum error (kept for API compatibility but not used)
        max_color: Color for maximum error (kept for API compatibility but not used)

    Returns:
        Hex color string
    """
    if max_val == min_val:
        return min_color

    # Normalize value to 0-1 range
    t = (value - min_val) / (max_val - min_val)

    # Viridis color scheme REVERSED: yellow (low) -> green -> teal -> purple (high)
    # Key colors from the viridis palette (reversed)
    viridis_colors = [
        (0.477504, 0.821444, 0.318195),  # Yellow-green
        (0.266941, 0.748751, 0.440573),  # Green
        (0.134692, 0.658636, 0.517649),  # Green-teal
        (0.127568, 0.566949, 0.550556),  # Teal
        (0.163625, 0.471133, 0.558148),  # Teal-blue
        (0.206756, 0.371758, 0.553117),  # Blue
        (0.253935, 0.265254, 0.529983),  # Blue-purple
        (0.282623, 0.140926, 0.457517),  # Purple
        (0.267004, 0.004874, 0.329415),  # Dark purple (high values)
    ]

    # Find the two colors to interpolate between
    n_colors = len(viridis_colors)
    scaled_t = t * (n_colors - 1)
    idx = int(scaled_t)

    if idx >= n_colors - 1:
        # At or beyond the maximum
        r, g, b = viridis_colors[-1]
    else:
        # Interpolate between two adjacent colors
        local_t = scaled_t - idx
        c1 = viridis_colors[idx]
        c2 = viridis_colors[idx + 1]

        r = c1[0] + local_t * (c2[0] - c1[0])
        g = c1[1] + local_t * (c2[1] - c1[1])
        b = c1[2] + local_t * (c2[2] - c1[2])

    # Convert from 0-1 range to 0-255 range
    return rgb_to_hex((int(r * 255), int(g * 255), int(b * 255)))


# Qubit positions (0-indexed node_num -> (x, y, label))
# QB1 = 0, QB2 = 1, etc.
QUBIT_POSITIONS = {
    0: (718.6600000000001, 668.6600000000001, "QB1"),
    1: (785.9200000000001, 601.495, "QB2"),
    2: (517.1650000000001, 735.9200000000001, "QB3"),
    3: (584.33, 668.6600000000001, "QB4"),
    4: (651.495, 601.495, "QB5"),
    5: (718.6600000000001, 534.33, "QB6"),
    6: (785.9200000000001, 467.1650000000001, "QB7"),
    7: (382.83500000000004, 735.9200000000001, "QB8"),
    8: (450, 668.6600000000001, "QB9"),
    9: (517.1650000000001, 601.495, "QB10"),
    10: (584.33, 534.33, "QB11"),
    11: (651.495, 467.1650000000001, "QB12"),
    12: (718.6600000000001, 400, "QB13"),
    13: (785.9200000000001, 332.83500000000004, "QB14"),
    14: (248.505, 735.9200000000001, "QB15"),
    15: (315.67, 668.6600000000001, "QB16"),
    16: (382.83500000000004, 601.495, "QB17"),
    17: (450, 534.33, "QB18"),
    18: (517.1650000000001, 467.1650000000001, "QB19"),
    19: (584.33, 400, "QB20"),
    20: (651.495, 332.83500000000004, "QB21"),
    21: (718.6600000000001, 265.67, "QB22"),
    22: (181.34000000000003, 668.6600000000001, "QB23"),
    23: (248.505, 601.495, "QB24"),
    24: (315.67, 534.33, "QB25"),
    25: (382.83500000000004, 467.1650000000001, "QB26"),
    26: (450, 400, "QB27"),
    27: (517.1650000000001, 332.83500000000004, "QB28"),
    28: (584.33, 265.67, "QB29"),
    29: (651.495, 198.505, "QB30"),
    30: (718.6600000000001, 131.34, "QB31"),
    31: (181.34000000000003, 534.33, "QB32"),
    32: (248.505, 467.1650000000001, "QB33"),
    33: (315.67, 400, "QB34"),
    34: (382.83500000000004, 332.83500000000004, "QB35"),
    35: (450, 265.67, "QB36"),
    36: (517.1650000000001, 198.505, "QB37"),
    37: (584.33, 131.34, "QB38"),
    38: (651.495, 64.07999999999998, "QB39"),
    39: (114.07999999999998, 467.1650000000001, "QB40"),
    40: (181.34000000000003, 400, "QB41"),
    41: (248.505, 332.83500000000004, "QB42"),
    42: (315.67, 265.67, "QB43"),
    43: (382.83500000000004, 198.505, "QB44"),
    44: (450, 131.34, "QB45"),
    45: (517.1650000000001, 64.07999999999998, "QB46"),
    46: (114.07999999999998, 332.83500000000004, "QB47"),
    47: (181.34000000000003, 265.67, "QB48"),
    48: (248.505, 198.505, "QB49"),
    49: (315.67, 131.34, "QB50"),
    50: (382.83500000000004, 64.07999999999998, "QB51"),
    51: (114.07999999999998, 198.505, "QB52"),
    52: (181.34000000000003, 131.34, "QB53"),
    53: (248.505, 64.07999999999998, "QB54"),
}

# Coupler (edge) definitions: (start_node, end_node) -> (x, y, rotation)
# These are the connections between qubits
# Extracted from emerald_old.svg
COUPLER_CONNECTIONS = {
    (0, 1): (752.2900000000001, 635.125, 225.0),
    (0, 6): (752.2900000000001, 567.96, 315.0),
    (0, 7): (550.795, 702.29, 225.0),
    (0, 13): (752.2900000000001, 500.79499999999996, 225.0),
    (0, 14): (483.6300000000001, 702.29, 315.0),
    (0, 39): (416.37, 567.96, 225.0),
    (0, 45): (617.96, 366.37000000000006, 225.0),
    (0, 46): (416.37, 500.79499999999996, 315.0),
    (0, 50): (550.795, 366.37000000000006, 315.0),
    (0, 51): (416.37, 433.63000000000005, 225.0),
    (1, 3): (685.1250000000001, 635.125, 315.0),
    (1, 8): (617.96, 635.125, 225.0),
    (1, 15): (550.795, 635.125, 315.0),
    (1, 22): (483.6300000000001, 635.125, 225.0),
    (1, 30): (752.2900000000001, 366.37000000000006, 225.0),
    (1, 37): (685.1250000000001, 366.37000000000006, 315.0),
    (1, 52): (483.6300000000001, 366.37000000000006, 225.0),
    (2, 12): (617.96, 567.96, 315.0),
    (2, 15): (416.37, 702.29, 225.0),
    (2, 19): (550.795, 567.96, 225.0),
    (2, 21): (617.96, 500.79499999999996, 225.0),
    (2, 24): (416.37, 635.125, 315.0),
    (2, 28): (550.795, 500.79499999999996, 315.0),
    (2, 30): (617.96, 433.63000000000005, 315.0),
    (2, 37): (550.795, 433.63000000000005, 225.0),
    (3, 6): (685.1250000000001, 567.96, 225.0),
    (3, 13): (685.1250000000001, 500.79499999999996, 315.0),
    (3, 39): (349.20500000000004, 567.96, 315.0),
    (3, 46): (349.20500000000004, 500.79499999999996, 225.0),
    (5, 38): (685.1250000000001, 299.20500000000004, 225.0),
    (5, 45): (617.96, 299.20500000000004, 315.0),
    (5, 50): (550.795, 299.20500000000004, 225.0),
    (5, 51): (416.37, 366.37000000000006, 315.0),
    (6, 19): (685.1250000000001, 433.63000000000005, 225.0),
    (6, 30): (752.2900000000001, 299.20500000000004, 315.0),
    (6, 31): (483.6300000000001, 500.79499999999996, 225.0),
    (6, 40): (483.6300000000001, 433.63000000000005, 315.0),
    (6, 52): (483.6300000000001, 299.20500000000004, 315.0),
    (8, 14): (349.20500000000004, 702.29, 315.0),
    (8, 39): (282.04, 567.96, 225.0),
    (8, 46): (282.04, 500.79499999999996, 315.0),
    (8, 53): (349.20500000000004, 366.37000000000006, 225.0),
    (12, 38): (685.1250000000001, 232.04000000000002, 315.0),
    (12, 45): (617.96, 232.04000000000002, 225.0),
    (12, 50): (550.795, 232.04000000000002, 315.0),
    (12, 51): (416.37, 299.20500000000004, 225.0),
    (13, 52): (483.6300000000001, 232.04000000000002, 225.0),
    (14, 15): (282.04, 702.29, 225.0),
    (14, 17): (349.20500000000004, 635.125, 225.0),
    (14, 22): (214.875, 702.29, 315.0),
    (14, 24): (282.04, 635.125, 315.0),
    (14, 31): (214.875, 635.125, 225.0),
    (14, 44): (349.20500000000004, 433.63000000000005, 315.0),
    (14, 49): (282.04, 433.63000000000005, 225.0),
    (14, 52): (214.875, 433.63000000000005, 315.0),
    (15, 39): (214.875, 567.96, 315.0),
    (15, 46): (214.875, 500.79499999999996, 225.0),
    (15, 53): (282.04, 366.37000000000006, 315.0),
    (17, 53): (349.20500000000004, 299.20500000000004, 315.0),
    (21, 38): (685.1250000000001, 164.875, 225.0),
    (21, 50): (550.795, 164.875, 225.0),
    (21, 51): (416.37, 232.04000000000002, 315.0),
    (21, 53): (483.6300000000001, 164.875, 315.0),
    (22, 51): (147.70999999999998, 433.63000000000005, 225.0),
    (24, 51): (214.875, 366.37000000000006, 225.0),
    (24, 53): (282.04, 299.20500000000004, 225.0),
    (28, 53): (416.37, 164.875, 225.0),
    (30, 38): (685.1250000000001, 97.71, 315.0),
    (30, 45): (617.96, 97.71, 225.0),
    (30, 53): (483.6300000000001, 97.71, 225.0),
    (31, 51): (147.70999999999998, 366.37000000000006, 315.0),
    (33, 51): (214.875, 299.20500000000004, 315.0),
    (35, 53): (349.20500000000004, 164.875, 315.0),
    (37, 53): (416.37, 97.71, 315.0),
    (40, 51): (147.70999999999998, 299.20500000000004, 225.0),
    (42, 51): (214.875, 232.04000000000002, 225.0),
    (44, 53): (349.20500000000004, 97.71, 225.0),
    (47, 51): (147.70999999999998, 232.04000000000002, 315.0),
    (47, 53): (214.875, 164.875, 315.0),
    (49, 53): (282.04, 97.71, 315.0),
    (51, 52): (147.70999999999998, 164.875, 225.0),
    (52, 53): (214.875, 97.71, 225.0),
}


def generate_emerald_svg(
    node_errors: Sequence[Tuple[int, float]],
    edge_errors: Sequence[Tuple[Sequence[int], float]],
    output_path: str = "emerald.svg",
    path: Optional[Sequence[Sequence[int]]] = None,
) -> None:
    """
    Generate emerald.svg with error-based coloring and optional path animation.

    Args:
        node_errors: List of (node_num, error_pct) tuples (0-indexed)
        edge_errors: List of ([start_node, end_node], error_pct) tuples
        output_path: Output file path (relative paths will be resolved relative to this script)
        path: Optional list of edges [[start, end], ...] to animate as a path
    """
    # Resolve output path relative to the script's directory
    script_dir = Path(__file__).parent
    resolved_path = script_dir / output_path
    # Create dictionaries for quick lookup
    node_error_dict = {node: error for node, error in node_errors}
    edge_error_dict = {tuple(sorted(edge)): error for edge, error in edge_errors}

    # Find min and max values for color scaling
    all_errors = [e for _, e in node_errors] + [e for _, e in edge_errors]
    if not all_errors:
        min_error = 0
        max_error = 1
    else:
        min_error = min(all_errors)
        max_error = max(all_errors)

    # Build path animation data if path is provided
    path_nodes = set()
    path_edges_map = {}  # edge -> step index
    node_step_map = {}  # node -> step index

    if path:
        # Extract unique nodes from path
        for i, (start, end) in enumerate(path):
            edge_key = tuple(sorted([start, end]))
            path_edges_map[edge_key] = i

            # First occurrence of this node
            if start not in node_step_map:
                node_step_map[start] = len(node_step_map)
            if end not in node_step_map:
                node_step_map[end] = len(node_step_map)

            path_nodes.add(start)
            path_nodes.add(end)

    # Start building SVG
    svg_parts = []
    svg_parts.append(
        '<svg xmlns="http://www.w3.org/2000/svg" style="font-family: sans-serif; '
        'width: 100%; height: auto;" viewBox="0 0 1060 942.2166748046875"\n'
        '    preserveAspectRatio="xMidYmid none">\n'
    )

    # Add animation styles if path is provided
    if path:
        num_steps = len(node_step_map)
        total_animation_time = num_steps * 0.5  # 0.5s per step
        hold_time = 3.0  # Hold for 3 seconds
        fade_time = 0.6  # Fade out over 0.6s
        pause_time = 0.6  # Pause before loop
        total_cycle = total_animation_time + hold_time + fade_time + pause_time

        svg_parts.append("    <defs>\n        <style>\n")
        svg_parts.append("            /* Path tracing animation */\n")
        svg_parts.append("            @keyframes highlight-pulse {\n")
        svg_parts.append("                0% {\n")
        svg_parts.append(
            "                    filter: drop-shadow(0 0 8px gold) drop-shadow(0 0 16px gold);\n"
        )
        svg_parts.append("                }\n")
        svg_parts.append("                50% {\n")
        svg_parts.append(
            "                    filter: drop-shadow(0 0 12px gold) drop-shadow(0 0 24px gold);\n"
        )
        svg_parts.append("                }\n")
        svg_parts.append("                100% {\n")
        svg_parts.append(
            "                    filter: drop-shadow(0 0 6px gold) drop-shadow(0 0 12px gold);\n"
        )
        svg_parts.append("                }\n")
        svg_parts.append("            }\n\n")

        svg_parts.append(
            f"            /* Looping animation: total cycle is {total_cycle}s */\n"
        )
        svg_parts.append("            @keyframes path-loop {\n")
        hold_start_pct = (total_animation_time / total_cycle) * 100
        hold_end_pct = ((total_animation_time + hold_time) / total_cycle) * 100
        fade_end_pct = (
            (total_animation_time + hold_time + fade_time) / total_cycle
        ) * 100
        svg_parts.append(f"                0%, {hold_start_pct:.1f}% {{\n")
        svg_parts.append(
            "                    filter: drop-shadow(0 0 6px gold) drop-shadow(0 0 12px gold);\n"
        )
        svg_parts.append("                }\n")
        svg_parts.append(f"                {hold_end_pct:.1f}% {{\n")
        svg_parts.append(
            "                    filter: drop-shadow(0 0 6px gold) drop-shadow(0 0 12px gold);\n"
        )
        svg_parts.append("                }\n")
        svg_parts.append(f"                {fade_end_pct:.1f}%, 100% {{\n")
        svg_parts.append("                    filter: none;\n")
        svg_parts.append("                }\n")
        svg_parts.append("            }\n\n")

        # Generate node step animations
        for step in range(num_steps):
            delay = step * 0.5
            loop_delay = delay + 0.6  # 0.6s for the pulse animation
            svg_parts.append(f"            .path-step-{step} {{\n")
            svg_parts.append(
                f"                animation: highlight-pulse 0.6s ease-in-out {delay}s 2,\n"
            )
            svg_parts.append(
                f"                           path-loop {total_cycle}s linear {loop_delay}s infinite;\n"
            )
            svg_parts.append("            }\n")

        # Generate edge step animations (offset by 0.25s from node animations)
        for step in range(len(path)):
            delay = step * 0.5 + 0.25
            loop_delay = delay + 0.6
            svg_parts.append(f"            .edge-step-{step} {{\n")
            svg_parts.append(
                f"                animation: highlight-pulse 0.6s ease-in-out {delay}s 2,\n"
            )
            svg_parts.append(
                f"                           path-loop {total_cycle}s linear {loop_delay}s infinite;\n"
            )
            svg_parts.append("            }\n")

        svg_parts.append("        </style>\n    </defs>\n")

    svg_parts.append(
        '    <g class="topology" transform="translate(-12.017037256377762, -10.684619319104726) scale(1.2044823458562661)">\n'
    )

    # Generate qubits (nodes)
    for node_num in range(54):
        if node_num not in QUBIT_POSITIONS:
            continue

        x, y, label = QUBIT_POSITIONS[node_num]
        error = node_error_dict.get(node_num, min_error)
        color = interpolate_color(error, min_error, max_error)

        # Add path animation class if this node is in the path
        class_str = "qubit _component_1sx65_1"
        if node_num in node_step_map:
            class_str += f" path-step-{node_step_map[node_num]}"

        svg_parts.append(
            f'        <g class="{class_str}" transform="translate({x}, {y})"\n'
            f'            style="filter: none;">\n'
            f'            <circle r="22" stroke-dasharray="0" fill="{color}" '
            f'style="cursor: pointer;" transform=""></circle>\n'
            f'            <text text-anchor="middle" dy="0.35em" fill="white" font-size="13px"\n'
            f'                style="pointer-events: none;">{label}</text>\n'
            f"        </g>\n"
        )

    # Generate couplers (edges)
    for (start, end), (x, y, rotation) in COUPLER_CONNECTIONS.items():
        edge_key = tuple(sorted([start, end]))
        error = edge_error_dict.get(edge_key, min_error)
        color = interpolate_color(error, min_error, max_error)

        # Add path animation class if this edge is in the path
        class_str = "coupler _component_1sx65_1"
        if edge_key in path_edges_map:
            class_str += f" edge-step-{path_edges_map[edge_key]}"

        svg_parts.append(
            f'        <g class="{class_str}" '
            f'transform="translate({x}, {y}) rotate({rotation})"\n'
            f'            style="filter: none;">\n'
            f'            <rect width="30" height="30" x="-15" y="-15" rx="1" stroke-width="1"\n'
            f'                stroke-dasharray="0" fill="{color}" style="cursor: pointer;" '
            f'transform=""></rect>\n'
            f'            <text text-anchor="middle" dy="1.12em" fill="#8b8b8b" font-size="8px" \n'
            f'                pointer-events="none"></text>\n'
            f"        </g>\n"
        )

    svg_parts.append("    </g>\n</svg>")

    # Write to file
    with open(resolved_path, "w") as f:
        f.write("".join(svg_parts))

    print(f"Generated {resolved_path}")


if __name__ == "__main__":
    import sys

    example_node_errors = [
        (48, 1.375000000000004),
        (25, 1.2499999999999956),
        (52, 1.9750000000000045),
        (8, 2.100000000000002),
        (18, 1.849999999999996),
        (2, 1.8000000000000016),
        (21, 1.9000000000000017),
        (15, 1.0499999999999954),
        (32, 1.5750000000000042),
        (4, 2.749999999999997),
        (53, 3.774999999999995),
        (29, 2.1499999999999964),
        (10, 1.4249999999999985),
        (44, 1.7750000000000044),
        (6, 4.725000000000001),
        (31, 2.0750000000000046),
        (47, 1.924999999999999),
        (12, 1.7249999999999988),
        (42, 1.924999999999999),
        (30, 5.6499999999999995),
        (35, 1.6249999999999987),
        (46, 3.300000000000003),
        (43, 5.85),
        (23, 1.749999999999996),
        (1, 1.2499999999999956),
        (39, 2.3249999999999993),
        (13, 1.8249999999999988),
        (24, 2.275000000000005),
        (27, 4.174999999999995),
        (3, 9.299999999999997),
        (49, 2.0000000000000018),
        (16, 2.300000000000002),
        (28, 4.425000000000001),
        (34, 2.9750000000000054),
        (38, 2.5249999999999995),
        (33, 1.2249999999999983),
        (5, 0.8750000000000036),
        (9, 2.649999999999997),
        (40, 1.7249999999999988),
        (0, 2.1750000000000047),
        (51, 2.300000000000002),
        (45, 2.575000000000005),
        (11, 2.0499999999999963),
        (26, 1.375000000000004),
        (7, 1.275000000000004),
        (20, 1.3000000000000012),
        (19, 2.1750000000000047),
        (22, 1.2249999999999983),
        (36, 2.7249999999999996),
        (17, 1.100000000000001),
        (37, 2.124999999999999),
        (50, 5.449999999999999),
        (41, 2.3249999999999993),
        (14, 2.0750000000000046),
    ]

    example_edge_errors = [
        ([52, 53], 0.9155800964991245),
        ([3, 4], 1.8192316747080772),
        ([8, 7], 0.8837752740216809),
        ([12, 11], 0.9179964548055808),
        ([0, 4], 0.4257549737402777),
        ([17, 18], 0.2892841978012206),
        ([15, 7], 0.9855673254395625),
        ([24, 23], 0.7284042308198013),
        ([19, 11], 0.564546975788438),
        ([21, 13], 2.307193598446955),
        ([28, 27], 0.22167952116319833),
        ([24, 32], 0.5802886120780393),
        ([33, 34], 1.386738188299319),
        ([26, 34], 1.298701162486282),
        ([37, 38], 0.4757243476306794),
        ([28, 36], 0.4290387536496798),
        ([40, 39], 0.3029503978040027),
        ([44, 43], 0.3891400040439974),
        ([30, 38], 0.15279853480204553),
        ([47, 48], 0.26309781865180293),
        ([47, 41], 0.4781683165119399),
        ([5, 4], 0.7060813815400802),
        ([8, 9], 0.4725270605374421),
        ([43, 49], 7.441027613791384),
        ([8, 2], 0.1789868354317603),
        ([12, 13], 0.3124558953408352),
        ([15, 14], 0.8528258775608033),
        ([10, 4], 1.2190286433494846),
        ([15, 23], 0.7590435226697223),
        ([19, 18], 0.23497714522078272),
        ([17, 25], 1.287840147756203),
        ([24, 25], 0.42324688082245876),
        ([28, 29], 0.8232818908602946),
        ([19, 27], 0.6036426716165555),
        ([35, 34], 0.5154203254698619),
        ([21, 29], 3.828431211490979),
        ([40, 32], 0.8966610275229048),
        ([40, 41], 0.19168903490524025),
        ([44, 45], 0.9923925197859562),
        ([5, 6], 0.4204247051326382),
        ([34, 42], 1.507906821331917),
        ([10, 9], 0.6636227556801599),
        ([44, 36], 0.7065378485403784),
        ([47, 51], 0.937432189311882),
        ([15, 16], 0.49705311576799804),
        ([19, 20], 0.31972606602314046),
        ([49, 53], 1.3614456624297167),
        ([5, 1], 0.7347207561420821),
        ([25, 26], 0.9224362410974796),
        ([30, 29], 0.6561243117543025),
        ([8, 16], 0.8686489120027563),
        ([10, 18], 0.1844503641643791),
        ([31, 32], 0.35412621646764286),
        ([35, 36], 0.31000562740644),
        ([41, 42], 0.3146466929474956),
        ([12, 20], 1.446927754521643),
        ([49, 50], 0.8950663667080749),
        ([52, 51], 1.3691388044951625),
        ([31, 23], 0.6084517668385026),
        ([33, 25], 0.47185893050541994),
        ([35, 27], 1.105280334830916),
        ([40, 46], 0.8937829418373955),
        ([44, 50], 0.0857038840365365),
        ([3, 9], 0.5735334860861974),
        ([5, 11], 0.46191574895697585),
        ([22, 14], 0.21275771631790175),
        ([24, 16], 0.6479119798293964),
        ([18, 26], 1.060549061816396),
        ([28, 20], 3.992889730025484),
        ([33, 41], 0.6790866218461589),
        ([35, 43], 0.5091354090133837),
        ([52, 48], 0.19723307076128238),
        ([0, 1], 0.2323351364436177),
        ([3, 2], 1.2249253218507228),
        ([10, 11], 0.22812821171529496),
        ([17, 16], 0.6612857561426821),
        ([21, 20], 1.7297983852624599),
        ([22, 23], 0.33762959353725863),
        ([26, 27], 3.1258999408530386),
        ([33, 32], 0.2460590329208978),
        ([36, 37], 0.48785240074543657),
        ([47, 46], 0.3287896774973653),
    ]

    example_path = [
        [0, 1],
        [1, 5],
        [5, 11],
        [11, 10],
        [10, 18],
        [18, 17],
        [17, 25],
        [25, 33],
        [33, 32],
        [32, 31],
        [31, 23],
        [23, 22],
    ]

    generate_emerald_svg(example_node_errors, example_edge_errors, path=example_path)
