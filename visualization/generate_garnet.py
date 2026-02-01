#!/usr/bin/env python3
"""
Generate garnet.svg with error-based coloring for nodes and edges.

The color scale goes from #7ad151 (minimum error) to #482475 (maximum error).
"""

import colorsys
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
    Interpolate color between min_color and max_color based on value.

    Args:
        value: The error percentage value
        min_val: Minimum error percentage in dataset
        max_val: Maximum error percentage in dataset
        min_color: Color for minimum error (default: green #7ad151)
        max_color: Color for maximum error (default: purple #482475)

    Returns:
        Hex color string
    """
    if max_val == min_val:
        return min_color

    # Normalize value to 0-1 range
    t = (value - min_val) / (max_val - min_val)

    # Convert hex to RGB
    min_rgb = hex_to_rgb(min_color)
    max_rgb = hex_to_rgb(max_color)

    # Interpolate each channel
    r = int(min_rgb[0] + t * (max_rgb[0] - min_rgb[0]))
    g = int(min_rgb[1] + t * (max_rgb[1] - min_rgb[1]))
    b = int(min_rgb[2] + t * (max_rgb[2] - min_rgb[2]))

    return rgb_to_hex((r, g, b))


# Qubit positions (0-indexed node_num -> (x, y, label))
QUBIT_POSITIONS = {
    0: (517.165, 601.495, "QB1"),
    1: (584.33, 534.33, "QB2"),
    2: (382.83500000000004, 601.495, "QB3"),
    3: (450, 534.33, "QB4"),
    4: (517.165, 467.165, "QB5"),
    5: (584.33, 400, "QB6"),
    6: (651.495, 332.83500000000004, "QB7"),
    7: (315.66999999999996, 534.33, "QB8"),
    8: (382.83500000000004, 467.165, "QB9"),
    9: (450, 400, "QB10"),
    10: (517.165, 332.83500000000004, "QB11"),
    11: (584.33, 265.67, "QB12"),
    12: (248.505, 467.165, "QB13"),
    13: (315.66999999999996, 400, "QB14"),
    14: (382.83500000000004, 332.83500000000004, "QB15"),
    15: (450, 265.67, "QB16"),
    16: (517.165, 198.505, "QB17"),
    17: (248.505, 332.83500000000004, "QB18"),
    18: (315.66999999999996, 265.67, "QB19"),
    19: (382.83500000000004, 198.505, "QB20"),
}

# Coupler (edge) definitions: (start_node, end_node) -> (x, y, rotation)
# These are the connections between qubits
COUPLER_CONNECTIONS = {
    (0, 1): (550.795, 567.96, 225),
    (0, 4): (550.795, 500.79499999999996, 315),
    (2, 8): (416.37, 500.79499999999996, 315),
    (3, 4): (483.63, 500.79499999999996, 225),
    (4, 5): (550.795, 433.63, 225),
    (4, 9): (483.63, 433.63, 315),
    (5, 10): (550.795, 366.37, 315),
    (6, 10): (617.96, 299.205, 315),
    (7, 8): (349.205, 500.79499999999996, 225),
    (7, 12): (282.04, 500.79499999999996, 315),
    (8, 9): (416.37, 433.63, 225),
    (8, 13): (349.205, 433.63, 315),
    (9, 10): (483.63, 366.37, 225),
    (9, 14): (416.37, 366.37, 315),
    (10, 11): (550.795, 299.205, 225),
    (10, 15): (483.63, 299.205, 315),
    (11, 15): (550.795, 232.04, 315),
    (12, 13): (282.04, 433.63, 225),
    (13, 14): (349.205, 366.37, 225),
    (13, 17): (282.04, 366.37, 315),
    (14, 15): (416.37, 299.205, 225),
    (14, 18): (349.205, 299.205, 315),
    (15, 16): (483.63, 232.04, 225),
    (15, 19): (416.37, 232.04, 315),
    (17, 18): (282.04, 299.205, 225),
    (18, 19): (349.205, 232.04, 225),
    (5, 6): (617.96, 366.37, 225),
    (11, 16): (550.795, 232.04, 225),
    (3, 8): (416.4175, 500.79499999999996, 225),
}


def generate_garnet_svg(
    node_errors: Sequence[Tuple[int, float]],
    edge_errors: Sequence[Tuple[Sequence[int], float]],
    output_path: str = "garnet.svg",
    path: Optional[Sequence[Sequence[int]]] = None,
) -> None:
    """
    Generate garnet.svg with error-based coloring and optional path animation.

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
        '    <g class="topology" transform="translate(-55, -189.45650634765624) scale(1.3)">\n'
    )

    # Generate qubits (nodes)
    for node_num in range(20):
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

    # Example usage - replace with your actual data
    example_node_errors = [
        (4, 3.4499999999999975),
        (16, 1.7750000000000044),
        (3, 7.325000000000004),
        (18, 1.8000000000000016),
        (11, 2.3249999999999993),
        (6, 1.4000000000000012),
        (13, 1.8000000000000016),
        (14, 3.2749999999999946),
        (12, 1.8750000000000044),
        (10, 2.849999999999997),
        (19, 1.375000000000004),
        (17, 1.5750000000000042),
        (15, 1.6750000000000043),
        (1, 1.1499999999999955),
        (0, 1.9499999999999962),
        (8, 6.25),
        (5, 2.4249999999999994),
        (7, 2.500000000000002),
        (2, 1.9000000000000017),
        (9, 1.9750000000000045),
    ]

    example_edge_errors = [
        ([18, 17], 0.18368804661031968),
        ([9, 10], 0.29178665557484385),
        ([13, 12], 1.1162589543305956),
        ([18, 19], 0.12760287712831886),
        ([3, 4], 7.698242462739158),
        ([11, 10], 0.4849068709610771),
        ([13, 14], 0.9584088942435565),
        ([1, 4], 0.21131905149636143),
        ([13, 8], 0.4934355869787632),
        ([15, 10], 0.5757440802106051),
        ([9, 4], 0.5104884287033373),
        ([11, 6], 0.25524898234603466),
        ([13, 17], 0.42224936140939917),
        ([15, 19], 0.36463650336719944),
        ([7, 12], 0.6137507088124439),
        ([9, 14], 0.6632456296422617),
        ([11, 16], 0.39494925123836344),
        ([3, 8], 6.19555484287252),
        ([5, 10], 0.8317877542178387),
        ([18, 14], 0.39979274222825545),
        ([5, 4], 0.8913503940611966),
        ([7, 8], 1.2429394607743616),
        ([15, 14], 0.26519635616050197),
        ([1, 0], 0.2398695990178834),
        ([5, 6], 0.5416455042012602),
        ([9, 8], 1.0088600974843764),
        ([15, 16], 0.38932518390986104),
    ]

    # Example path through the qubits
    example_path = [
        [0, 1],
        [1, 4],
        [4, 9],
        [9, 10],
        [10, 5],
        [5, 6],
        [6, 11],
        [11, 16],
        [16, 15],
        [15, 19],
        [19, 18],
        [18, 17],
        [17, 13],
        [13, 12],
        [12, 7],
    ]

    generate_garnet_svg(example_node_errors, example_edge_errors, path=example_path)
