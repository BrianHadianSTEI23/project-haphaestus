"""
Phase 7: Circuit Extraction & Neural Architecture Reconstruction
Script: build_circuit_dependency_graph.py
Author: Brian A. Hadian
Project: Action-Conditioned World Models & Language Navigation on AMD Kria KR260 FPGA

Constructs the formal directed circuit dependency graph for H-JEPA and world model architectures,
categorizing nodes into Essential, Redundant, Shared, Task-Specific, and Replaceable components.
Exports JSON, DOT (for Graphviz), and CSV summary tables.
"""

import os
import sys
import json
import csv

def build_dependency_graph():
    out_dir = os.path.abspath(os.path.dirname(__file__))
    os.makedirs(out_dir, exist_ok=True)

    print("\n=======================================================")
    print("  PHASE 7: BUILDING CIRCUIT DEPENDENCY GRAPH")
    print("=======================================================\n")

    # Nodes definition
    nodes = [
        {
            "id": "rgb_obs",
            "name": "RGB Camera Observation (64x64x3)",
            "module": "sensor_interface",
            "type": "input",
            "params": 0,
            "role": "essential",
            "circuits": ["spatial_relation_grounding_001", "obstacle_clearance_boundary_002", "appearance_filtering_invariance_006"],
            "pruning_recommendation": "retain_input",
            "rationale": "Primary perceptual modality for visual servoing and obstacle detection."
        },
        {
            "id": "encoder_conv1",
            "name": "Encoder Conv2d Layer 1 (3 -> 16, 4x4, s=2)",
            "module": "encoder",
            "type": "convolution",
            "params": 784,
            "role": "shared",
            "circuits": ["spatial_relation_grounding_001", "obstacle_clearance_boundary_002", "appearance_filtering_invariance_006"],
            "pruning_recommendation": "prune_channels_25pct",
            "rationale": "Extracts low-level edges and textures. 4 of 16 channels respond to high-frequency color and can be pruned."
        },
        {
            "id": "encoder_conv2",
            "name": "Encoder Conv2d Layer 2 (16 -> 32, 4x4, s=2)",
            "module": "encoder",
            "type": "convolution",
            "params": 8224,
            "role": "shared",
            "circuits": ["spatial_relation_grounding_001", "obstacle_clearance_boundary_002", "appearance_filtering_invariance_006"],
            "pruning_recommendation": "prune_channels_25pct",
            "rationale": "Extracts mid-level spatial contours and obstacle silhouettes."
        },
        {
            "id": "encoder_conv3",
            "name": "Encoder Conv2d Layer 3 (32 -> 64, 4x4, s=2)",
            "module": "encoder",
            "type": "convolution",
            "params": 32832,
            "role": "replaceable",
            "circuits": ["spatial_relation_grounding_001", "obstacle_clearance_boundary_002"],
            "pruning_recommendation": "prune_channels_50pct",
            "rationale": "High parameter cost (32.8K). Channel attribution shows 50% redundancy with no loss in clearance accuracy."
        },
        {
            "id": "encoder_conv4",
            "name": "Encoder Conv2d Layer 4 (64 -> 64, 4x4, s=2)",
            "module": "encoder",
            "type": "convolution",
            "params": 65600,
            "role": "replaceable",
            "circuits": ["spatial_relation_grounding_001", "obstacle_clearance_boundary_002"],
            "pruning_recommendation": "prune_channels_50pct",
            "rationale": "Largest block in model (65.6K params, 39.5% of total). Can be reduced from 64 to 32 filters."
        },
        {
            "id": "encoder_linear_proj",
            "name": "Encoder Latent Linear Projection (1024 -> 32)",
            "module": "encoder",
            "type": "linear",
            "params": 32800,
            "role": "replaceable",
            "circuits": ["spatial_relation_grounding_001", "obstacle_clearance_boundary_002", "egocentric_heading_steering_004"],
            "pruning_recommendation": "compress_to_16dim",
            "rationale": "Compresses spatial feature map into latent state. Phase 4 proved 16 dimensions suffice for 98.5% variance."
        },
        {
            "id": "encoder_layernorm",
            "name": "Encoder Latent LayerNorm (32)",
            "module": "encoder",
            "type": "normalization",
            "params": 64,
            "role": "essential",
            "circuits": ["action_conditioned_foresight_005", "macro_subgoal_planning_003"],
            "pruning_recommendation": "retain_compressed_16dim",
            "rationale": "Crucial for dynamic range bounding within [-3.0, +3.0], ensuring Q8.8 fixed-point stability."
        },
        {
            "id": "action_candidate_input",
            "name": "Macro Action Candidate Input (3-dim [vx, vy, wz])",
            "module": "mpc_planner",
            "type": "input",
            "params": 0,
            "role": "essential",
            "circuits": ["macro_subgoal_planning_003", "action_conditioned_foresight_005"],
            "pruning_recommendation": "retain_input",
            "rationale": "Hypothesized candidate actions over macro stride K=5."
        },
        {
            "id": "macro_fc1",
            "name": "Macro Predictor FC1 (35 -> 64)",
            "module": "macro_predictor",
            "type": "linear_mlp",
            "params": 2304,
            "role": "shared",
            "circuits": ["macro_subgoal_planning_003", "obstacle_clearance_boundary_002", "action_conditioned_foresight_005"],
            "pruning_recommendation": "prune_mlp_to_32dim",
            "rationale": "Joint embedding projection of latent state and macro action. Can be pruned from 64 to 32 hidden units."
        },
        {
            "id": "macro_fc2",
            "name": "Macro Predictor FC2 (64 -> 64)",
            "module": "macro_predictor",
            "type": "linear_mlp",
            "params": 4160,
            "role": "shared",
            "circuits": ["macro_subgoal_planning_003", "obstacle_clearance_boundary_002", "action_conditioned_foresight_005"],
            "pruning_recommendation": "prune_mlp_to_32dim",
            "rationale": "Hidden representation of action-conditioned transition dynamics."
        },
        {
            "id": "macro_delta_head",
            "name": "Macro Latent Delta Head (64 -> 32)",
            "module": "macro_predictor",
            "type": "prediction_head",
            "params": 2080,
            "role": "essential",
            "circuits": ["macro_subgoal_planning_003", "action_conditioned_foresight_005"],
            "pruning_recommendation": "compress_head_to_16dim",
            "rationale": "Predicts forward residual latent change delta_S for multi-step subgoal optimization."
        },
        {
            "id": "macro_col_head",
            "name": "Macro Collision Head (64 -> 1)",
            "module": "macro_predictor",
            "type": "prediction_head",
            "params": 65,
            "role": "essential",
            "circuits": ["obstacle_clearance_boundary_002", "macro_subgoal_planning_003"],
            "pruning_recommendation": "retain_100pct",
            "rationale": "Implements the internal safety barrier function B(s, a). Crucial for obstacle penalty."
        },
        {
            "id": "micro_pol_fc1",
            "name": "Micro Policy FC1 (64 -> 64)",
            "module": "micro_policy",
            "type": "linear_mlp",
            "params": 4160,
            "role": "replaceable",
            "circuits": ["egocentric_heading_steering_004"],
            "pruning_recommendation": "prune_mlp_to_32dim",
            "rationale": "Maps current state and macro subgoal into primitive velocity commands."
        },
        {
            "id": "micro_pol_fc2",
            "name": "Micro Policy FC2 (64 -> 64)",
            "module": "micro_policy",
            "type": "linear_mlp",
            "params": 4160,
            "role": "replaceable",
            "circuits": ["egocentric_heading_steering_004"],
            "pruning_recommendation": "prune_mlp_to_32dim",
            "rationale": "Processes goal-tracking feature representations."
        },
        {
            "id": "micro_action_head",
            "name": "Micro Policy Action Head (64 -> 3)",
            "module": "micro_policy",
            "type": "prediction_head",
            "params": 195,
            "role": "essential",
            "circuits": ["egocentric_heading_steering_004"],
            "pruning_recommendation": "retain_100pct",
            "rationale": "Outputs physical Mecanum wheel drive commands [vx, vy, wz]."
        },
        {
            "id": "micro_dyn_fc1",
            "name": "Micro Dynamics Predictor FC1 (35 -> 64)",
            "module": "micro_dynamics_auxiliary",
            "type": "linear_mlp",
            "params": 2304,
            "role": "redundant",
            "circuits": [],
            "pruning_recommendation": "prune_module_100pct",
            "rationale": "Training auxiliary for 1-step self-supervision; completely unreferenced during hierarchical navigation."
        },
        {
            "id": "micro_dyn_fc2",
            "name": "Micro Dynamics Predictor FC2 (64 -> 64)",
            "module": "micro_dynamics_auxiliary",
            "type": "linear_mlp",
            "params": 4160,
            "role": "redundant",
            "circuits": [],
            "pruning_recommendation": "prune_module_100pct",
            "rationale": "Training auxiliary; redundant during inference deployment."
        },
        {
            "id": "micro_dyn_delta",
            "name": "Micro Dynamics Delta Head (64 -> 32)",
            "module": "micro_dynamics_auxiliary",
            "type": "prediction_head",
            "params": 2080,
            "role": "redundant",
            "circuits": [],
            "pruning_recommendation": "prune_module_100pct",
            "rationale": "Training auxiliary; redundant during inference deployment."
        },
        {
            "id": "micro_dyn_col",
            "name": "Micro Dynamics Collision Head (64 -> 1)",
            "module": "micro_dynamics_auxiliary",
            "type": "prediction_head",
            "params": 65,
            "role": "redundant",
            "circuits": [],
            "pruning_recommendation": "prune_module_100pct",
            "rationale": "Training auxiliary; redundant during inference deployment."
        },
        {
            "id": "sigreg_regularizer",
            "name": "SIGReg Spectral Regularizer",
            "module": "loss_module",
            "type": "loss_module",
            "params": 0,
            "role": "task_specific",
            "circuits": [],
            "pruning_recommendation": "prune_module_100pct",
            "rationale": "Training loss operator only; zero weights and zero computation at inference time."
        }
    ]

    # Directed Edges definition
    edges = [
        {"from": "rgb_obs", "to": "encoder_conv1", "type": "feedforward"},
        {"from": "encoder_conv1", "to": "encoder_conv2", "type": "feedforward"},
        {"from": "encoder_conv2", "to": "encoder_conv3", "type": "feedforward"},
        {"from": "encoder_conv3", "to": "encoder_conv4", "type": "feedforward"},
        {"from": "encoder_conv4", "to": "encoder_linear_proj", "type": "feedforward"},
        {"from": "encoder_linear_proj", "to": "encoder_layernorm", "type": "feedforward"},
        # Latent state to Macro Predictor
        {"from": "encoder_layernorm", "to": "macro_fc1", "type": "feedforward"},
        {"from": "action_candidate_input", "to": "macro_fc1", "type": "feedforward"},
        {"from": "macro_fc1", "to": "macro_fc2", "type": "feedforward"},
        {"from": "macro_fc2", "to": "macro_delta_head", "type": "feedforward"},
        {"from": "macro_fc2", "to": "macro_col_head", "type": "feedforward"},
        # Latent state to Micro Policy
        {"from": "encoder_layernorm", "to": "micro_pol_fc1", "type": "feedforward"},
        {"from": "macro_delta_head", "to": "micro_pol_fc1", "type": "subgoal_conditioning"},
        {"from": "micro_pol_fc1", "to": "micro_pol_fc2", "type": "feedforward"},
        {"from": "micro_pol_fc2", "to": "micro_action_head", "type": "feedforward"},
        # Redundant Training Branches
        {"from": "encoder_layernorm", "to": "micro_dyn_fc1", "type": "training_auxiliary"},
        {"from": "micro_dyn_fc1", "to": "micro_dyn_fc2", "type": "training_auxiliary"},
        {"from": "micro_dyn_fc2", "to": "micro_dyn_delta", "type": "training_auxiliary"},
        {"from": "micro_dyn_fc2", "to": "micro_dyn_col", "type": "training_auxiliary"},
        {"from": "encoder_layernorm", "to": "sigreg_regularizer", "type": "training_loss"}
    ]

    # Compute parameter statistics by role
    role_params = {}
    for n in nodes:
        r = n["role"]
        role_params[r] = role_params.get(r, 0) + n["params"]

    total_params = sum(n["params"] for n in nodes)
    print(f"Total Model Parameters: {total_params:,}")
    print("\nParameter Breakdown by Architectural Role:")
    for r, p in sorted(role_params.items(), key=lambda x: -x[1]):
        pct = (p / total_params) * 100
        print(f"  - {r.upper():<16}: {p:>8,} params ({pct:>5.1f}%)")

    # Export JSON
    graph_dict = {
        "metadata": {
            "author": "Brian A. Hadian",
            "project": "Action-Conditioned World Models & Language Navigation on AMD Kria KR260 FPGA",
            "phase": "Phase 7 - Circuit Dependency Graph",
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "total_parameters": total_params,
            "parameter_breakdown_by_role": role_params
        },
        "nodes": nodes,
        "edges": edges
    }

    json_path = os.path.join(out_dir, "circuit_dependency_graph.json")
    with open(json_path, "w") as f:
        json.dump(graph_dict, f, indent=2)
    print(f"\n[Saved JSON] {json_path}")

    # Export CSV summary
    csv_path = os.path.join(out_dir, "dependency_graph_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Node ID", "Name", "Module", "Type", "Parameters", "Role", "Circuits Supported", "Pruning Action"])
        for n in nodes:
            writer.writerow([
                n["id"], n["name"], n["module"], n["type"], n["params"],
                n["role"], "; ".join(n["circuits"]), n["pruning_recommendation"]
            ])
    print(f"[Saved CSV]  {csv_path}")

    # Export DOT file for visualization
    dot_path = os.path.join(out_dir, "circuit_dependency_graph.dot")
    with open(dot_path, "w") as f:
        f.write("digraph CircuitDependencyGraph {\n")
        f.write("  rankdir=LR;\n")
        f.write("  node [shape=box, style=filled, fontname=\"Helvetica\", fontsize=10];\n")
        f.write("  edge [fontname=\"Helvetica\", fontsize=8];\n\n")

        role_colors = {
            "essential": "#98FB98",      # Pale green
            "shared": "#87CEEB",         # Sky blue
            "replaceable": "#FFD700",    # Gold
            "task_specific": "#DDA0DD",  # Plum
            "redundant": "#FFB6C1"       # Light pink (prune)
        }

        for n in nodes:
            col = role_colors.get(n["role"], "#FFFFFF")
            lbl = f"{n['name']}\\n[{n['role'].upper()}: {n['params']:,} params]"
            f.write(f"  \"{n['id']}\" [label=\"{lbl}\", fillcolor=\"{col}\"];\n")

        f.write("\n")
        for e in edges:
            style = "dashed, color=red" if "training" in e["type"] else "solid, color=black"
            f.write(f"  \"{e['from']}\" -> \"{e['to']}\" [style=\"{style}\", label=\"{e['type']}\"];\n")

        f.write("}\n")
    print(f"[Saved DOT]  {dot_path}")

    print("\nDependency Graph analysis successfully completed.")

if __name__ == "__main__":
    build_dependency_graph()
