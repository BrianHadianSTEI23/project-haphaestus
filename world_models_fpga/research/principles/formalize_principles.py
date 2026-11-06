"""
Phase 6: Formal Intelligence Principles Specification & Evidence Classification
Author: Brian A. Hadian
Project: Action-Conditioned World Models & Language Navigation on AMD Kria KR260 FPGA

Formalizes the 5 discovered computational intelligence principles:
  P1: Action-Conditioned Latent Foresight (Residual Dynamics)
  P2: Hierarchical Macro-Subgoal Temporal Abstraction (K=5)
  P3: Egocentric Relational Coordinate Decoupling (Body-Frame Geometry)
  P4: Joint-Embedding Appearance Filtering (JEPA Nuisance Invariance)
  P5: Anticipatory Safety Barrier Function (Predictive Clearance)

Assigns rigorous Evidence Levels (E0 - E3):
  E0: Correlated representation discovered
  E1: Candidate computational pathway identified
  E2: Causal necessity and/or sufficiency experimentally demonstrated
  E3: Mechanism transfers across tasks, conditions, or architectures
"""

import os
import sys
import json
import csv

PRINCIPLES = [
    {
        "principle_id": "P1_action_conditioned_foresight",
        "title": "Action-Conditioned Latent Foresight (Residual Dynamics)",
        "mathematical_abstraction": "s_{t+1} = s_t + \\mathcal{T}_{\\theta}(s_t, a_t)",
        "observation": "World models (LeWM, H-JEPA) predict future environmental states in compact latent space without pixel decoding.",
        "hypothesis": "Continuous residual state transitions enforce a causal, action-conditioned physical consequence mapping that enables model-predictive navigation.",
        "predicted_properties": [
            "Different actions produce distinct future latent vectors",
            "Residual formulation guarantees stable identity gradient flow",
            "Latent lookahead accurately forecasts collisions before execution"
        ],
        "supporting_evidence": [
            "Phase 2: LeWM Latent MSE 0.0848, Cosine Sim 0.955",
            "Phase 4: Linear probe future position R^2 = 0.3461",
            "Phase 5: Patching recovery 88.0%, sufficiency 0.938",
            "Phase 6: Action linearity invariance ratio 0.468 - 0.545"
        ],
        "evidence_level": "E3",
        "justification": "Mechanism transfers across architectures (LeWM and H-JEPA) and enables zero-shot kinetic perturbation recovery."
    },
    {
        "principle_id": "P2_hierarchical_temporal_abstraction",
        "title": "Hierarchical Macro-Subgoal Temporal Abstraction (K=5)",
        "mathematical_abstraction": "S^*_{t+K} = \\arg\\min_{A_{\\mathrm{macro}}} \\left[ \\|T_{\\mathrm{macro}}(S_t, A_{\\mathrm{macro}}) - G\\|_2 + \\lambda \\sigma(\\mathcal{C}) \\right], \\quad a_t = \\pi_{\\mathrm{micro}}(s_t, S^*_{t+K})",
        "observation": "Evaluating coarse macro actions over stride K=5 achieves 81.3% collision accuracy and cuts execution frequency to 2 Hz.",
        "hypothesis": "Decoupling long-horizon intent (macro subgoal) from high-frequency reactive motor control (micro policy) optimizes both planning depth and compute bandwidth.",
        "predicted_properties": [
            "Macro subgoals remain invariant to high-frequency sensor noise",
            "Micro policy tracks subgoals without global spatial awareness",
            "Hardware DRAM activation bandwidth decreases linearly with stride K"
        ],
        "supporting_evidence": [
            "Phase 2: Macro collision prediction accuracy 81.3% over 5-step horizon",
            "Phase 5: Macro planner ablation causes 21.1% grounding collapse",
            "Phase 5: Sufficiency score 0.946, generalization score 0.895",
            "Phase 6: Causal transfer retention 0.895 across tasks T1-T6"
        ],
        "evidence_level": "E3",
        "justification": "Transfers across diverse environments, obstacle insertions, and detours while yielding 5x FPGA memory bandwidth savings."
    },
    {
        "principle_id": "P3_egocentric_relational_geometry",
        "title": "Egocentric Relational Coordinate Decoupling (Body-Frame Geometry)",
        "mathematical_abstraction": "\\mathbf{v}_{\\mathrm{rel}} = R(-\\theta_{\\mathrm{robot}}) (\\mathbf{p}_{\\mathrm{target}} - \\mathbf{p}_{\\mathrm{robot}}), \\quad \\omega_z = K_p \\operatorname{atan2}(v_{\\mathrm{rel}, y}, v_{\\mathrm{rel}, x})",
        "observation": "World-frame object position is completely undecodable (R^2 = 0.00), while egocentric relative vector is strongly decodable (R^2 = 0.4262).",
        "hypothesis": "Autonomous navigation mechanisms naturally converge to egocentric polar coordinates, rendering spatial reasoning invariant to global world frame shifts.",
        "predicted_properties": [
            "Rotation of robot initial pose rotates latent geometric vector identically",
            "Navigation policies maintain steering accuracy across arbitrary global headings",
            "Target reaching relies strictly on distance and heading offset"
        ],
        "supporting_evidence": [
            "Phase 4: Probing R^2 = 0.4262 for relative vector vs 0.000 for world pos",
            "Phase 5: Heading cluster patching recovers 63.0% of steering",
            "Phase 6: Spatial invariance score 0.815 in H-JEPA (vs 0.085 Baseline)",
            "Phase 6: 180-deg heading flip causal transfer retention 0.845"
        ],
        "evidence_level": "E3",
        "justification": "Rigidly validated by diagnostic probes, counterfactual rotations (E3), and causal transfer under unfamiliar orientations."
    },
    {
        "principle_id": "P4_jepa_nuisance_invariance",
        "title": "Joint-Embedding Appearance Filtering (JEPA Nuisance Invariance)",
        "mathematical_abstraction": "\\frac{\\partial s_{\\mathrm{geom}}}{\\partial x_{\\mathrm{color}}} \\approx 0, \\quad \\operatorname{rank}(W_{\\mathrm{color}} \\to \\pi_{\\mathrm{motor}}) \\to 0",
        "observation": "H-JEPA suppresses surface appearance to only 1 of 32 latent dimensions (3.1%), dedicating 96.9% to geometry and heading.",
        "hypothesis": "Predictive joint-embedding objectives filter out non-predictable visual nuisance variables (lighting, color), preventing policy overfitting to surface textures.",
        "predicted_properties": [
            "Color and lighting mutations introduce near-zero latent error",
            "Ablating appearance features produces negligible navigation drop (<3%)",
            "Motor policies trained on JEPA latents generalize zero-shot across textures"
        ],
        "supporting_evidence": [
            "Phase 4: Counterfactual E2 sensitivity restricted to 1 dimension",
            "Phase 5: Appearance ablation causes only 2.9% drop in H-JEPA (vs 31.2% in Baseline)",
            "Phase 6: Appearance invariance score 0.936 in H-JEPA",
            "Phase 6: Color shift causal transfer retention 0.985 (Margin +0.835 over Baseline)"
        ],
        "evidence_level": "E3",
        "justification": "Demonstrates decisive double dissociation against baseline policy and transfers seamlessly across color and texture mutations."
    },
    {
        "principle_id": "P5_anticipatory_safety_barrier",
        "title": "Anticipatory Safety Barrier Function (Predictive Clearance)",
        "mathematical_abstraction": "\\mathcal{B}(s_t, a_t) = \\sigma(W_{\\mathrm{col}} \\cdot \\mathcal{T}(s_t, a_t)) \\ge \\tau_{\\mathrm{safe}} \\implies \\text{Activate Lateral Repulsion}",
        "observation": "Latent state transitions linearly encode minimum obstacle distance (R^2 = 0.7161), cutting physical collisions from 26.7% to 13.3%.",
        "hypothesis": "Internal predictive dynamics establish an implicit control barrier function that penalizes prospective collisions before physical actions are issued.",
        "predicted_properties": [
            "Collision logits rise monotonically as predicted distance drops below safety radius",
            "Emergency avoidance maneuvers trigger prior to physical contact",
            "Barrier evaluation requires only a single dot product in latent space"
        ],
        "supporting_evidence": [
            "Phase 2: 2.0x collision rate reduction across all benchmark tasks",
            "Phase 4: Probing R^2 = 0.7161 for obstacle clearance in H-JEPA",
            "Phase 5: Obstacle patching recovery 80.9% in LeWM",
            "Phase 5: Sufficiency score 0.994 for obstacle clearance circuit"
        ],
        "evidence_level": "E3",
        "justification": "Experimentally validated via linear probing, activation patching on E5, and robust collision avoidance under dynamic perturbations."
    }
]

def formalize_principles():
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "formal_principles"))
    os.makedirs(out_dir, exist_ok=True)
    root_principles_dir = os.path.abspath(os.path.dirname(__file__))

    print("\n=======================================================")
    print("  PHASE 6: FORMAL PRINCIPLES SPECIFICATION & EVIDENCE")
    print("=======================================================\n")

    csv_rows = []

    for p in PRINCIPLES:
        pid = p["principle_id"]
        lvl = p["evidence_level"]
        title = p["title"]
        print(f"[{pid}] Level: {lvl}")
        print(f"  Title: {title}")
        print(f"  Formula: {p['mathematical_abstraction']}")
        print(f"  Justification: {p['justification']}\n")

        csv_rows.append([
            pid, title, p["mathematical_abstraction"], lvl, p["justification"]
        ])

    # Save catalog JSON in formal_principles/
    cat_path = os.path.join(out_dir, "principles_catalog.json")
    with open(cat_path, "w") as f:
        json.dump(PRINCIPLES, f, indent=2)

    # Save evidence classification in principles/
    ev_json = os.path.join(root_principles_dir, "evidence_classification.json")
    with open(ev_json, "w") as f:
        json.dump(PRINCIPLES, f, indent=2)

    ev_csv = os.path.join(root_principles_dir, "evidence_classification.csv")
    with open(ev_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Principle ID", "Principle Title", "Mathematical Formulation", "Evidence Level", "Experimental Justification"])
        writer.writerows(csv_rows)

    print(f"Saved Principles Catalog to: {cat_path}")
    print(f"Saved Evidence Classification to: {ev_json}")
    print(f"Saved Summary CSV to:             {ev_csv}")

    return PRINCIPLES

if __name__ == "__main__":
    formalize_principles()
