# Independent Out-of-Sample Validation Results

Evaluation across datasets completely separated from the training and distillation phases:

### 1. Novel Independent Validation Suite (1,821 Transitions, Seed 8888)

| Model Variant | Parameters | Dynamics Prediction MSE | Collision Accuracy (%) | Relative Grounding Align | Action Tracking MSE | Kinematic Robustness |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original H-JEPA (Teacher/Target)** | 166,037 | **0.21519** | **92.0%** | **0.1003** | **0.26152** | **0.9917** |
| **Option A: Weight-Preserving** | 157,428 | **0.1969** | **92.0%** | **0.1003** | **0.26456** | **0.9921** |
| **Option B: Reconstructed Compact** | **39,804** | **0.01656** | **99.8%** | **0.0268** | **0.16431** | **0.9767** |

### 2. Held-Out Test Benchmark (2,885 Transitions, Unseen Obstacle Layouts)

| Model Variant | Parameters | Dynamics Prediction MSE | Collision Accuracy (%) | Relative Grounding Align | Action Tracking MSE | Kinematic Robustness |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original H-JEPA (Teacher/Target)** | 166,037 | **0.20183** | **87.8%** | **0.1174** | **0.25434** | **0.9918** |
| **Option A: Weight-Preserving** | 157,428 | **0.19254** | **87.2%** | **0.1174** | **0.25733** | **0.9921** |
| **Option B: Reconstructed Compact** | **39,804** | **0.01027** | **99.8%** | **0.1437** | **0.1574** | **0.978** |
