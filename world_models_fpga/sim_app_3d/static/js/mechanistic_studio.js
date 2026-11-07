/**
 * Mechanistic Interpretability & 3D Linear Space Interactive Studio
 * Author: Brian A. Hadian
 * Project: Action-Conditioned World Models & Language Navigation on AMD Kria KR260 FPGA
 *
 * Provides:
 *  1. Interactive Layer-by-Layer Neural Architecture Graph with real-time neuron firing,
 *     signal pulses along causal synapses, circuit capability envelopes, and weight inspections.
 *  2. Interactive 3D/2D Linear Space Design Canvas with orbit, pan, zoom, hover raycasting,
 *     safety barrier hyperplane visualization, and end-to-end data transformation vectors.
 *  3. Dynamic cross-model rendering across all 6 architectures (Baseline, LeWM, H-JEPA,
 *     Weight-Preserving Option A, Reconstructed Compact Option B, Novel Composition).
 */

class MechanisticStudio {
    constructor() {
        // UI Elements
        this.graphCanvas = document.getElementById('nnGraphCanvas');
        this.graphCtx = this.graphCanvas ? this.graphCanvas.getContext('2d') : null;
        this.linearCanvas = document.getElementById('linearSpaceCanvas');
        this.linearCtx = this.linearCanvas ? this.linearCanvas.getContext('2d') : null;
        this.tooltip = document.getElementById('mechanisticTooltip');
        this.modelSelect = document.getElementById('modelSelector');

        // State
        this.currentModel = 'opt_b';
        this.architectureSpec = null;
        this.liveState = null;
        this.hoveredNeuron = null;
        this.hoveredPoint = null;
        this.selectedNeuron = null;
        this.showHyperplane = true;
        this.showVectors = true;
        this.is3DMode = true;

        // Neural Graph Camera (Pan/Zoom)
        this.graphCamera = {
            x: 40,
            y: 40,
            scale: 1.0,
            isDragging: false,
            startX: 0,
            startY: 0
        };

        // Linear Space Camera (3D Orbit / Pan / Zoom)
        this.linearCamera = {
            rotX: 0.45,   // Pitch
            rotY: -0.65,  // Yaw
            panX: 0.0,
            panY: 0.0,
            zoom: 180.0,  // Base projection scale
            isDragging: false,
            isPanning: false,
            startX: 0,
            startY: 0
        };

        // Synapse Particle Simulation
        this.particles = [];
        this.maxParticles = 60;

        // Layout Cache
        this.nodeLayouts = new Map(); // neuron_id -> {x, y, r, layer_idx, neuron}

        // Init
        this.setupEventHandlers();
        this.resizeCanvases();
        this.loadArchitecture(this.currentModel);
        this.startLiveTelemetryLoop();
        this.startAnimationLoop();
    }

    // =========================================================================
    // Initialization & Data Loading
    // =========================================================================

    async loadArchitecture(modelName) {
        try {
            const resp = await fetch(`/api/model_architecture?model=${encodeURIComponent(modelName)}`);
            if (!resp.ok) throw new Error(`HTTP error ${resp.status}`);
            this.architectureSpec = await resp.json();
            this.currentModel = this.architectureSpec.model_id;
            if (this.modelSelect) this.modelSelect.value = this.currentModel;

            this.computeGraphLayout();
            this.updateModelMetricsUI();
            console.log(`[MechanisticStudio] Loaded architecture: ${this.architectureSpec.display_name}`);
        } catch (err) {
            console.error('[MechanisticStudio] Failed to load architecture:', err);
        }
    }

    updateModelMetricsUI() {
        if (!this.architectureSpec) return;
        const s = this.architectureSpec;

        const setText = (id, txt) => {
            const el = document.getElementById(id);
            if (el) el.innerText = txt;
        };

        setText('metricModelName', s.display_name);
        setText('metricParams', `${s.total_params.toLocaleString()} params (${s.pruned_params_pct}% pruned)`);
        setText('metricLatency', `${s.latency_us.toFixed(1)} µs (${s.target_hardware})`);
        setText('metricEnergy', `${s.energy_mj} mJ / inference`);
        setText('metricModularity', `${(s.modularity_score * 100).toFixed(0)}% Disentangled`);
        setText('metricLinearManifold', s.linear_space.manifold_type);

        // Update circuit tags list
        const circuitList = document.getElementById('circuitLegend');
        if (circuitList && s.circuits) {
            circuitList.innerHTML = s.circuits.map(c => `
                <div class="circuit-chip" style="border-left: 4px solid ${c.color};">
                    <div class="chip-title">
                        <span class="chip-dot" style="background:${c.color}"></span>
                        <strong>${c.name}</strong> (${c.neurons} neurons)
                    </div>
                    <div class="chip-desc">${c.capability}</div>
                </div>
            `).join('');
        }
    }

    // =========================================================================
    // Layout Computation (High-DPI Aware)
    // =========================================================================

    computeGraphLayout() {
        if (!this.architectureSpec || !this.graphCanvas) return;
        this.nodeLayouts.clear();

        const w = this.graphCanvas.width / (window.devicePixelRatio || 1);
        const h = this.graphCanvas.height / (window.devicePixelRatio || 1);
        const layers = this.architectureSpec.layers;
        if (!layers || layers.length === 0) return;

        const colWidth = (w - 180) / Math.max(1, layers.length - 1);
        const startX = 90;

        layers.forEach((layer, layerIdx) => {
            const colX = startX + layerIdx * colWidth;
            const neurons = layer.neurons;
            const totalN = neurons.length;
            const maxVisible = Math.min(totalN, 24); // Cap max vertical nodes for crisp aesthetics
            const rowHeight = (h - 160) / Math.max(1, maxVisible - 1);
            const startY = 90;

            neurons.forEach((neuron, nIdx) => {
                if (nIdx < maxVisible) {
                    const nodeY = startY + nIdx * rowHeight;
                    const r = (layer.type === 'output' || layer.type === 'input') ? 8 : 6;
                    this.nodeLayouts.set(neuron.id, {
                        x: colX,
                        y: nodeY,
                        r: r,
                        layerIdx: layerIdx,
                        layerName: layer.name,
                        neuron: neuron,
                        circuit: neuron.circuit,
                        circuitName: neuron.circuit_name
                    });
                }
            });
        });
    }

    // =========================================================================
    // Live Telemetry Stream
    // =========================================================================

    startLiveTelemetryLoop() {
        const fetchTelemetry = async () => {
            try {
                const resp = await fetch('/api/interactive_model_state');
                if (resp.ok) {
                    this.liveState = await resp.json();
                    this.updateLiveHUD();
                    this.spawnSynapseParticles();
                }
            } catch (e) {
                // Ignore transient network ticks
            }
            setTimeout(fetchTelemetry, 65); // ~15 FPS sync
        };
        fetchTelemetry();
    }

    updateLiveHUD() {
        if (!this.liveState) return;
        const st = this.liveState;

        const setText = (id, txt) => {
            const el = document.getElementById(id);
            if (el) el.innerText = txt;
        };

        setText('hudLivePrompt', `"${st.prompt}"`);
        setText('hudLiveBehavior', st.active_behavior);
        setText('hudLivePose', `X: ${st.robot_pose[0]}m | Y: ${st.robot_pose[1]}m | θ: ${(st.robot_pose[2] * 180 / Math.PI).toFixed(1)}°`);
        setText('hudLiveAction', `vx: ${st.action[0]} | vy: ${st.action[1]} | wz: ${st.action[2]}`);
        setText('hudLiveBarrier', `${(st.col_risk * 100).toFixed(1)}%`);

        const barrierMeter = document.getElementById('hudBarrierMeter');
        if (barrierMeter) {
            barrierMeter.style.width = `${Math.min(100, st.col_risk * 100)}%`;
            barrierMeter.style.background = st.col_risk > 0.45 ? '#ff3366' : '#00ff88';
        }
    }

    // =========================================================================
    // Synapse Particle System
    // =========================================================================

    spawnSynapseParticles() {
        if (!this.architectureSpec || this.nodeLayouts.size === 0) return;
        if (this.particles.length >= this.maxParticles) return;

        const layers = this.architectureSpec.layers;
        if (!layers || layers.length < 2) return;

        // Pick an active connection between adjacent layers
        for (let l = 0; l < layers.length - 1; l++) {
            const l1 = layers[l];
            const l2 = layers[l + 1];
            if (Math.random() < 0.35) {
                const n1 = l1.neurons[Math.floor(Math.random() * Math.min(l1.neurons.length, 12))];
                const n2 = l2.neurons[Math.floor(Math.random() * Math.min(l2.neurons.length, 12))];
                const p1 = this.nodeLayouts.get(n1.id);
                const p2 = this.nodeLayouts.get(n2.id);
                if (p1 && p2) {
                    this.particles.push({
                        x1: p1.x,
                        y1: p1.y,
                        x2: p2.x,
                        y2: p2.y,
                        t: 0.0,
                        speed: 0.025 + Math.random() * 0.02,
                        color: this.getCircuitColor(n2.circuit)
                    });
                }
            }
        }
    }

    updateParticles() {
        for (let i = this.particles.length - 1; i >= 0; i--) {
            const p = this.particles[i];
            p.t += p.speed;
            if (p.t >= 1.0) {
                this.particles.splice(i, 1);
            }
        }
    }

    // =========================================================================
    // Canvas 1: Neural Circuit Firing Graph Rendering
    // =========================================================================

    renderNeuralGraph() {
        if (!this.graphCtx || !this.architectureSpec) return;
        const ctx = this.graphCtx;
        const w = this.graphCanvas.width / (window.devicePixelRatio || 1);
        const h = this.graphCanvas.height / (window.devicePixelRatio || 1);

        ctx.save();
        ctx.clearRect(0, 0, w, h);

        // Apply Camera Transform
        ctx.translate(this.graphCamera.x, this.graphCamera.y);
        ctx.scale(this.graphCamera.scale, this.graphCamera.scale);

        // Draw Layer Headers & Boundary Containers
        this.drawLayerEnvelopes(ctx);

        // Draw Synapses (Connections)
        this.drawSynapses(ctx);

        // Draw Traveling Signal Pulses
        this.drawParticles(ctx);

        // Draw Neurons with Dynamic Firing Glow
        this.drawNeurons(ctx);

        ctx.restore();
    }

    drawLayerEnvelopes(ctx) {
        if (!this.architectureSpec.layers) return;
        ctx.font = '12px "Inter", "Segoe UI", sans-serif';
        ctx.textAlign = 'center';

        this.architectureSpec.layers.forEach((layer, idx) => {
            const firstNode = this.nodeLayouts.get(layer.neurons[0]?.id);
            if (!firstNode) return;

            const colX = firstNode.x;
            ctx.fillStyle = '#a0aec0';
            ctx.fillText(layer.name.toUpperCase(), colX, 40);

            ctx.fillStyle = '#4a5568';
            ctx.font = '10px monospace';
            ctx.fillText(`${layer.neurons.length} Units`, colX, 56);
            ctx.font = '12px "Inter", "Segoe UI", sans-serif';
        });
    }

    drawSynapses(ctx) {
        if (!this.architectureSpec.layers) return;
        const layers = this.architectureSpec.layers;
        const isOptB = (this.currentModel === 'opt_b' || this.currentModel === 'novel_composition');

        for (let l = 0; l < layers.length - 1; l++) {
            const currLayer = layers[l];
            const nextLayer = layers[l + 1];

            nextLayer.neurons.forEach(targetNeuron => {
                const targetPos = this.nodeLayouts.get(targetNeuron.id);
                if (!targetPos) return;

                // Draw top weights
                (targetNeuron.top_weights || []).forEach(edge => {
                    const sourcePos = this.nodeLayouts.get(edge.from);
                    if (!sourcePos) return;

                    const isHovered = (this.hoveredNeuron && (this.hoveredNeuron.id === targetNeuron.id || this.hoveredNeuron.id === edge.from));
                    const isPruned = isOptB && Math.abs(edge.val) < 0.05;

                    ctx.beginPath();
                    ctx.moveTo(sourcePos.x, sourcePos.y);
                    const cpX = (sourcePos.x + targetPos.x) / 2;
                    ctx.bezierCurveTo(cpX, sourcePos.y, cpX, targetPos.y, targetPos.x, targetPos.y);

                    if (isHovered) {
                        ctx.strokeStyle = '#00ffff';
                        ctx.lineWidth = 2.5;
                        ctx.shadowColor = '#00ffff';
                        ctx.shadowBlur = 8;
                    } else if (isPruned) {
                        ctx.strokeStyle = 'rgba(60, 70, 90, 0.15)';
                        ctx.lineWidth = 0.5;
                        ctx.shadowBlur = 0;
                    } else {
                        const alpha = Math.min(0.65, Math.abs(edge.val) * 0.8 + 0.15);
                        ctx.strokeStyle = `rgba(100, 150, 220, ${alpha})`;
                        ctx.lineWidth = Math.min(2.0, Math.max(0.7, Math.abs(edge.val) * 1.5));
                        ctx.shadowBlur = 0;
                    }
                    ctx.stroke();
                    ctx.shadowBlur = 0;
                });
            });
        }
    }

    drawParticles(ctx) {
        this.particles.forEach(p => {
            const curX = p.x1 + (p.x2 - p.x1) * p.t;
            const curY = p.y1 + (p.y2 - p.y1) * p.t;

            ctx.beginPath();
            ctx.arc(curX, curY, 2.5, 0, Math.PI * 2);
            ctx.fillStyle = p.color;
            ctx.shadowColor = p.color;
            ctx.shadowBlur = 6;
            ctx.fill();
            ctx.shadowBlur = 0;
        });
    }

    drawNeurons(ctx) {
        this.nodeLayouts.forEach((layout, neuronId) => {
            const n = layout.neuron;
            const isHovered = this.hoveredNeuron && this.hoveredNeuron.id === neuronId;
            const baseColor = this.getCircuitColor(n.circuit);

            // Fetch live activation
            let actVal = 0.0;
            if (this.liveState && this.liveState.layer_activations) {
                const layerData = this.liveState.layer_activations[layout.layerName] || this.liveState.layer_activations[layout.layerIdx];
                if (layerData && layerData.post) {
                    actVal = layerData.post[n.index % layerData.post.length] || 0.0;
                }
            }

            // Radial Firing Glow
            const glowRadius = layout.r + (Math.abs(actVal) * 8.0);
            const grad = ctx.createRadialGradient(layout.x, layout.y, 1, layout.x, layout.y, glowRadius);
            grad.addColorStop(0, baseColor);
            grad.addColorStop(0.6, baseColor.replace('rgb', 'rgba').replace(')', ', 0.3)'));
            grad.addColorStop(1, 'transparent');

            ctx.beginPath();
            ctx.arc(layout.x, layout.y, glowRadius, 0, Math.PI * 2);
            ctx.fillStyle = grad;
            ctx.fill();

            // Core Node
            ctx.beginPath();
            ctx.arc(layout.x, layout.y, layout.r, 0, Math.PI * 2);
            ctx.fillStyle = isHovered ? '#ffffff' : (Math.abs(actVal) > 0.3 ? '#ffffff' : baseColor);
            ctx.strokeStyle = isHovered ? '#00ffff' : '#1a202c';
            ctx.lineWidth = isHovered ? 2.5 : 1.5;
            ctx.fill();
            ctx.stroke();

            // Label for output or hovered
            if (isHovered || layout.r >= 8) {
                ctx.fillStyle = isHovered ? '#00ffff' : '#e2e8f0';
                ctx.font = isHovered ? 'bold 11px sans-serif' : '10px monospace';
                ctx.textAlign = 'left';
                ctx.fillText(n.name, layout.x + layout.r + 6, layout.y + 3);
            }
        });
    }

    // =========================================================================
    // Canvas 2: Interactive 3D / 2D Linear Space Design
    // =========================================================================

    renderLinearSpace() {
        if (!this.linearCtx || !this.architectureSpec) return;
        const ctx = this.linearCtx;
        const w = this.linearCanvas.width / (window.devicePixelRatio || 1);
        const h = this.linearCanvas.height / (window.devicePixelRatio || 1);

        ctx.save();
        ctx.clearRect(0, 0, w, h);

        const cx = w / 2 + this.linearCamera.panX;
        const cy = h / 2 + this.linearCamera.panY;

        // 3D Perspective Projection Function
        const project = (x, y, z) => {
            const rx = this.linearCamera.rotX;
            const ry = this.linearCamera.rotY;

            // Yaw around Y
            const x1 = Math.cos(ry) * x + Math.sin(ry) * z;
            const z1 = -Math.sin(ry) * x + Math.cos(ry) * z;

            // Pitch around X
            const y2 = Math.cos(rx) * y - Math.sin(rx) * z1;
            const z2 = Math.sin(rx) * y + Math.cos(rx) * z1;

            const scale = this.linearCamera.zoom;
            return {
                x: cx + x1 * scale,
                y: cy - y2 * scale, // Inverted Y for screen space
                depth: z2
            };
        };

        // 1. Draw 3D Ground Grid
        this.drawLinearGrid(ctx, project);

        // 2. Draw 3D Coordinate Axes (Z1, Z2, Z3)
        this.drawLinearAxes(ctx, project);

        // 3. Draw Safety Barrier Hyperplane
        if (this.showHyperplane) {
            this.drawSafetyHyperplane(ctx, project);
        }

        // 4. Draw Candidate Action State Distributions
        this.drawLinearDistributions(ctx, project);

        // 5. Draw End-to-End Data Transformation Vector (on Hover)
        if (this.showVectors && this.hoveredNeuron) {
            this.drawTransformationVector(ctx, project);
        }

        ctx.restore();
    }

    drawLinearGrid(ctx, project) {
        ctx.strokeStyle = 'rgba(74, 85, 104, 0.25)';
        ctx.lineWidth = 1;

        const size = 2.0;
        const step = 0.5;

        for (let x = -size; x <= size; x += step) {
            const p1 = project(x, 0, -size);
            const p2 = project(x, 0, size);
            ctx.beginPath();
            ctx.moveTo(p1.x, p1.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.stroke();
        }

        for (let z = -size; z <= size; z += step) {
            const p1 = project(-size, 0, z);
            const p2 = project(size, 0, z);
            ctx.beginPath();
            ctx.moveTo(p1.x, p1.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.stroke();
        }
    }

    drawLinearAxes(ctx, project) {
        const axes = this.architectureSpec.linear_space.axes || ['Z1', 'Z2', 'Z3'];
        const o = project(0, 0, 0);

        const drawAxis = (p, label, color) => {
            ctx.beginPath();
            ctx.moveTo(o.x, o.y);
            ctx.lineTo(p.x, p.y);
            ctx.strokeStyle = color;
            ctx.lineWidth = 2.5;
            ctx.stroke();

            ctx.fillStyle = color;
            ctx.font = 'bold 11px sans-serif';
            ctx.fillText(label, p.x + 8, p.y + 4);
        };

        drawAxis(project(1.8, 0, 0), axes[0] || 'Z1 (Progress)', '#00d2ff');
        drawAxis(project(0, 1.8, 0), axes[1] || 'Z2 (Obstacle Clearance)', '#ff3366');
        drawAxis(project(0, 0, 1.8), axes[2] || 'Z3 (Heading Norm)', '#00ff88');
    }

    drawSafetyHyperplane(ctx, project) {
        // Hyperplane: w1*x + w2*y + w3*z + offset = 0
        const hp = this.architectureSpec.linear_space.hyperplane || { normal: [0.8, 0.6, -0.1], offset: 0.3 };
        const [nx, ny, nz] = hp.normal;
        const d = hp.offset;

        // Draw 4 corners of hyperplane quad
        const s = 1.4;
        const corners = [
            project(-s, (-d - nx * (-s) - nz * (-s)) / (ny || 1), -s),
            project(s, (-d - nx * s - nz * (-s)) / (ny || 1), -s),
            project(s, (-d - nx * s - nz * s) / (ny || 1), s),
            project(-s, (-d - nx * (-s) - nz * s) / (ny || 1), s)
        ];

        ctx.beginPath();
        ctx.moveTo(corners[0].x, corners[0].y);
        ctx.lineTo(corners[1].x, corners[1].y);
        ctx.lineTo(corners[2].x, corners[2].y);
        ctx.lineTo(corners[3].x, corners[3].y);
        ctx.closePath();

        // Shaded Barrier Surface
        ctx.fillStyle = 'rgba(255, 51, 102, 0.18)';
        ctx.fill();
        ctx.strokeStyle = 'rgba(255, 51, 102, 0.85)';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // Label on Hyperplane
        const center = project(0, -d / (ny || 1), 0);
        ctx.fillStyle = '#ff3366';
        ctx.font = 'bold 10px monospace';
        ctx.fillText('SAFETY BARRIER HYPERPLANE B(s,a) = 0.5', center.x - 70, center.y - 10);
    }

    drawLinearDistributions(ctx, project) {
        if (!this.liveState || !this.liveState.linear_space_state) return;
        const pts = this.liveState.linear_space_state.candidate_points || [];

        // Draw Candidate Action Scatter Cloud
        pts.forEach((pt, idx) => {
            const p = project(pt.x, pt.y, pt.z);
            const isCol = pt.is_collision;

            ctx.beginPath();
            ctx.arc(p.x, p.y, isCol ? 4.5 : 3.5, 0, Math.PI * 2);
            ctx.fillStyle = isCol ? '#ff3366' : '#00ff88';
            ctx.shadowColor = ctx.fillStyle;
            ctx.shadowBlur = 4;
            ctx.fill();
            ctx.shadowBlur = 0;
        });

        // Current Active State Diamond
        const curZ = this.liveState.linear_space_state.current_z || [0, 0, 0];
        const curP = project(curZ[0], curZ[1], curZ[2]);

        ctx.save();
        ctx.translate(curP.x, curP.y);
        ctx.rotate(Math.PI / 4);
        ctx.fillStyle = '#00ffff';
        ctx.shadowColor = '#00ffff';
        ctx.shadowBlur = 12;
        ctx.fillRect(-7, -7, 14, 14);
        ctx.restore();

        ctx.fillStyle = '#00ffff';
        ctx.font = 'bold 11px sans-serif';
        ctx.fillText('Z_curr (Robot Active State)', curP.x + 14, curP.y + 4);
    }

    drawTransformationVector(ctx, project) {
        if (!this.liveState || !this.hoveredNeuron) return;
        const tInfo = (this.liveState.transformations || {})[this.hoveredNeuron.id];
        if (!tInfo) return;

        const curZ = this.liveState.linear_space_state.current_z || [0, 0, 0];
        const v = tInfo.vector;
        const p1 = project(curZ[0], curZ[1], curZ[2]);
        const p2 = project(curZ[0] + v[0] * 0.8, curZ[1] + v[1] * 0.8, curZ[2] + v[2] * 0.8);

        // Vector Arrow
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.strokeStyle = '#ffd700';
        ctx.lineWidth = 3.0;
        ctx.shadowColor = '#ffd700';
        ctx.shadowBlur = 10;
        ctx.stroke();
        ctx.shadowBlur = 0;

        // Arrow Head
        ctx.beginPath();
        ctx.arc(p2.x, p2.y, 5, 0, Math.PI * 2);
        ctx.fillStyle = '#ffd700';
        ctx.fill();

        ctx.fillStyle = '#ffd700';
        ctx.font = 'bold 11px monospace';
        ctx.fillText(`Δy: [${v.join(', ')}]`, p2.x + 8, p2.y - 4);
    }

    // =========================================================================
    // Hover Tooltip HUD
    // =========================================================================

    showNeuronTooltip(e, neuronInfo) {
        if (!this.tooltip) return;
        const n = neuronInfo.neuron;
        const circColor = this.getCircuitColor(n.circuit);

        // Fetch Live Values
        let actVal = 0.0;
        let preVal = 0.0;
        if (this.liveState && this.liveState.layer_activations) {
            const lData = this.liveState.layer_activations[neuronInfo.layerName] || this.liveState.layer_activations[neuronInfo.layerIdx];
            if (lData) {
                const idx = n.index || 0;
                postVal = lData.post ? lData.post[idx % lData.post.length] : 0.0;
                preVal = lData.pre ? lData.pre[idx % lData.pre.length] : 0.0;
            }
        }

        const tInfo = (this.liveState?.transformations || {})[n.id] || { vector: [0, 0, 0], interpretation: 'Linear transform: 0' };

        this.tooltip.innerHTML = `
            <div class="tooltip-header" style="border-left: 4px solid ${circColor};">
                <div class="tooltip-title">${n.name}</div>
                <div class="tooltip-subtitle">${neuronInfo.layerName}</div>
            </div>
            <div class="tooltip-body">
                <div class="tooltip-row">
                    <span class="label">Circuit:</span>
                    <span class="val" style="color:${circColor}">${n.circuit_name}</span>
                </div>
                <div class="tooltip-row">
                    <span class="label">Capability:</span>
                    <span class="val">${n.capability}</span>
                </div>
                <div class="tooltip-row">
                    <span class="label">Principle:</span>
                    <span class="val" style="color:#00ff88">${n.principle}</span>
                </div>
                <div class="tooltip-divider"></div>
                <div class="tooltip-row">
                    <span class="label">Pre-Activation (z):</span>
                    <span class="val monospace">${preVal >= 0 ? '+' : ''}${preVal.toFixed(3)}</span>
                </div>
                <div class="tooltip-row">
                    <span class="label">Post-Activation (a):</span>
                    <span class="val monospace" style="color:#00ffff">${postVal >= 0 ? '+' : ''}${postVal.toFixed(3)}</span>
                </div>
                <div class="tooltip-row">
                    <span class="label">Weight Norm ||W||:</span>
                    <span class="val monospace">${(n.weight_norm || 1.0).toFixed(3)}</span>
                </div>
                <div class="tooltip-divider"></div>
                <div class="tooltip-section-title">End-to-End Linear Transformation:</div>
                <div class="tooltip-transform-box">
                    <code>Δy_linear = [${tInfo.vector.join(', ')}]</code>
                    <div class="transform-desc">${tInfo.interpretation}</div>
                </div>
            </div>
        `;

        this.tooltip.style.display = 'block';
        const rect = this.graphCanvas.getBoundingClientRect();
        this.tooltip.style.left = `${Math.min(window.innerWidth - 340, rect.left + e.clientX - rect.left + 18)}px`;
        this.tooltip.style.top = `${Math.min(window.innerHeight - 300, rect.top + e.clientY - rect.top + 18)}px`;
    }

    hideNeuronTooltip() {
        if (this.tooltip) this.tooltip.style.display = 'none';
    }

    // =========================================================================
    // Event Handlers & Interactivity (Pan, Zoom, Hover)
    // =========================================================================

    setupEventHandlers() {
        // Window Resize
        window.addEventListener('resize', () => {
            this.resizeCanvases();
            this.computeGraphLayout();
        });

        // Model Selector
        if (this.modelSelect) {
            this.modelSelect.addEventListener('change', async (e) => {
                await this.loadArchitecture(e.target.value);
                // Also notify backend simulation controller
                await fetch('/api/model', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model: e.target.value })
                });
            });
        }

        // Setup Graph Canvas Controls
        if (this.graphCanvas) {
            this.graphCanvas.addEventListener('mousedown', (e) => {
                this.graphCamera.isDragging = true;
                this.graphCamera.startX = e.clientX - this.graphCamera.x;
                this.graphCamera.startY = e.clientY - this.graphCamera.y;
            });

            window.addEventListener('mouseup', () => {
                this.graphCamera.isDragging = false;
                this.linearCamera.isDragging = false;
                this.linearCamera.isPanning = false;
            });

            this.graphCanvas.addEventListener('mousemove', (e) => {
                if (this.graphCamera.isDragging) {
                    this.graphCamera.x = e.clientX - this.graphCamera.startX;
                    this.graphCamera.y = e.clientY - this.graphCamera.startY;
                } else {
                    this.handleGraphHover(e);
                }
            });

            this.graphCanvas.addEventListener('wheel', (e) => {
                e.preventDefault();
                const zoomFactor = e.deltaY < 0 ? 1.08 : 0.92;
                this.graphCamera.scale = Math.min(3.0, Math.max(0.4, this.graphCamera.scale * zoomFactor));
            });

            this.graphCanvas.addEventListener('mouseleave', () => {
                this.hoveredNeuron = null;
                this.hideNeuronTooltip();
            });
        }

        // Setup Linear Space Canvas Controls
        if (this.linearCanvas) {
            this.linearCanvas.addEventListener('mousedown', (e) => {
                if (e.shiftKey || e.button === 1) {
                    this.linearCamera.isPanning = true;
                } else {
                    this.linearCamera.isDragging = true;
                }
                this.linearCamera.startX = e.clientX;
                this.linearCamera.startY = e.clientY;
            });

            this.linearCanvas.addEventListener('contextmenu', (e) => e.preventDefault());

            this.linearCanvas.addEventListener('mousemove', (e) => {
                const dx = e.clientX - this.linearCamera.startX;
                const dy = e.clientY - this.linearCamera.startY;

                if (this.linearCamera.isPanning) {
                    this.linearCamera.panX += dx;
                    this.linearCamera.panY += dy;
                    this.linearCamera.startX = e.clientX;
                    this.linearCamera.startY = e.clientY;
                } else if (this.linearCamera.isDragging) {
                    this.linearCamera.rotY += dx * 0.008;
                    this.linearCamera.rotX = Math.min(1.4, Math.max(-1.4, this.linearCamera.rotX + dy * 0.008));
                    this.linearCamera.startX = e.clientX;
                    this.linearCamera.startY = e.clientY;
                }
            });

            this.linearCanvas.addEventListener('wheel', (e) => {
                e.preventDefault();
                const zoomFactor = e.deltaY < 0 ? 1.10 : 0.90;
                this.linearCamera.zoom = Math.min(600.0, Math.max(60.0, this.linearCamera.zoom * zoomFactor));
            });
        }

        // Button Controls
        const btnResetGraph = document.getElementById('btnResetGraph');
        if (btnResetGraph) btnResetGraph.onclick = () => { this.graphCamera.x = 40; this.graphCamera.y = 40; this.graphCamera.scale = 1.0; };

        const btnResetLinear = document.getElementById('btnResetLinear');
        if (btnResetLinear) btnResetLinear.onclick = () => { this.linearCamera.rotX = 0.45; this.linearCamera.rotY = -0.65; this.linearCamera.panX = 0; this.linearCamera.panY = 0; this.linearCamera.zoom = 180.0; };

        const btnToggleHyperplane = document.getElementById('btnToggleHyperplane');
        if (btnToggleHyperplane) btnToggleHyperplane.onclick = () => { this.showHyperplane = !this.showHyperplane; };

        const btnToggleVectors = document.getElementById('btnToggleVectors');
        if (btnToggleVectors) btnToggleVectors.onclick = () => { this.showVectors = !this.showVectors; };
    }

    handleGraphHover(e) {
        const rect = this.graphCanvas.getBoundingClientRect();
        const mx = (e.clientX - rect.left - this.graphCamera.x) / this.graphCamera.scale;
        const my = (e.clientY - rect.top - this.graphCamera.y) / this.graphCamera.scale;

        let found = null;
        for (const [id, layout] of this.nodeLayouts.entries()) {
            const dist = Math.hypot(mx - layout.x, my - layout.y);
            if (dist <= layout.r + 5) {
                found = layout;
                break;
            }
        }

        if (found) {
            this.hoveredNeuron = found.neuron;
            this.showNeuronTooltip(e, found);
        } else {
            this.hoveredNeuron = null;
            this.hideNeuronTooltip();
        }
    }

    resizeCanvases() {
        const dpr = window.devicePixelRatio || 1;
        const setupCanvas = (c) => {
            if (!c) return;
            const w = c.clientWidth;
            const h = c.clientHeight;
            if (c.width !== w * dpr || c.height !== h * dpr) {
                c.width = w * dpr;
                c.height = h * dpr;
                const ctx = c.getContext('2d');
                ctx.scale(dpr, dpr);
            }
        };
        setupCanvas(this.graphCanvas);
        setupCanvas(this.linearCanvas);
    }

    getCircuitColor(circuitId) {
        const colors = {
            1: '#00d2ff', // Heading Attractor
            2: '#ff3366', // Safety Barrier
            3: '#00ff88', // Macro Horizon
            4: '#b366ff', // Appearance Filter
            5: '#ffb700', // Motor Actuation
            6: '#e056fd', // Dynamic Sequencer
            0: '#718096'  // Entangled Non-Causal
        };
        return colors[circuitId] || '#a0aec0';
    }

    // =========================================================================
    // Master Animation Loop (60 FPS WebGL / Canvas)
    // =========================================================================

    startAnimationLoop() {
        const loop = () => {
            this.updateParticles();
            this.renderNeuralGraph();
            this.renderLinearSpace();
            requestAnimationFrame(loop);
        };
        requestAnimationFrame(loop);
    }
}

// Global initialization
window.addEventListener('DOMContentLoaded', () => {
    window.studio = new MechanisticStudio();
});
