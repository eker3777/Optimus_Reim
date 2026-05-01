# Optimus Reim: Contextual xT for Hockey Sequence Valuation

Optimus Reim is a sequence-aware hockey valuation framework designed to measure two-way event impact beyond shot-only models.

The project goal is to move from terminal shot quality to continuous possession-state threat accounting over Sportlogiq micro-events.

## Public Snapshot Scope

This repository is a curated public push, not the full private working tree.

- Full public boundaries are defined in `PUBLISH_MANIFEST.md`.
- The published artifacts are focused on one aligned Phase 6 run.
- Raw/private/internal folders are intentionally excluded.

## Problem Framing

Traditional public hockey models are strongest at shot-level evaluation. They often under-credit transitional and defensive actions that materially change near-term scoring risk, including:

- passes and receptions,
- carries and puck protection,
- loose puck recoveries,
- interruption events that alter transition danger.

Optimus Reim addresses this by modeling actor-relative sequence state and attributing probability deltas at event level.

## Core Definitions

- Expected Threat (xT): probability of a goal within the model horizon conditioned on current sequence state.
- xT_For_Added: event-level change in the acting team's scoring threat.
- xT_Against_Added: event-level change in opponent counter-threat.
- Net_xT_Impact: `xT_For_Added - xT_Against_Added`.

This is a descriptive accounting framework rather than a pure shot-opportunity metric.

## Optimus Reim Transformer (published focus)

Models events as an ordered process with actor-relative three-class outputs:

- `p_actor_goal`
- `p_opp_goal`
- `p_no_goal`

Per-event probability deltas are converted into the xT ledger for player and team attribution.

Key implementation choices reflected in this repository:

- causal Transformer with ALiBi-style temporal bias,
- sliding windows (`max_len=128`, `stride=64`),
- token-level event attribution,
- fold-stable game-level cross-validation,
- sidecar restoration logic for counterpart events,
- restart handling after stoppages and boundaries.

## Key Report Insights

- xG vs xT distinction: xG estimates shot quality; xT captures contextual sequence execution.
- Non-shot value capture: transition-heavy actions can drive substantial aggregate impact.
- Execution asymmetry: similar tactical states can produce materially different marginal outcomes.
- Reliability: ranking/calibration behavior is stable enough to indicate transferable sequence structure.

## Included Public Artifacts

### Published run paths

- Models: `Models/Transformer_xT/run_20260325_013112/`
- Results: `Results/Transformer_xT/run_20260325_013112/`

### Included notebooks

- `Notebooks/phase1_data_cleaning.ipynb`
- `Notebooks/phase2_tensor_ready.ipynb`
- `Notebooks/phase3_final_datsets.ipynb`
- `Notebooks/phase6_inspection_simplified.ipynb`
- `Notebooks/phase6_modeling_consolidated.ipynb`
- `Notebooks/simplified_player_goalie_analysis.ipynb`

### Included code surface

- `scripts/data_prep/` for staged pipeline logic,
- `scripts/train_phase6_transformer.py` and entrypoint wrapper,
- `scripts/postprocess_phase6_outputs.py` and validation helpers,
- supporting ridge and GNN experimental scripts kept for scope continuity.

## Report Links

- Local report artifact: `Optimus-Reim-Report.pdf`
- External report URL: `https://019d40b7-44ff-0a16-e037-1e0bebdc2d87.share.connect.posit.cloud/`

## Repository Notes

- Large file handling follows `.gitattributes` policy.
- Publication boundaries and exclusions are authoritative in `PUBLISH_MANIFEST.md`.

