# Optimus Reim: Contextual xT for Hockey Sequence Valuation

This repository contains the primary modeling stack for the HALO project, centered on two complementary objectives:

- **Phase 4**: establish a reliable comparative **xG baseline** with XGBoost.
- **Phase 6**: build **Optimus Reim**, a sequence-aware Transformer that assigns two-way event-level net impact (`Net_xT_Impact`).

The technical aim is to move from shot-terminal valuation to continuous sequence valuation using Sportlogiq micro-events.

---

## 1) Problem Framing

Traditional hockey public models are largely built on RTSS-style discrete events and are strongest at shot-level evaluation. They generally miss or under-credit the value of micro-transitions such as:

- passes and receptions,
- carries and puck protection,
- loose puck recoveries (LPRs),
- defensive interruption actions that alter transition risk.

Optimus Reim addresses this gap by modeling **actor-relative near-term threat** over event sequences and attributing probability deltas to the acting player.

---

## 2) Core Definitions

- **Expected Threat (xT)**: probability that a goal occurs within the model horizon, conditioned on current sequence state.
- **xT_For_Added**: event-level change in the acting team’s scoring threat.
- **xT_Against_Added**: event-level change in opponent counter-threat (negative is favorable defensive impact).
- **Net_xT_Impact**: `xT_For_Added - xT_Against_Added` (two-way ledger value per event).

This is a **descriptive accounting framework**, not a pure shot-opportunity metric.

---

## 3) Phase 4: XGBoost Comparative Baseline

Phase 4 provides a practical benchmark for shot quality using binary `target_xg` on shot/deflection events.

### Technical role
- Establishes reference performance against Sportlogiq shot xG fields.
- Quantifies incremental value of tracking-derived tabular features.
- Produces interpretable importance artifacts for model sanity checks.

### Why it stays in the stack
- Fast to train and compare.
- Strong baseline for calibration/performance-per-compute.
- Essential comparator for assessing whether sequence models provide actionable incremental value.

### Artifact locations
- Models: `Models/XGBoost_xG/`
- Results: `Results/XGBoost_xG/`

---

## 4) Phase 6: Optimus Reim Transformer (Event-Level Net Impact)

Phase 6 models hockey as an ordered event process with actor-relative 3-class outputs:

- `p_actor_goal`
- `p_opp_goal`
- `p_no_goal`

Per-event deltas are converted into the xT ledger (`xT_For_Added`, `xT_Against_Added`, `Net_xT_Impact`) for player/team attribution.

### 4.1 Input design
- Event metadata and sequence ordering features.
- Spatiotemporal geometry (coordinates, distance/angle dynamics, timing gaps, speed context).
- Game-state context (skaters, score state, net-empty flags).
- Text-semantic context via cleaned event description embeddings.

### 4.2 Sequence modeling choices
- Causal Transformer with ALiBi-style temporal biasing.
- Sliding sequence windows (`max_len=128`, `stride=64`).
- Token-level prediction head for event-wise attribution.
- 5-fold game-level CV with fold-stable rare-event ranking behavior.

### 4.3 Paired-event integrity: sidecar protocol
To avoid sequence hallucinations from dual-actor rows:

- Keep only primary timeline rows for training/scoring.
- Move loser/counterpart rows to sidecars (`faceoff_reference_df`, `penaltydrawn_reference_df`) with `kept_sl_event_id` linkage.
- Restore counterpart rows post-scoring with **inverse credit application**.
- Preserve provenance through `restored_source` and QC checks on restored event typing.

This preserves causal sequence integrity while retaining two-way event accounting.

### 4.4 State reset protocol after stoppages
Threat carry-over is explicitly controlled with restart logic:

- reset if prior event is terminal (`whistle`, `goal`, `period_end`),
- reset on hard-reset current events (`faceoff`, `period_start`),
- reset at structural boundaries (new game/sequence transitions).

Operationally, the prior aligned state is set to zero when `reset_prior_mask` is true, so post-whistle and faceoff events are evaluated from a clean baseline.

### Artifact locations
- Best weights: `Models/Transformer_xT/run_20260325_013112/`
- Outputs: `Results/Transformer_xT/run_20260325_013112/`

---

## 5) Key Technical Insights (from final report)

### xG vs xT behavior
- **xG** is predictive shot-quality estimation.
- **xT ledger** is contextual, execution-aware accounting over full sequences.
- This distinction is central: high-impact non-shot actions receive explicit credit in xT but not in shot-only frameworks.

### Shot execution asymmetry (context-aware)
- In elevated threat states, shot execution and miss outcomes produce materially different marginal effects.
- Missed-net outcomes can produce disproportionate transition-risk debits relative to on-net attempts.

### Possession-transition valuation
- High-volume transition events (especially LPRs) contribute substantial aggregate net impact.
- The framework surfaces players whose value is often underrepresented in box-score or shot-attempt-only summaries.

### Reliability characteristics
- Rare-event setting remains highly imbalanced by nature.
- Fold-level stability in ranking/calibration metrics supports that the model is learning transferable sequence structure rather than memorizing isolated games.

---

## 6) Scope Notes

- Visual leaderboards/figures are intentionally omitted from this README.
- Forward-looking implementation roadmap is intentionally omitted here.
- For full narrative, appendix details, and visual sections, see:
  - `Optimus-Reim-Report.pdf`
  - External report URL: `https://019d40b7-44ff-0a16-e037-1e0bebdc2d87.share.connect.posit.cloud/`

---

## 7) Repository Notes

- Large artifacts are managed with Git LFS (`.gitattributes`).
- This public snapshot is intentionally curated to include scripts, selected notebooks, and one aligned model/results run.
- Current publication package and scope boundaries are tracked in `PUBLISH_MANIFEST.md`.

