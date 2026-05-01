from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TrackingSlotSchema:
    actor_slots: int = 1
    teammate_slots: int = 5
    opponent_slots: int = 6


@dataclass(frozen=True)
class TargetTiming:
    period_max_seconds: float = 1200.0
    default_horizon_seconds: float = 20.0
    penalty_horizon_seconds: float = 120.0


@dataclass(frozen=True)
class DataPrepConfig:
    sequence_disorder_threshold_seconds: float = -1.0
    sequence_disorder_warn_threshold_rows: int = 0
    boundary_tolerance_seconds: float = 20.0
    pass_merge_max_delta_events: int = 3
    required_phase1_event_types: tuple[str, ...] = ("reception", "penalty", "penaltydrawn")
    hard_fail_on_score_mismatch: bool = True
    normalize_end_of_period_to_whistle: bool = True
    phase2_zero_fill_value: float = 0.0
    phase2_max_distance_to_net_feet: float = 200.0
    phase2_max_distance_from_last_event_feet: float = 200.0
    phase2_max_speed_from_last_event_ft_per_sec: float = 120.0
    absolute_tracking_slots_per_side: int = 6
    enable_tracking_ghost_filter: bool = True
    phase3_embedding_model_name: str = "all-MiniLM-L6-v2"
    phase3_embedding_batch_size: int = 32
    phase3_embedding_normalize: bool = True
    target_timing: TargetTiming = field(default_factory=TargetTiming)
    slot_schema: TrackingSlotSchema = field(default_factory=TrackingSlotSchema)


@dataclass(frozen=True)
class DataPrepPaths:
    base_dir: Path
    run_label: str = "run_current"

    @property
    def data_dir(self) -> Path:
        return self.base_dir / "Data"

    @property
    def raw_data_dir(self) -> Path:
        return self.base_dir / "HALO Hackathon Data"

    @property
    def raw_events_path(self) -> Path:
        return self.raw_data_dir / "events.parquet"

    @property
    def raw_tracking_path(self) -> Path:
        return self.raw_data_dir / "tracking.parquet"

    @property
    def raw_stints_path(self) -> Path:
        return self.raw_data_dir / "stints.parquet"

    @property
    def raw_players_path(self) -> Path:
        return self.raw_data_dir / "players.parquet"

    @property
    def raw_games_path(self) -> Path:
        return self.raw_data_dir / "games.parquet"

    @property
    def tensor_ready_dir(self) -> Path:
        return self.data_dir / "Tensor-Ready Data"

    @property
    def final_datasets_dir(self) -> Path:
        return self.tensor_ready_dir / "Final Datasets"

    @property
    def runs_root_dir(self) -> Path:
        return self.data_dir / "Pipeline Runs" / self.run_label

    @property
    def phase1_dir(self) -> Path:
        return self.runs_root_dir / "phase1"

    @property
    def phase2_dir(self) -> Path:
        return self.runs_root_dir / "phase2"

    @property
    def phase3_dir(self) -> Path:
        return self.runs_root_dir / "phase3"

    @property
    def phase2_diagnostics_dir(self) -> Path:
        return self.phase2_dir / "diagnostics"

    @property
    def phase3_diagnostics_dir(self) -> Path:
        return self.phase3_dir / "diagnostics"

    @property
    def prep_logs_dir(self) -> Path:
        return self.runs_root_dir / "logs"

    @property
    def phase1_events_output(self) -> Path:
        return self.phase1_dir / "events_clean_phase1.parquet"

    @property
    def phase1_extra_goals_output(self) -> Path:
        return self.phase1_dir / "extra_goals_for_verification.csv"

    @property
    def phase1_score_report_output(self) -> Path:
        return self.prep_logs_dir / "phase1_score_verification_report.json"

    @property
    def phase2_events_output(self) -> Path:
        return self.phase2_dir / "events_phase2_enriched.parquet"

    @property
    def phase2_tracking_event_relative_output(self) -> Path:
        return self.phase2_dir / "tracking_event_relative.parquet"

    @property
    def phase2_tracking_absolute_output(self) -> Path:
        return self.phase2_dir / "tracking_absolute_pinned.parquet"

    @property
    def phase2_tracking_slot_mapping_output(self) -> Path:
        return self.phase2_dir / "tracking_slot_mapping.parquet"

    @property
    def phase2_tracking_stint_changes_output(self) -> Path:
        return self.phase2_dir / "tracking_stint_changes.parquet"

    @property
    def phase2_faceoff_reference_output(self) -> Path:
        return self.phase2_dir / "faceoff_reference.parquet"

    @property
    def phase2_penalty_reference_output(self) -> Path:
        return self.phase2_dir / "penalty_reference.parquet"

    @property
    def phase2_summary_output(self) -> Path:
        return self.prep_logs_dir / "phase2_summary.json"

    @property
    def phase3_events_with_embedding_indices_output(self) -> Path:
        return self.phase3_dir / "events_with_embedding_indices.parquet"

    @property
    def phase3_text_embeddings_output(self) -> Path:
        return self.phase3_dir / "text_embeddings.npy"

    @property
    def phase3_text_embedding_idx_map_output(self) -> Path:
        return self.phase3_dir / "text_embedding_idx_map.parquet"

    @property
    def phase3_tensor_ready_dataset_output(self) -> Path:
        return self.phase3_dir / "tensor_ready_dataset.parquet"

    @property
    def phase3_vocabularies_output(self) -> Path:
        return self.phase3_dir / "vocabularies.json"

    @property
    def phase3_feature_definitions_output(self) -> Path:
        return self.phase3_dir / "feature_definitions.json"

    @property
    def phase3_gnn_embeddings_output(self) -> Path:
        return self.phase3_dir / "gnn_embeddings.parquet"

    @property
    def phase3_gnn_embeddings_actor_emph_output(self) -> Path:
        return self.phase3_dir / "gnn_embeddings_actor_emph.parquet"

    @property
    def phase3_gnn_embeddings_actor_rel_output(self) -> Path:
        return self.phase3_dir / "gnn_embeddings_actor_rel.parquet"

    @property
    def phase3_gnn_base_feats_output(self) -> Path:
        return self.phase3_dir / "base_feats.npy"

    @property
    def phase3_gnn_base_adj_output(self) -> Path:
        return self.phase3_dir / "base_adj.npy"

    @property
    def phase3_gnn_base_mask_output(self) -> Path:
        return self.phase3_dir / "base_mask.npy"

    @property
    def phase3_gnn_actor_rel_feats_output(self) -> Path:
        return self.phase3_dir / "actor_rel_feats.npy"

    @property
    def phase3_gnn_actor_rel_adj_output(self) -> Path:
        return self.phase3_dir / "actor_rel_adj.npy"

    @property
    def phase3_gnn_actor_rel_mask_output(self) -> Path:
        return self.phase3_dir / "actor_rel_mask.npy"

    @property
    def phase3_gnn_actor_rel_ctx_feats_output(self) -> Path:
        return self.phase3_dir / "actor_rel_ctx_feats.npy"

    @property
    def phase3_gnn_actor_rel_ctx_adj_output(self) -> Path:
        return self.phase3_dir / "actor_rel_ctx_adj.npy"

    @property
    def phase3_gnn_actor_rel_ctx_mask_output(self) -> Path:
        return self.phase3_dir / "actor_rel_ctx_mask.npy"

    @property
    def phase3_threat_vectors_output(self) -> Path:
        return self.phase3_dir / "threat_vectors.npy"

    @property
    def phase3_threat_vector_scaler_output(self) -> Path:
        return self.phase3_dir / "threat_vector_scaler.json"

    @property
    def phase3_threat_vector_summary_output(self) -> Path:
        return self.prep_logs_dir / "phase3_threat_vectors_summary.json"

    @property
    def phase3_summary_output(self) -> Path:
        return self.prep_logs_dir / "phase3_summary.json"

    def ensure_dirs(self) -> None:
        for path in [
            self.data_dir,
            self.tensor_ready_dir,
            self.final_datasets_dir,
            self.runs_root_dir,
            self.phase1_dir,
            self.phase2_dir,
            self.phase3_dir,
            self.phase2_diagnostics_dir,
            self.phase3_diagnostics_dir,
            self.prep_logs_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)
