import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class Phase6SequenceDataset(Dataset):
    _shared_cache: Dict = {}
    _cache_schema_version: int = 1

    def __init__(
        self,
        df: pd.DataFrame,
        text_embeddings: np.ndarray,
        *,
        max_seq_length: int,
        window_stride: Optional[int],
        min_window_tokens: int,
        use_tracking: bool,
        debug_validate_items: bool,
        categorical_cols: List[str],
        continuous_cols: List[str],
        binary_cols: List[str],
        tracking_cols: List[str],
        game_ids: Optional[List] = None,
        sequences: Optional[List[Tuple]] = None,
        threat_vectors: Optional[np.ndarray] = None,
        threat_row_indexer: Optional[np.ndarray] = None,
    ):
        self.max_seq_length = int(max_seq_length)
        self.stride = int(window_stride) if (window_stride is not None and int(window_stride) > 0) else max(1, self.max_seq_length // 2)
        self.min_window_tokens = max(1, int(min_window_tokens))
        self.use_tracking = bool(use_tracking)
        self.debug_validate_items = bool(debug_validate_items)

        self.text_embeddings = text_embeddings.astype(np.float32, copy=False)

        self.categorical_cols = list(categorical_cols)
        self.continuous_cols = list(continuous_cols)
        self.binary_cols = list(binary_cols)
        self.tracking_cols = list(tracking_cols) if self.use_tracking else []

        self.threat_vectors = threat_vectors
        self.threat_row_indexer = threat_row_indexer

        if self.use_tracking:
            if self.threat_vectors is None or self.threat_row_indexer is None:
                raise RuntimeError(
                    "Threat vectors are required for with_tracking. "
                    "Run Phase 3 threat export to generate phase3/threat_vectors.npy."
                )
            if len(self.threat_row_indexer) != len(df):
                raise RuntimeError(
                    "Threat row indexer size does not match df_events. "
                    f"expected={len(df):,} got={len(self.threat_row_indexer):,}"
                )
            if self.threat_vectors.ndim != 2:
                raise RuntimeError(
                    f"Threat vectors must be 2D [N, F]. got={self.threat_vectors.shape}"
                )
            if int(self.threat_vectors.shape[1]) != 36:
                raise RuntimeError(
                    "Threat vectors must have 36 features. "
                    f"got={int(self.threat_vectors.shape[1])}"
                )

        cache_key = (
            id(df),
            tuple(self.categorical_cols),
            tuple(self.continuous_cols),
            tuple(self.binary_cols),
            tuple(self.tracking_cols),
            int(self.max_seq_length),
            int(self._cache_schema_version),
        )

        if cache_key not in self._shared_cache:
            game_indices = df.groupby(["game_id"], sort=False).indices
            period_time_series = df["period_time"] if "period_time" in df.columns else pd.Series(np.nan, index=df.index)
            period_time_sec = pd.to_numeric(
                df["period_time_sec"] if "period_time_sec" in df.columns else period_time_series,
                errors="coerce",
            )
            bad_period_time = period_time_sec.isna()
            if bad_period_time.any():
                mmss = period_time_series.astype(str).str.extract(r"^(\d{1,2}):(\d{2})$")
                parsed = pd.to_numeric(mmss[0], errors="coerce") * 60 + pd.to_numeric(mmss[1], errors="coerce")
                period_time_sec.loc[bad_period_time] = parsed.loc[bad_period_time]

            event_type_norm = (
                df["event_type"].astype(str).str.strip().str.lower().to_numpy(copy=False)
                if "event_type" in df.columns
                else np.full(len(df), "", dtype=object)
            )
            meta_cols = [
                c
                for c in [
                    "game_id",
                    "sl_event_id",
                    "game_event_id",
                    "period",
                    "period_time",
                    "sequence_id",
                    "sequence_event_id",
                    "team_id",
                    "player_id",
                    "event_type",
                    "outcome",
                    "detail",
                ]
                if c in df.columns
            ]
            meta_arrays = {col: df[col].to_numpy(copy=False) for col in meta_cols}
            payload = {
                "game_indices": game_indices,
                "cat": df[self.categorical_cols].to_numpy(copy=False).astype(np.int64)
                if self.categorical_cols
                else np.zeros((len(df), 0), dtype=np.int64),
                "cont": df[self.continuous_cols].to_numpy(copy=False).astype(np.float32)
                if self.continuous_cols
                else np.zeros((len(df), 0), dtype=np.float32),
                "bin": df[self.binary_cols].to_numpy(copy=False).astype(np.float32)
                if self.binary_cols
                else np.zeros((len(df), 0), dtype=np.float32),
                "track": df[self.tracking_cols].to_numpy(copy=False).astype(np.float32)
                if self.tracking_cols
                else np.zeros((len(df), 0), dtype=np.float32),
                "target": df["target"].to_numpy(copy=False).astype(np.int64),
                "text_idx": df["text_embedding_idx"].to_numpy(copy=False).astype(np.int64),
                "meta_cols": tuple(meta_cols),
                "meta_arrays": meta_arrays,
                "is_eos": (
                    df["is_eos"].to_numpy(copy=False).astype(np.int64)
                    if "is_eos" in df.columns
                    else np.zeros(len(df), dtype=np.int64)
                ),
                "period_time_sec": period_time_sec.to_numpy(copy=False).astype(np.float32),
                "event_type_norm": event_type_norm,
            }
            self._shared_cache[cache_key] = payload

        self.backend = self._shared_cache[cache_key]

        if game_ids is not None:
            game_keys = [g for g in list(game_ids) if g in self.backend["game_indices"]]
        elif sequences is not None:
            fallback_games = [seq[0] for seq in list(sequences) if len(seq) > 0]
            game_keys = [g for g in fallback_games if g in self.backend["game_indices"]]
        else:
            game_keys = list(self.backend["game_indices"].keys())

        self.samples: List[Dict] = []
        self.filtered_all_eos_games = 0
        self.split_on_large_gap_games = 0
        self.faceoff_gap_rows_dropped = 0
        self.max_gap_seconds = 10.0

        def _split_on_large_gaps(row_idx: np.ndarray) -> List[np.ndarray]:
            if len(row_idx) <= 1:
                return [row_idx]

            def _gap_breaks(idx: np.ndarray) -> np.ndarray:
                t = self.backend["period_time_sec"][idx]
                dt = np.diff(t)
                finite = np.isfinite(t)
                valid_dt = finite[:-1] & finite[1:]
                return np.where(valid_dt & (dt > self.max_gap_seconds))[0] + 1

            breaks = _gap_breaks(row_idx)

            if len(breaks) > 0:
                first_break = int(breaks[0])
                if first_break == 2 and len(row_idx) > 2:
                    row_idx = row_idx[2:]
                    self.faceoff_gap_rows_dropped += 2
                    if len(row_idx) <= 1:
                        return [row_idx] if len(row_idx) else []
                    breaks = _gap_breaks(row_idx)

            if len(breaks) == 0:
                return [row_idx]

            chunks = []
            start = 0
            for b in breaks:
                seg = row_idx[start:b]
                if len(seg) > 0:
                    chunks.append(seg)
                start = int(b)
            tail = row_idx[start:]
            if len(tail) > 0:
                chunks.append(tail)
            return chunks

        for game_key in game_keys:
            base_idx = np.asarray(self.backend["game_indices"][game_key], dtype=np.int64)
            eos = self.backend["is_eos"][base_idx]
            clean_idx = base_idx[eos == 0]

            if len(clean_idx) == 0:
                self.filtered_all_eos_games += 1
                continue

            game_segments = _split_on_large_gaps(clean_idx)
            if len(game_segments) > 1:
                self.split_on_large_gap_games += 1

            for seg_row_idx in game_segments:
                if len(seg_row_idx) < self.min_window_tokens:
                    continue

                if len(seg_row_idx) <= self.max_seq_length:
                    self.samples.append(
                        {
                            "game_key": game_key,
                            "row_idx": seg_row_idx,
                            "chunk_start": 0,
                            "chunk_index": 0,
                            "is_first_chunk": 1,
                            "drop_prefix_tokens": 0,
                        }
                    )
                else:
                    chunk_counter = 0
                    stop = len(seg_row_idx) - self.max_seq_length + self.stride
                    for start in range(0, max(stop, 1), self.stride):
                        chunk = seg_row_idx[start : start + self.max_seq_length]
                        if len(chunk) < self.min_window_tokens:
                            continue
                        is_first = 1 if start == 0 else 0
                        drop_prefix = 0 if is_first else min(self.stride, len(chunk))
                        self.samples.append(
                            {
                                "game_key": game_key,
                                "row_idx": chunk,
                                "chunk_start": int(start),
                                "chunk_index": int(chunk_counter),
                                "is_first_chunk": int(is_first),
                                "drop_prefix_tokens": int(drop_prefix),
                            }
                        )
                        chunk_counter += 1

        if len(self.samples) == 0:
            raise RuntimeError("No non-EOS windows available. Check is_eos filtering and source data quality.")

        self.filtered_all_eos_sequences = self.filtered_all_eos_games
        self.split_on_large_gap_sequences = self.split_on_large_gap_games

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        row_idx = np.asarray(sample["row_idx"], dtype=np.int64)

        eos_slice = self.backend["is_eos"][row_idx]
        if np.any(eos_slice != 0):
            raise RuntimeError("EOS token leakage detected in dataset sample. Expected all row_idx to be non-EOS.")

        seq_len = len(row_idx)
        max_len = self.max_seq_length

        cat = np.zeros((max_len, self.backend["cat"].shape[1]), dtype=np.int64)
        cont = np.zeros((max_len, self.backend["cont"].shape[1]), dtype=np.float32)
        binv = np.zeros((max_len, self.backend["bin"].shape[1]), dtype=np.float32)
        track = np.zeros((max_len, self.backend["track"].shape[1]), dtype=np.float32)
        text = np.zeros((max_len, self.text_embeddings.shape[1]), dtype=np.float32)
        threat_dim = int(self.threat_vectors.shape[1]) if self.threat_vectors is not None else 0
        threat_arr = np.zeros((max_len, threat_dim), dtype=np.float32)
        target = np.full((max_len,), -100, dtype=np.int64)
        valid_mask = np.zeros((max_len,), dtype=bool)

        if self.backend["cat"].shape[1] > 0:
            cat[:seq_len] = self.backend["cat"][row_idx]
        if self.backend["cont"].shape[1] > 0:
            cont[:seq_len] = self.backend["cont"][row_idx]
        if self.backend["bin"].shape[1] > 0:
            binv[:seq_len] = self.backend["bin"][row_idx]
        if self.backend["track"].shape[1] > 0:
            track[:seq_len] = self.backend["track"][row_idx]
        if self.threat_vectors is not None and self.threat_row_indexer is not None:
            threat_idx = self.threat_row_indexer[row_idx]
            if np.any(threat_idx < 0) or np.any(threat_idx >= len(self.threat_vectors)):
                raise RuntimeError("Threat index out of bounds in dataset window.")
            threat_arr[:seq_len] = self.threat_vectors[threat_idx]

        text_idx = self.backend["text_idx"][row_idx]
        if np.any(text_idx < 0) or np.any(text_idx >= len(self.text_embeddings)):
            raise RuntimeError("text_embedding_idx out of bounds in dataset window.")
        text[:seq_len] = self.text_embeddings[text_idx]
        target[:seq_len] = self.backend["target"][row_idx]
        valid_mask[:seq_len] = True

        if self.debug_validate_items:
            if (
                (self.backend["cont"].shape[1] > 0 and not np.isfinite(cont[:seq_len]).all())
                or (self.backend["bin"].shape[1] > 0 and not np.isfinite(binv[:seq_len]).all())
                or (self.backend["track"].shape[1] > 0 and not np.isfinite(track[:seq_len]).all())
                or (not np.isfinite(text[:seq_len]).all())
                or (self.threat_vectors is not None and not np.isfinite(threat_arr[:seq_len]).all())
            ):
                raise RuntimeError("Non-finite values detected in model inputs for a dataset sample.")

        meta_slice = {
            "meta_cols": self.backend["meta_cols"],
            "meta_arrays": {col: self.backend["meta_arrays"][col][row_idx] for col in self.backend["meta_cols"]},
            "chunk_start": int(sample["chunk_start"]),
            "chunk_index": int(sample["chunk_index"]),
            "is_first_chunk": int(sample["is_first_chunk"]),
            "drop_prefix_tokens": int(sample["drop_prefix_tokens"]),
            "seq_len": int(seq_len),
        }

        return {
            "categorical": torch.from_numpy(cat),
            "continuous": torch.from_numpy(cont),
            "binary": torch.from_numpy(binv),
            "tracking": torch.from_numpy(track),
            "text_emb": torch.from_numpy(text),
            "threat_vec": torch.from_numpy(threat_arr),
            "target": torch.from_numpy(target),
            "valid_mask": torch.from_numpy(valid_mask),
            "seq_len": torch.tensor(seq_len, dtype=torch.long),
            "meta": meta_slice,
        }


def phase6_collate_fn(batch):
    return {
        "categorical": torch.stack([b["categorical"] for b in batch]),
        "continuous": torch.stack([b["continuous"] for b in batch]),
        "binary": torch.stack([b["binary"] for b in batch]),
        "tracking": torch.stack([b["tracking"] for b in batch]),
        "text_emb": torch.stack([b["text_emb"] for b in batch]),
        "threat_vec": torch.stack([b["threat_vec"] for b in batch]),
        "target": torch.stack([b["target"] for b in batch]),
        "valid_mask": torch.stack([b["valid_mask"] for b in batch]),
        "seq_len": torch.stack([b["seq_len"] for b in batch]),
        "meta": [b["meta"] for b in batch],
    }


def phase6_seed_dataloader_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
