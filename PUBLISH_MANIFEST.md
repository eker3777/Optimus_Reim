# Publish Manifest (Curated Public Snapshot)

## Purpose
This manifest defines the exact scope of the public release for this repository snapshot.

## Included Top-Level Files
- `.gitattributes`
- `.gitignore`
- `README.md`
- `requirements.txt`
- `environment.yml`
- `Optimus-Reim-Report.pdf`

## Included Folders
- `scripts/`
- `docs/`
- `Notebooks/` (selected notebooks only)
- `Models/Transformer_xT/run_20260325_013112/`
- `Results/Transformer_xT/run_20260325_013112/`

## Selected Notebooks
- `Notebooks/phase1_data_cleaning.ipynb`
- `Notebooks/phase2_tensor_ready.ipynb`
- `Notebooks/phase3_final_datsets.ipynb`
- `Notebooks/phase6_inspection_simplified.ipynb`
- `Notebooks/phase6_modeling_consolidated.ipynb`
- `Notebooks/simplified_player_goalie_analysis.ipynb`

## Excluded Folders
- `Chats/`
- `Markdown/`
- `Data/`
- `HALO Hackathon Data/`
- `Tensor-Ready Data/`
- `TensorBoard/`
- `Presentation/`
- `Documents/`

## Excluded Policy Notes
- Non-selected notebooks are removed from the publish snapshot.
- Model and results artifacts are limited to the single aligned run listed above.
- Internal/scratch and private workflow artifacts are excluded.
