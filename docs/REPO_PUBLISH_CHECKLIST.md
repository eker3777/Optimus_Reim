# Repo Publish Checklist (Private)

## Pre-Push

- [ ] `git lfs install`
- [ ] Confirm `.gitattributes` is committed
- [ ] Confirm `.gitignore` is committed
- [ ] Ensure selected notebooks and artifacts match `PUBLISH_MANIFEST.md`
- [ ] Remove or archive non-core notebooks from first publish scope

## Large File Validation

- [ ] `git lfs ls-files` lists parquet/model binaries
- [ ] No oversized binaries staged outside LFS
- [ ] `TensorBoard/` is not staged

## Notebook Validation

- [ ] Core notebooks open without missing-file path errors
- [ ] Run-order documented in README and manifest
- [ ] Path bootstrap is repo-relative (no machine-specific absolute roots in setup cells)

## Results Validation

- [ ] Event-level exports present for Transformer xG and Optimus Reim
- [ ] Player/team aggregate outputs present
- [ ] Summary JSON/CSV artifacts present

## Suggested Initial Push Commands

```powershell
git init
git lfs install
git add .gitattributes .gitignore README.md PUBLISH_MANIFEST.md docs/REPO_PUBLISH_CHECKLIST.md requirements.txt environment.yml
git add Notebooks Results Models "Processed Sequence Data" "Tensor-Ready Data" "HALO Hackathon Data"
git commit -m "Initial private publication package: phases 1-4, 6, 7 with data/results"
git remote add origin <private-repo-url>
git push -u origin main
```
