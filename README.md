# Multimodal Deepfake Detection (Audio + Video)

Minor Project (7th semester) — Dept. of CSE

Goal: detect audio-video deepfakes from the FakeAVCeleb dataset using a
two-branch network — one branch looks at face frames (EfficientNet-B0),
the other listens to the audio track (log-mel spectrogram + small CNN).
The two feature vectors are fused to give (a) a real/fake verdict and
(b) which modality was faked.

Dataset: FakeAVCeleb v1.2 (500 real + 19,500 fake videos).
Put the extracted dataset under `data/FakeAVCeleb_v1.2/` (not tracked by git).

## Progress

- [x] literature survey (17 papers)
- [x] dataset downloaded + category statistics
- [ ] data loader (started — `src/data_loader.py`)
- [ ] two-branch model (started — `src/model.py`)
- [ ] training / evaluation (`src/train.py`)
- [ ] results (accuracy / F1 / AUC)

## Setup

```bash
pip install -r requirements.txt
```

Nothing runs end-to-end yet — the training entry point will be
`python src/train.py` once the pipeline is wired up.
