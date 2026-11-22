# Thermal Camera Search-and-Rescue Decision-Support Prototype

## Implemented in this development branch

- Classical OpenCV preprocessing: median denoising and CLAHE.
- Otsu/percentile thresholding, morphology and contour-based candidate extraction.
- Geometric/intensity descriptors; candidate regions are explicitly not confirmed people.
- Pixel intensity (DN) is kept separate from calibrated radiometric temperature (°C).
- Temperature statistics and background comparison helpers for a supplied calibrated temperature matrix.
- Still-image endpoint: `POST /api/v1/analyze`.
- Video endpoint: `POST /api/v1/analyze-video`.
- Video frame sampling, greedy nearest-centroid candidate association and temporal ROI intensity extraction.
- Welch power spectral density analysis in an exploratory respiratory-frequency band.
- Frontend image/video upload and result presentation.
- Unit tests for image conversion, candidate API semantics, temperature units and signal-analysis minimum-data behavior.

No AI, machine learning, pretrained model, neural network or model download is used.

## Video analysis pipeline

1. OpenCV decodes the uploaded video and reads its FPS/frame metadata.
2. Frames are sampled at up to 10 Hz.
3. Each sampled frame is processed with the same classical candidate detector as still images.
4. Candidate regions are associated between frames using nearest-centroid distance.
5. Mean grayscale intensity in each tracked ROI forms a temporal signal.
6. Tracks with less than 70% sample coverage are withheld from signal analysis.
7. The signal is detrended and Welch PSD is calculated. A dominant component in 0.10–0.70 Hz is reported as a periodic component compatible with that frequency band, not as proof of breathing.

## Important limitations

1. Candidate regions are not confirmed people. Classical thresholding/geometry cannot reliably distinguish humans from hot objects, animals, reflections or terrain.
2. Video ROI association is a simple nearest-centroid heuristic. Camera motion, occlusion, crossing people and changing scale can break identity continuity.
3. The current video signal is grayscale pixel intensity in DN, not calibrated temperature. Per-frame normalization/contrast processing can alter intensity; robust respiratory analysis requires stable radiometric temperature or a carefully controlled signal extraction pipeline.
4. The respiratory-band analysis is exploratory and unvalidated. A positive component does not prove breathing or life; a negative result does not prove absence of life.
5. No camera-specific radiometric decoder/calibration adapter is connected to the API.
6. No PostgreSQL persistence, analysis history, report export, dataset manifest, annotation workflow or benchmark pipeline is implemented yet.
7. Precision, recall, F1, false positives/negatives, processing time, temperature error and respiratory detection performance have not been measured on a labelled real-camera dataset.

## Safety

This is a research prototype, not a medical device or validated search-and-rescue life detector. Never use its output as the sole basis for rescue, triage or life/death decisions. Human review and independent rescue procedures remain essential.

## Run

Backend (from `backend`):
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Frontend (from `frontend`):
```bash
npm install
npm run dev
```

Run backend unit tests (from `backend`):
```bash
pytest -q
```

API documentation: `http://localhost:8000/docs`

## Next milestones

1. Select and document real thermal datasets; separate data suitable for detection, radiometry and respiratory validation.
2. Add camera-specific radiometric file decoding and calibration metadata.
3. Improve temporal tracking (motion compensation, optical flow or Kalman tracking; classical methods only).
4. Add per-frame calibrated temperature extraction and stable anatomical/ROI selection.
5. Build labelled-data evaluation: precision, recall, F1, false positives/negatives, processing time and signal-detection metrics.
6. Add PostgreSQL models/migrations, analysis history and PDF/CSV reports.
7. Add upload provenance, operator settings, ROI review and reproducible analysis parameters.
