# Termal Kamera ile Kayıp Personel Aday Bölgesi Tespiti

## Current implementation status

This branch removes the pretrained YOLOX/COCO dependency and uses classical OpenCV operations only.

### Implemented in this branch
- Median denoising and CLAHE contrast enhancement.
- Otsu + percentile-based thermal foreground thresholding.
- Morphological closing/opening and contour extraction.
- Geometry and region-intensity descriptors for candidate ranking.
- Annotated candidate-region output.
- Explicit separation of raw pixel intensity (DN) from calibrated radiometric temperature (°C).
- Background comparison helper for a calibrated temperature matrix.
- Exploratory Welch-PSD analysis in a respiratory-frequency band, with strict non-medical wording.
- Image upload size limit and clearer API semantics.

### Important limitations (not yet satisfied)
1. Candidate regions are NOT confirmed people. Thresholding and geometry alone cannot reliably distinguish humans from hot objects, animals, reflections or terrain.
2. A normal JPEG/PNG/TIFF pixel value is not automatically a temperature. The API currently has no camera-specific radiometric decoder/calibration input; results are therefore labelled DN.
3. The current public analysis endpoint processes still images only. Video upload/storage is not equivalent to video analysis.
4. Respiratory signal analysis is a service function but is not yet connected to a video ROI-tracking pipeline or API.
5. No PostgreSQL persistence, analysis history, report export, dataset manifest, annotation workflow or benchmark evaluation pipeline is implemented yet.
6. Precision, recall, F1, false positives/negatives, processing time and temperature error have not been measured on a labelled, real-camera dataset.

## Safety and interpretation
This is a research/decision-support prototype, not a medical device or a validated search-and-rescue life detector. A positive respiratory-band periodic component is not proof of breathing or life. Failure to detect it is not evidence of death. Never use this output as the sole basis for rescue, triage or life/death decisions.

## Run
From `backend`, install `requirements.txt` and run:
```bash
uvicorn app.main:app --reload
```
From `frontend`, run:
```bash
npm install
npm run dev
```

API documentation: `http://localhost:8000/docs`

## Next engineering milestones
1. Add real radiometric camera/file support (e.g. camera-specific raw formats and documented calibration metadata).
2. Add video decoding, timestamp/FPS validation, ROI tracking and per-frame temperature extraction.
3. Build labelled real-thermal dataset manifests and train/test split by recording/session (not random adjacent frames).
4. Add reproducible benchmark scripts and confusion-matrix metrics.
5. Add PostgreSQL models/migrations, analysis history and PDF/CSV report export.
6. Add UI controls for ROI selection, thresholds, dataset provenance and quality warnings.
