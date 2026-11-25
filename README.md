# Thermal Camera Search-and-Rescue Decision-Support Prototype

## Implemented in this development branch

- Classical OpenCV candidate extraction: denoising, local-contrast map, multi-level adaptive thresholding, morphology, contour/geometry filters, splitting of side-by-side blobs, merging of stacked fragments and rejection of UI graphics (colour bars, text).
- Candidate regions are explicitly not labelled as confirmed people.
- Raw pixel intensity (DN) is separated from calibrated radiometric temperature (°C).
- Still-image and video analysis endpoints.
- Heuristic video ROI association and exploratory Welch PSD signal analysis.
- PostgreSQL persistence for image and video analysis results.
- Analysis history and detail endpoints.
- Downloadable PDF report for every saved analysis.
- React interface for upload, history, record details and PDF downloads.
- No AI, machine-learning model, pretrained weights or model download is used.

## PostgreSQL setup

The repository includes a local PostgreSQL service in `docker-compose.yml`.

1. Start PostgreSQL from the repository root:
```bash
docker compose up -d postgres
```

2. Create the backend environment file:
```bash
cd backend
cp .env.example .env
```

3. Install backend requirements and start the API:
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

The example connection string targets the local Docker PostgreSQL service:
`postgresql+psycopg2://thermal_user:thermal_password@localhost:5432/thermal_camera`

On API startup, SQLAlchemy creates the `analysis_records` table if it does not exist. For this prototype, schema initialization uses `create_all`; a versioned Alembic migration workflow should be added before production deployment.

For quick local development without PostgreSQL, remove/rename `backend/.env` and the application falls back to a local SQLite database file.

## Analysis history and PDF endpoints

- `POST /api/v1/analyze` — analyze an image and persist its result.
- `POST /api/v1/analyze-video` — analyze a video and persist its result.
- `GET /api/v1/analyses?limit=50&offset=0` — list saved analyses.
- `GET /api/v1/analyses/{id}` — retrieve one analysis and its stored result data.
- `GET /api/v1/analyses/{id}/report.pdf` — download a PDF report.

PDF reports include analysis metadata, summary metrics, technical result data and interpretation/safety limitations. A Unicode font is used when a supported DejaVu Sans, Liberation Sans or Arial font is available on the host.

## Frontend

From `frontend`:
```bash
npm install
npm run dev
```

The frontend displays the analysis ID, refreshable analysis history, stored record details and PDF download links.

## Detector evaluation

Real precision/recall/F1 needs hand-labelled ground truth. Label your images with the bundled mouse-driven tool:

1. Open `tools/label_boxes.html` in a browser (double-click; no server, no internet needed).
2. Click **Klasör seç** and choose `data/samples` (or **Görüntü seç** for single files).
3. Drag a box tightly around every visible person. `E` marks an image as "no people", `Z` undoes, `Delete` removes the selected box, arrow keys switch images. Progress is autosaved in the browser.
4. Click **Dışa aktar** and save the downloaded `ground_truth.json` into `data/samples/`. Only reviewed images are exported.
5. Run the evaluation (add `--vis` to get result images: green = correct, red = false detection, yellow = missed person):

```bash
cd backend
python scripts/evaluate_detector.py --images ../data/samples --truth ../data/samples/ground_truth.json --vis ../data/eval_vis --out ../data/eval_result.json
```

Boxes are `[x, y, width, height]` in image pixels; a detection counts as correct when IoU >= 0.5 (one-to-one greedy matching). Ground-truth JSON may be loaded back into the tool with **Etiket yükle (JSON)** to continue labelling.

Informal check, 11 real thermal images (people counted by eye, NOT labelled ground truth, so these are not precision/recall values): the detector matched the counted number exactly on 3 images, was within one on 3 more, and was off by three or more on the remaining 5 (animals, close-up subjects, distant or dense crowds, a cluttered street scene with 12 candidates for 7 people). Mean time was about 0.12 s per image on a CPU.

## Important technical limitations

1. Candidate regions are not confirmed people. Thresholding and geometry cannot reliably distinguish humans from hot objects, animals, reflections or terrain.
2. Video ROI association is a simple nearest-centroid heuristic; camera motion, occlusion and crossing people can break track identity.
3. The current temporal signal is grayscale pixel intensity in DN, not calibrated temperature.
4. Respiratory-band periodicity is exploratory and unvalidated. A positive component is not proof of breathing or life; a negative result is not proof of death.
5. No camera-specific radiometric decoder/calibration adapter is connected to the API.
6. No labelled real-camera benchmark has been executed; precision, recall, F1, temperature error and respiratory detection performance are unknown. Only an informal visual check exists (see "Detector evaluation"). Animals are not separated from people, close-up subjects may be split into fragments and dense crowds of small subjects are under-counted.
7. PostgreSQL stores analysis metadata and result JSON. Uploaded source media files themselves are not archived in the database or object storage.

## Safety

This is a research prototype, not a medical device or validated search-and-rescue life detector. Never use its output as the sole basis for rescue, triage or life/death decisions. Human review and independent rescue procedures remain essential.

API documentation: `http://localhost:8000/docs`
