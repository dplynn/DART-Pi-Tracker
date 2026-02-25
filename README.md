# DART-Pi-Tracker

Live GTFS transit overlay for:
- Dallas, TX (DART): `http://www.dart.org/transitdata/latest/google_transit.zip`
- Hattiesburg, MS (HCT): `https://api.transloc.com/gtfs/hct.zip`

The app downloads both feeds, unzips/parses GTFS, and displays routes + moving vehicle dots on a grayscale map.

## Features

- Dallas/Hattiesburg city toggle (single-city view)
- Route overlays from GTFS shapes (with fallback geometry)
- Moving vehicles based on GTFS schedules
- Simulated fallback movement when no trips are active at the current time
- Route/vehicle color matching
- Vehicle-type filters (`bus`, `train`, `other`)
- Dallas train-line dropdown filter (per-line multi-select)
- Light and dark mode toggle (dark mode mutes line colors)
- Manual refresh endpoint for feed reload

## Tech Stack

- Backend: Flask + Pandas + Requests
- Frontend: Leaflet + vanilla JS/CSS
- Deployment: Gunicorn (Render-ready)

## Requirements

- Python 3.11+
- Internet access

## Local Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open: `http://127.0.0.1:5000`

## Data and Behavior

- Feeds are refreshed on startup.
- You can force refresh with `POST /api/refresh`.
- Vehicle positions are schedule-derived from static GTFS (not GTFS-RT).
- TRE is excluded from Dallas data.
- If there are no currently active trips, a service-window simulation is used so vehicles still animate.

## API Endpoints

- `GET /health`
- `GET /api/cities`
- `GET /api/routes?city=dallas|hattiesburg`
- `GET /api/vehicles?city=dallas|hattiesburg`
- `POST /api/refresh`

## Deploy on Render

This repo includes `render.yaml`.

1. Push this repo to GitHub.
2. In Render, create a new Web Service from the repo.
3. Render will use:
   - Build: `pip install -r requirements.txt`
   - Start: `gunicorn app:app`
4. Deploy.

The app binds to `0.0.0.0:$PORT` for hosted environments.
