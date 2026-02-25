from __future__ import annotations

import math
import os
import re
import shutil
import threading
import time
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dateutil import tz
from flask import Flask, jsonify, request, send_from_directory


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static"

CITY_CONFIG: dict[str, dict[str, Any]] = {
    "dallas": {
        "name": "Dallas (DART)",
        "url": "http://www.dart.org/transitdata/latest/google_transit.zip",
        "center": [32.7767, -96.7970],
        "zoom": 11,
    },
    "hattiesburg": {
        "name": "Hattiesburg (HCT)",
        "url": "https://api.transloc.com/gtfs/hct.zip",
        "center": [31.3271, -89.2903],
        "zoom": 12,
    },
}

WEEKDAY_COLS = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
]

TRE_PATTERN = re.compile(r"\bTRE\b|TRINITY\s+RAILWAY\s+EXPRESS", re.IGNORECASE)
TRAIN_ROUTE_TYPES = {"0", "1", "2", "5", "6", "7", "11", "12"}
BUS_ROUTE_TYPES = {"3", "700", "701", "702", "704", "705", "706", "707", "708", "709", "710", "711", "712", "713", "714", "715", "716", "800"}


app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")
cache_lock = threading.Lock()
data_cache: dict[str, dict[str, Any]] = {}
initialized = False
loading_cities: set[str] = set()
loading_lock = threading.Lock()


def parse_gtfs_time_seconds(value: str | float | int | None) -> int | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    parts = text.split(":")
    if len(parts) != 3:
        return None
    try:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2])
    except ValueError:
        return None
    return (hours * 3600) + (minutes * 60) + seconds


def safe_read_csv(
    gtfs_dir: Path, filename: str, required: bool = True, usecols: list[str] | None = None
) -> pd.DataFrame:
    path = gtfs_dir / filename
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing required GTFS file: {filename}")
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, usecols=usecols, low_memory=False)


def parse_gtfs_time_series(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str).str.strip()
    valid = text.str.match(r"^\d{1,3}:\d{1,2}:\d{1,2}$")
    parts = text.where(valid).str.split(":", expand=True)
    if parts is None or parts.shape[1] != 3:
        return pd.to_numeric(pd.Series([None] * len(series)), errors="coerce")
    h = pd.to_numeric(parts[0], errors="coerce")
    m = pd.to_numeric(parts[1], errors="coerce")
    s = pd.to_numeric(parts[2], errors="coerce")
    return (h * 3600) + (m * 60) + s


def download_and_extract(city_id: str, url: str) -> Path:
    city_dir = DATA_DIR / city_id
    zip_path = city_dir / "feed.zip"
    city_dir.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, timeout=60)
    response.raise_for_status()
    zip_path.write_bytes(response.content)

    stamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    gtfs_dir = city_dir / f"gtfs_{stamp}"
    gtfs_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(gtfs_dir)

    # Best-effort cleanup of old extracted folders on Windows/OneDrive setups.
    for old in city_dir.glob("gtfs_*"):
        if old == gtfs_dir:
            continue
        try:
            shutil.rmtree(old)
        except Exception:
            pass

    return gtfs_dir


def get_timezone(agency_df: pd.DataFrame) -> tz.tzfile | None:
    if agency_df.empty or "agency_timezone" not in agency_df.columns:
        return None
    zone = agency_df["agency_timezone"].dropna().astype(str)
    if zone.empty:
        return None
    return tz.gettz(zone.iloc[0])


def service_ids_for_date(calendar_df: pd.DataFrame, calendar_dates_df: pd.DataFrame, service_day: date) -> set[str]:
    active: set[str] = set()
    ymd = service_day.strftime("%Y%m%d")
    weekday_col = WEEKDAY_COLS[service_day.weekday()]

    if not calendar_df.empty:
        calendar = calendar_df.copy()
        calendar = calendar.fillna("")
        mask = (
            (calendar.get("start_date", "") <= ymd)
            & (calendar.get("end_date", "") >= ymd)
            & (calendar.get(weekday_col, "0") == "1")
        )
        if "service_id" in calendar.columns:
            active.update(calendar.loc[mask, "service_id"].astype(str).tolist())

    if not calendar_dates_df.empty and "service_id" in calendar_dates_df.columns:
        cdx = calendar_dates_df.copy().fillna("")
        rows = cdx[cdx.get("date", "") == ymd]
        for _, row in rows.iterrows():
            sid = str(row.get("service_id", "")).strip()
            ex_type = str(row.get("exception_type", "")).strip()
            if not sid:
                continue
            if ex_type == "1":
                active.add(sid)
            elif ex_type == "2":
                active.discard(sid)

    return active


def build_route_features(
    routes_df: pd.DataFrame,
    trips_df: pd.DataFrame,
    shapes_df: pd.DataFrame,
    stop_times_df: pd.DataFrame,
    stops_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    route_meta = {}
    if not routes_df.empty:
        for _, row in routes_df.fillna("").iterrows():
            rid = str(row.get("route_id", ""))
            if not rid:
                continue
            route_type = str(row.get("route_type", "")).strip()
            route_meta[rid] = {
                "route_short_name": str(row.get("route_short_name", "")),
                "route_long_name": str(row.get("route_long_name", "")),
                "route_color": str(row.get("route_color", "")),
                "route_type": route_type,
                "vehicle_type": vehicle_type_from_route_type(route_type),
                "is_tre": bool(
                    TRE_PATTERN.search(
                        " ".join(
                            [
                                str(row.get("route_short_name", "")),
                                str(row.get("route_long_name", "")),
                                str(row.get("route_desc", "")),
                            ]
                        )
                    )
                ),
            }

    features: list[dict[str, Any]] = []
    if not shapes_df.empty and not trips_df.empty:
        shapes = shapes_df.copy().fillna("")
        for col in ("shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"):
            if col not in shapes.columns:
                shapes[col] = ""
        shapes["shape_pt_lat"] = pd.to_numeric(shapes["shape_pt_lat"], errors="coerce")
        shapes["shape_pt_lon"] = pd.to_numeric(shapes["shape_pt_lon"], errors="coerce")
        shapes["shape_pt_sequence"] = pd.to_numeric(shapes["shape_pt_sequence"], errors="coerce")
        shapes = shapes.dropna(subset=["shape_pt_lat", "shape_pt_lon", "shape_id", "shape_pt_sequence"])

        trip_shape_map = (
            trips_df[["route_id", "shape_id"]]
            .dropna()
            .drop_duplicates(subset=["route_id", "shape_id"])
            .astype(str)
        )
        shape_route_map = dict(zip(trip_shape_map["shape_id"], trip_shape_map["route_id"]))

        for shape_id, grp in shapes.sort_values("shape_pt_sequence").groupby("shape_id"):
            coords = [[float(lon), float(lat)] for lat, lon in zip(grp["shape_pt_lat"], grp["shape_pt_lon"])]
            if len(coords) < 2:
                continue
            route_id = shape_route_map.get(str(shape_id), "")
            meta = route_meta.get(route_id, {})
            props = {"route_id": route_id, **meta}
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": coords},
                    "properties": props,
                }
            )

    if features:
        return features

    # Fallback when shapes.txt is incomplete: derive linework from trip stop order.
    if stop_times_df.empty or stops_df.empty or trips_df.empty:
        return features

    stops = stops_df.copy().fillna("")
    stops["stop_lat"] = pd.to_numeric(stops.get("stop_lat"), errors="coerce")
    stops["stop_lon"] = pd.to_numeric(stops.get("stop_lon"), errors="coerce")
    stops = stops.dropna(subset=["stop_id", "stop_lat", "stop_lon"])
    st = stop_times_df.copy().fillna("")
    st["stop_sequence"] = pd.to_numeric(st.get("stop_sequence"), errors="coerce")
    st = st.dropna(subset=["trip_id", "stop_id", "stop_sequence"])
    merged = st.merge(stops[["stop_id", "stop_lat", "stop_lon"]], on="stop_id", how="left")
    merged = merged.merge(trips_df[["trip_id", "route_id"]], on="trip_id", how="left")
    merged = merged.dropna(subset=["route_id", "stop_lat", "stop_lon"])
    merged = merged.sort_values(["route_id", "trip_id", "stop_sequence"])
    sample = merged.groupby("route_id").head(60)
    for route_id, grp in sample.groupby("route_id"):
        coords = [[float(lon), float(lat)] for lat, lon in zip(grp["stop_lat"], grp["stop_lon"])]
        if len(coords) < 2:
            continue
        meta = route_meta.get(str(route_id), {})
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {"route_id": str(route_id), **meta},
            }
        )
    return features


def vehicle_type_from_route_type(route_type: str) -> str:
    if route_type in TRAIN_ROUTE_TYPES:
        return "train"
    if route_type in BUS_ROUTE_TYPES:
        return "bus"
    return "other"


def filter_routes_and_trips_for_city(
    city_id: str, routes_df: pd.DataFrame, trips_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # No hard filtering at the backend; UI controls decide route visibility.
    return routes_df, trips_df


def build_shape_index(shapes_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if shapes_df.empty:
        return {}
    shapes = shapes_df.copy().fillna("")
    for col in ("shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"):
        if col not in shapes.columns:
            shapes[col] = ""
    shapes["shape_pt_lat"] = pd.to_numeric(shapes["shape_pt_lat"], errors="coerce")
    shapes["shape_pt_lon"] = pd.to_numeric(shapes["shape_pt_lon"], errors="coerce")
    shapes["shape_pt_sequence"] = pd.to_numeric(shapes["shape_pt_sequence"], errors="coerce")
    shapes = shapes.dropna(subset=["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"])

    idx: dict[str, dict[str, Any]] = {}
    for shape_id, grp in shapes.sort_values("shape_pt_sequence").groupby("shape_id"):
        pts = [(float(lat), float(lon)) for lat, lon in zip(grp["shape_pt_lat"], grp["shape_pt_lon"])]
        if len(pts) < 2:
            continue
        cum = [0.0]
        total = 0.0
        for i in range(1, len(pts)):
            dy = pts[i][0] - pts[i - 1][0]
            dx = pts[i][1] - pts[i - 1][1]
            total += math.hypot(dx, dy)
            cum.append(total)
        if total <= 0:
            continue
        fracs = [v / total for v in cum]
        idx[str(shape_id)] = {"points": pts, "fractions": fracs}
    return idx


def point_on_shape(shape_obj: dict[str, Any], frac: float) -> tuple[float, float, float]:
    points = shape_obj["points"]
    fracs = shape_obj["fractions"]
    x = min(1.0, max(0.0, float(frac)))
    for i in range(1, len(fracs)):
        if x <= fracs[i]:
            f1 = fracs[i - 1]
            f2 = fracs[i]
            ratio = 0.0 if f2 <= f1 else (x - f1) / (f2 - f1)
            lat1, lon1 = points[i - 1]
            lat2, lon2 = points[i]
            lat = lat1 + (lat2 - lat1) * ratio
            lon = lon1 + (lon2 - lon1) * ratio
            bearing = (math.degrees(math.atan2((lon2 - lon1), (lat2 - lat1))) + 360.0) % 360.0
            return lat, lon, bearing
    lat1, lon1 = points[-2]
    lat2, lon2 = points[-1]
    bearing = (math.degrees(math.atan2((lon2 - lon1), (lat2 - lat1))) + 360.0) % 360.0
    return points[-1][0], points[-1][1], bearing


def preprocess_city(city_id: str, gtfs_dir: Path) -> dict[str, Any]:
    routes_df = safe_read_csv(
        gtfs_dir,
        "routes.txt",
        required=True,
        usecols=["route_id", "route_short_name", "route_long_name", "route_desc", "route_color", "route_type"],
    )
    trips_df = safe_read_csv(
        gtfs_dir,
        "trips.txt",
        required=True,
        usecols=["trip_id", "route_id", "service_id", "shape_id", "trip_headsign"],
    )
    shapes_df = safe_read_csv(
        gtfs_dir,
        "shapes.txt",
        required=False,
        usecols=["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"],
    )
    stops_df = safe_read_csv(
        gtfs_dir,
        "stops.txt",
        required=True,
        usecols=["stop_id", "stop_lat", "stop_lon"],
    )
    stop_times_df = safe_read_csv(
        gtfs_dir,
        "stop_times.txt",
        required=True,
        usecols=["trip_id", "stop_id", "stop_sequence", "arrival_time", "departure_time", "shape_dist_traveled"],
    )
    calendar_df = safe_read_csv(
        gtfs_dir,
        "calendar.txt",
        required=False,
        usecols=["service_id", "start_date", "end_date", *WEEKDAY_COLS],
    )
    calendar_dates_df = safe_read_csv(
        gtfs_dir,
        "calendar_dates.txt",
        required=False,
        usecols=["service_id", "date", "exception_type"],
    )
    agency_df = safe_read_csv(gtfs_dir, "agency.txt", required=False, usecols=["agency_timezone"])

    routes_df, trips_df = filter_routes_and_trips_for_city(city_id, routes_df, trips_df)
    route_features = build_route_features(routes_df, trips_df, shapes_df, stop_times_df, stops_df)
    route_geojson = {"type": "FeatureCollection", "features": route_features}

    stops = stops_df.copy().fillna("")
    stops["stop_lat"] = pd.to_numeric(stops.get("stop_lat"), errors="coerce")
    stops["stop_lon"] = pd.to_numeric(stops.get("stop_lon"), errors="coerce")
    stops = stops.dropna(subset=["stop_id", "stop_lat", "stop_lon"])

    st = stop_times_df.copy().fillna("")
    st["stop_sequence"] = pd.to_numeric(st.get("stop_sequence"), errors="coerce")
    st["arrival_sec"] = parse_gtfs_time_series(st.get("arrival_time", pd.Series(dtype=str)))
    st["departure_sec"] = parse_gtfs_time_series(st.get("departure_time", pd.Series(dtype=str)))
    st["time_sec"] = st["departure_sec"].fillna(st["arrival_sec"])
    st["shape_dist_traveled"] = pd.to_numeric(st.get("shape_dist_traveled", ""), errors="coerce")
    st = st.dropna(subset=["trip_id", "stop_id", "stop_sequence", "time_sec"])

    trips_subset = trips_df.copy().fillna("")
    for col in ("trip_id", "route_id", "service_id", "shape_id", "trip_headsign"):
        if col not in trips_subset.columns:
            trips_subset[col] = ""
    trips_subset = trips_subset[["trip_id", "route_id", "service_id", "shape_id", "trip_headsign"]]

    merged = st.merge(stops[["stop_id", "stop_lat", "stop_lon"]], on="stop_id", how="left")
    merged = merged.merge(trips_subset, on="trip_id", how="left")
    merged = merged.dropna(subset=["stop_lat", "stop_lon", "service_id"])
    merged = merged.sort_values(["trip_id", "stop_sequence"]).reset_index(drop=True)

    route_label_map = {}
    route_color_map = {}
    vehicle_type_map = {}
    route_is_tre_map = {}
    if not routes_df.empty:
        for _, row in routes_df.fillna("").iterrows():
            rid = str(row.get("route_id", ""))
            if not rid:
                continue
            short = str(row.get("route_short_name", "")).strip()
            long_name = str(row.get("route_long_name", "")).strip()
            route_label_map[rid] = short or long_name or rid
            color = str(row.get("route_color", "")).strip()
            route_color_map[rid] = f"#{color}" if re.fullmatch(r"[0-9a-fA-F]{6}", color or "") else "#5b6671"
            route_type = str(row.get("route_type", "")).strip()
            vehicle_type_map[rid] = vehicle_type_from_route_type(route_type)
            tre_text = " ".join(
                [
                    str(row.get("route_short_name", "")),
                    str(row.get("route_long_name", "")),
                    str(row.get("route_desc", "")),
                ]
            )
            route_is_tre_map[rid] = bool(TRE_PATTERN.search(tre_text))

    shape_index = build_shape_index(shapes_df)

    tzinfo = get_timezone(agency_df)
    return {
        "city_id": city_id,
        "loaded_at": datetime.now(tz=tz.UTC).isoformat(),
        "timezone": tzinfo,
        "route_geojson": route_geojson,
        "route_label_map": route_label_map,
        "route_color_map": route_color_map,
        "vehicle_type_map": vehicle_type_map,
        "route_is_tre_map": route_is_tre_map,
        "shape_index": shape_index,
        "trip_points": merged,
        "calendar": calendar_df.fillna(""),
        "calendar_dates": calendar_dates_df.fillna(""),
    }


def compute_vehicle_positions(city_data: dict[str, Any], now_local: datetime) -> list[dict[str, Any]]:
    trip_points: pd.DataFrame = city_data["trip_points"]
    if trip_points.empty:
        return []

    calendar_df = city_data["calendar"]
    calendar_dates_df = city_data["calendar_dates"]
    shape_index = city_data.get("shape_index", {})
    now_sec_today = (now_local.hour * 3600) + (now_local.minute * 60) + now_local.second

    candidate_days = [
        (now_local.date(), now_sec_today),
        (now_local.date() - timedelta(days=1), now_sec_today + 86400),
    ]

    vehicles: list[dict[str, Any]] = []
    seen_trips: set[str] = set()

    def build_positions(scoped: pd.DataFrame, eval_sec: int, simulated: bool) -> list[dict[str, Any]]:
        local: list[dict[str, Any]] = []
        bounds = scoped.groupby("trip_id")["time_sec"].agg(["min", "max"])
        if bounds.empty:
            return local
        active_trip_ids = bounds[(bounds["min"] <= eval_sec) & (bounds["max"] >= eval_sec)].index.tolist()
        if not active_trip_ids:
            return local

        for trip_id in active_trip_ids:
            if trip_id in seen_trips:
                continue
            tdf = scoped[scoped["trip_id"] == trip_id].sort_values("time_sec")
            if len(tdf) < 2:
                continue
            times = tdf["time_sec"].astype(int).to_numpy()
            trip_start = int(times[0])
            trip_end = int(times[-1])
            idx = int(times.searchsorted(eval_sec, side="right"))
            if idx <= 0:
                p = tdf.iloc[0]
                lat = float(p["stop_lat"])
                lon = float(p["stop_lon"])
                bearing = 0.0
            elif idx >= len(tdf):
                p = tdf.iloc[-1]
                lat = float(p["stop_lat"])
                lon = float(p["stop_lon"])
                bearing = 0.0
            else:
                p1 = tdf.iloc[idx - 1]
                p2 = tdf.iloc[idx]
                t1 = int(p1["time_sec"])
                t2 = int(p2["time_sec"])
                ratio = 0.0 if t2 <= t1 else (eval_sec - t1) / (t2 - t1)
                lat1, lon1 = float(p1["stop_lat"]), float(p1["stop_lon"])
                lat2, lon2 = float(p2["stop_lat"]), float(p2["stop_lon"])
                lat = lat1 + (lat2 - lat1) * ratio
                lon = lon1 + (lon2 - lon1) * ratio
                dy = lat2 - lat1
                dx = lon2 - lon1
                bearing = (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0

            row = tdf.iloc[0]
            route_id = str(row.get("route_id", ""))
            label = city_data["route_label_map"].get(route_id, route_id)
            route_color = city_data["route_color_map"].get(route_id, "#5b6671")
            vehicle_type = city_data["vehicle_type_map"].get(route_id, "other")
            is_tre = bool(city_data.get("route_is_tre_map", {}).get(route_id, False))

            # Constrain train points to GTFS shapes for map-perfect line following.
            if vehicle_type == "train":
                shape_id = str(row.get("shape_id", "")).strip()
                shape_obj = shape_index.get(shape_id)
                if shape_obj and trip_end > trip_start:
                    frac = (eval_sec - trip_start) / (trip_end - trip_start)
                    lat, lon, bearing = point_on_shape(shape_obj, frac)

            local.append(
                {
                    "trip_id": str(trip_id),
                    "route_id": route_id,
                    "label": label,
                    "route_color": route_color,
                    "vehicle_type": vehicle_type,
                    "is_tre": is_tre,
                    "lat": round(lat, 6),
                    "lon": round(lon, 6),
                    "bearing_estimate": round(bearing, 1),
                    "simulated": simulated,
                }
            )
            seen_trips.add(str(trip_id))
        return local

    for service_day, eval_sec in candidate_days:
        service_ids = service_ids_for_date(calendar_df, calendar_dates_df, service_day)
        if not service_ids:
            continue
        scoped = trip_points[trip_points["service_id"].isin(service_ids)]
        if scoped.empty:
            continue
        vehicles.extend(build_positions(scoped, eval_sec, simulated=False))

    if vehicles:
        return vehicles

    # Fallback mode: if no active vehicles now, simulate movement within the service span.
    service_ids_today = service_ids_for_date(calendar_df, calendar_dates_df, now_local.date())
    if service_ids_today:
        scoped_today = trip_points[trip_points["service_id"].isin(service_ids_today)]
        if not scoped_today.empty:
            bounds = scoped_today.groupby("trip_id")["time_sec"].agg(["min", "max"])
            if not bounds.empty:
                min_sec = int(bounds["min"].min())
                max_sec = int(bounds["max"].max())
                span = max(1, max_sec - min_sec)
                simulated_sec = min_sec + (now_sec_today % span)
                vehicles.extend(build_positions(scoped_today, simulated_sec, simulated=True))

    return vehicles


def refresh_city(city_id: str) -> dict[str, Any]:
    cfg = CITY_CONFIG[city_id]
    gtfs_dir = download_and_extract(city_id, cfg["url"])
    city_data = preprocess_city(city_id, gtfs_dir)
    with cache_lock:
        data_cache[city_id] = city_data
    return city_data


def _load_city_background(city_id: str) -> None:
    global initialized
    try:
        refresh_city(city_id)
        with cache_lock:
            initialized = bool(data_cache)
    except Exception:
        pass
    finally:
        with loading_lock:
            loading_cities.discard(city_id)


def start_city_load(city_id: str) -> None:
    with cache_lock:
        if city_id in data_cache:
            return
    with loading_lock:
        if city_id in loading_cities:
            return
        loading_cities.add(city_id)
    t = threading.Thread(target=_load_city_background, args=(city_id,), daemon=True)
    t.start()


def refresh_all() -> dict[str, Any]:
    status: dict[str, Any] = {"ok": True, "updated": [], "errors": {}, "timestamp": datetime.utcnow().isoformat() + "Z"}
    for city_id in CITY_CONFIG:
        try:
            refresh_city(city_id)
            status["updated"].append(city_id)
        except Exception as exc:  # pragma: no cover - defensive runtime path
            status["ok"] = False
            status["errors"][city_id] = str(exc)
    return status


def ensure_initialized() -> None:
    global initialized
    if initialized:
        return
    with cache_lock:
        if initialized:
            return
    # Avoid request-time heavy initialization under hosted worker timeouts.
    for city_id in CITY_CONFIG:
        start_city_load(city_id)
    with cache_lock:
        initialized = bool(data_cache)


def city_or_400() -> str:
    city_id = request.args.get("city", "").strip().lower()
    if city_id not in CITY_CONFIG:
        raise ValueError("city must be one of: dallas, hattiesburg")
    return city_id


@app.route("/")
def root() -> Any:
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/health")
def health() -> Any:
    with loading_lock:
        loading = sorted(list(loading_cities))
    return jsonify(
        {
            "ok": True,
            "initialized": initialized,
            "cities_loaded": sorted(list(data_cache.keys())),
            "cities_loading": loading,
        }
    )


@app.route("/api/cities")
def api_cities() -> Any:
    for city_id in CITY_CONFIG:
        start_city_load(city_id)
    payload = [
        {
            "id": city_id,
            "name": cfg["name"],
            "center": cfg["center"],
            "zoom": cfg["zoom"],
        }
        for city_id, cfg in CITY_CONFIG.items()
    ]
    return jsonify(payload)


@app.route("/api/routes")
def api_routes() -> Any:
    try:
        city_id = city_or_400()
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    start_city_load(city_id)
    with cache_lock:
        city_data = data_cache.get(city_id)
    if not city_data:
        return jsonify({"ok": False, "loading": True, "error": f"Loading data for {city_id}"}), 503
    return jsonify(city_data["route_geojson"])


@app.route("/api/vehicles")
def api_vehicles() -> Any:
    try:
        city_id = city_or_400()
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    start_city_load(city_id)
    with cache_lock:
        city_data = data_cache.get(city_id)
    if not city_data:
        return jsonify({"ok": False, "loading": True, "error": f"Loading data for {city_id}"}), 503

    tzinfo = city_data.get("timezone")
    now_local = datetime.now(tz=tzinfo) if tzinfo else datetime.now()
    vehicles = compute_vehicle_positions(city_data, now_local)
    return jsonify({"as_of": now_local.isoformat(), "city": city_id, "vehicles": vehicles})


@app.route("/api/refresh", methods=["POST"])
def api_refresh() -> Any:
    global initialized
    status = refresh_all()
    with cache_lock:
        initialized = bool(data_cache)
    code = 200 if status["ok"] else 207
    return jsonify(status), code


if __name__ == "__main__":
    ensure_initialized()
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
