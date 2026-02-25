const POLL_MS = 8000;
const statusEl = document.getElementById("status-text");
const toggleEl = document.getElementById("city-toggle");
const vehicleToggleEl = document.getElementById("vehicle-toggle");
const lineFilterWrapEl = document.getElementById("line-filter-wrap");
const lineFilterButtonEl = document.getElementById("line-filter-button");
const lineFilterMenuEl = document.getElementById("line-filter-menu");
const themeToggleEl = document.getElementById("theme-toggle");

const map = L.map("map", {
  zoomControl: true,
  attributionControl: true,
});

L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
  attribution: "&copy; OpenStreetMap contributors",
}).addTo(map);

let cities = [];
let activeCity = null;
let routeLayer = null;
let vehicleLayer = L.layerGroup().addTo(map);
let markersByTrip = new Map();
let animationFramesByTrip = new Map();
let pollHandle = null;
let activeVehicleTypes = new Set();
let availableVehicleTypes = [];
let lastVehiclesPayload = null;
let dallasTrainLines = [];
let selectedDallasTrainRouteIds = new Set();
let treEnabled = false;
let theme = "light";

const VEHICLE_TYPE_LABELS = {
  bus: "Bus",
  train: "Train",
};

function setStatus(text) {
  statusEl.textContent = text;
}

function normalizeRouteColor(rawColor) {
  if (!rawColor) return "#5b6671";
  if (/^#[0-9a-fA-F]{6}$/.test(rawColor)) return rawColor;
  if (/^[0-9a-fA-F]{6}$/.test(rawColor)) return `#${rawColor}`;
  return "#5b6671";
}

function hexToRgb(hex) {
  const clean = hex.replace("#", "");
  const value = parseInt(clean, 16);
  return {
    r: (value >> 16) & 255,
    g: (value >> 8) & 255,
    b: value & 255,
  };
}

function rgbToHex(r, g, b) {
  return `#${[r, g, b]
    .map((n) => Math.max(0, Math.min(255, Math.round(n))).toString(16).padStart(2, "0"))
    .join("")}`;
}

function mutedColorForTheme(hexColor) {
  const base = normalizeRouteColor(hexColor);
  if (theme !== "dark") return base;
  const { r, g, b } = hexToRgb(base);
  const gray = 110;
  const mix = 0.52;
  return rgbToHex(r * (1 - mix) + gray * mix, g * (1 - mix) + gray * mix, b * (1 - mix) + gray * mix);
}

function applyTheme(nextTheme) {
  theme = nextTheme === "dark" ? "dark" : "light";
  document.body.setAttribute("data-theme", theme);
  themeToggleEl.textContent = theme === "dark" ? "Light mode" : "Dark mode";
  localStorage.setItem("map_theme", theme);
  applyRouteFilterStyles();
  if (lastVehiclesPayload) renderVehicles(lastVehiclesPayload);
}

function buildToggle() {
  toggleEl.innerHTML = "";
  cities.forEach((city) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = city.name;
    btn.dataset.cityId = city.id;
    btn.className = city.id === activeCity ? "active" : "";
    btn.addEventListener("click", () => switchCity(city.id));
    toggleEl.appendChild(btn);
  });
}

function paintToggle() {
  [...toggleEl.querySelectorAll("button")].forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.cityId === activeCity);
  });
}

function routeColor(feature) {
  return mutedColorForTheme(feature?.properties?.route_color || "");
}

function isDallasTrainRouteEnabled(routeId, vehicleType) {
  if (activeCity !== "dallas" || vehicleType !== "train") return true;
  return selectedDallasTrainRouteIds.has(routeId);
}

function routeOpacity(feature) {
  const type = feature?.properties?.vehicle_type || "other";
  const routeId = feature?.properties?.route_id || "";
  if (!activeVehicleTypes.has(type)) return 0;
  return isDallasTrainRouteEnabled(routeId, type) ? 0.82 : 0;
}

function applyRouteFilterStyles() {
  if (!routeLayer) return;
  routeLayer.setStyle((feature) => ({
    color: routeColor(feature),
    weight: 2.5,
    opacity: routeOpacity(feature),
  }));
}

function buildVehicleTypeToggle() {
  vehicleToggleEl.innerHTML = "";
  const order = ["bus", "train"];
  const sorted = [...availableVehicleTypes].sort((a, b) => order.indexOf(a) - order.indexOf(b));
  sorted.forEach((type) => {
    if (!VEHICLE_TYPE_LABELS[type]) return;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = VEHICLE_TYPE_LABELS[type] || type;
    btn.dataset.vehicleType = type;
    btn.className = activeVehicleTypes.has(type) ? "active" : "";
    btn.addEventListener("click", () => {
      if (activeVehicleTypes.has(type)) activeVehicleTypes.delete(type);
      else activeVehicleTypes.add(type);
      paintVehicleTypeToggle();
      applyRouteFilterStyles();
      if (lastVehiclesPayload) renderVehicles(lastVehiclesPayload);
    });
    vehicleToggleEl.appendChild(btn);
  });
}

function paintVehicleTypeToggle() {
  [...vehicleToggleEl.querySelectorAll("button")].forEach((btn) => {
    const type = btn.dataset.vehicleType || "";
    btn.classList.toggle("active", activeVehicleTypes.has(type));
  });
}

function toggleLineFilterMenu() {
  lineFilterMenuEl.classList.toggle("open");
}

function closeLineFilterMenu() {
  lineFilterMenuEl.classList.remove("open");
}

function buildDallasTrainLineFilter() {
  lineFilterMenuEl.innerHTML = "";
  if (activeCity !== "dallas" || !dallasTrainLines.length) {
    lineFilterWrapEl.classList.add("hidden");
    closeLineFilterMenu();
    return;
  }

  lineFilterWrapEl.classList.remove("hidden");
  const treBtn = document.createElement("button");
  treBtn.type = "button";
  treBtn.className = treEnabled ? "line-chip active" : "line-chip";
  treBtn.textContent = "TRE";
  treBtn.addEventListener("click", () => {
    treEnabled = !treEnabled;
    dallasTrainLines.filter((line) => line.is_tre).forEach((line) => {
      if (treEnabled) selectedDallasTrainRouteIds.add(line.route_id);
      else selectedDallasTrainRouteIds.delete(line.route_id);
    });
    buildDallasTrainLineFilter();
    applyRouteFilterStyles();
    if (lastVehiclesPayload) renderVehicles(lastVehiclesPayload);
  });
  lineFilterMenuEl.appendChild(treBtn);

  dallasTrainLines.forEach((line) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = selectedDallasTrainRouteIds.has(line.route_id) ? "line-chip active" : "line-chip";
    btn.textContent = line.label;
    btn.dataset.routeId = line.route_id;
    btn.addEventListener("click", () => {
      if (selectedDallasTrainRouteIds.has(line.route_id)) selectedDallasTrainRouteIds.delete(line.route_id);
      else selectedDallasTrainRouteIds.add(line.route_id);
      buildDallasTrainLineFilter();
      applyRouteFilterStyles();
      if (lastVehiclesPayload) renderVehicles(lastVehiclesPayload);
    });
    lineFilterMenuEl.appendChild(btn);
  });
}

async function loadRoutes(cityId) {
  const res = await fetch(`/api/routes?city=${encodeURIComponent(cityId)}`);
  if (!res.ok) throw new Error("Failed to load routes");
  const geojson = await res.json();
  const features = geojson?.features || [];

  const presentTypes = new Set(features.map((f) => f?.properties?.vehicle_type || "other").filter(Boolean));
  availableVehicleTypes = [...presentTypes];
  activeVehicleTypes = new Set(availableVehicleTypes.filter((t) => t === "bus" || t === "train"));
  buildVehicleTypeToggle();

  if (cityId === "dallas") {
    const byRoute = new Map();
    features.forEach((f) => {
      const p = f?.properties || {};
      if ((p.vehicle_type || "other") !== "train") return;
      const routeId = p.route_id || "";
      if (!routeId || byRoute.has(routeId)) return;
      const label = p.route_short_name || p.route_long_name || routeId;
      byRoute.set(routeId, { route_id: routeId, label, is_tre: !!p.is_tre });
    });
    dallasTrainLines = [...byRoute.values()].sort((a, b) => a.label.localeCompare(b.label));
    treEnabled = false;
    selectedDallasTrainRouteIds = new Set(dallasTrainLines.filter((r) => !r.is_tre).map((r) => r.route_id));
  } else {
    dallasTrainLines = [];
    selectedDallasTrainRouteIds = new Set();
    treEnabled = false;
  }
  buildDallasTrainLineFilter();

  if (routeLayer) map.removeLayer(routeLayer);
  routeLayer = L.geoJSON(geojson, {
    style: (feature) => ({
      color: routeColor(feature),
      weight: 2.5,
      opacity: routeOpacity(feature),
    }),
    className: "route-line",
    interactive: false,
  }).addTo(map);
}

function clearVehicles() {
  animationFramesByTrip.forEach((frameId) => cancelAnimationFrame(frameId));
  animationFramesByTrip.clear();
  vehicleLayer.clearLayers();
  markersByTrip = new Map();
}

function animateMarkerTo(tripId, marker, targetLatLng, durationMs) {
  const currentFrame = animationFramesByTrip.get(tripId);
  if (currentFrame) cancelAnimationFrame(currentFrame);

  const start = marker.getLatLng();
  const startLat = start.lat;
  const startLng = start.lng;
  const endLat = targetLatLng[0];
  const endLng = targetLatLng[1];
  const startTime = performance.now();
  const duration = Math.max(500, durationMs);

  const step = (now) => {
    const t = Math.min(1, (now - startTime) / duration);
    marker.setLatLng([startLat + (endLat - startLat) * t, startLng + (endLng - startLng) * t]);
    if (t < 1) {
      const frameId = requestAnimationFrame(step);
      animationFramesByTrip.set(tripId, frameId);
    } else {
      animationFramesByTrip.delete(tripId);
    }
  };
  const frameId = requestAnimationFrame(step);
  animationFramesByTrip.set(tripId, frameId);
}

function filterVehicle(v) {
  const type = v.vehicle_type || "other";
  if (!activeVehicleTypes.has(type)) return false;
  return isDallasTrainRouteEnabled(v.route_id || "", type);
}

function renderVehicles(payload) {
  lastVehiclesPayload = payload;
  const filteredVehicles = payload.vehicles.filter(filterVehicle);
  const incomingTripIds = new Set(filteredVehicles.map((v) => v.trip_id));

  markersByTrip.forEach((marker, tripId) => {
    if (!incomingTripIds.has(tripId)) {
      const frameId = animationFramesByTrip.get(tripId);
      if (frameId) {
        cancelAnimationFrame(frameId);
        animationFramesByTrip.delete(tripId);
      }
      vehicleLayer.removeLayer(marker);
      markersByTrip.delete(tripId);
    }
  });

  filteredVehicles.forEach((v) => {
    const latlng = [v.lat, v.lon];
    let marker = markersByTrip.get(v.trip_id);
    if (!marker) {
      marker = L.circleMarker(latlng, {
        radius: 7,
        weight: 3,
        color: mutedColorForTheme(v.route_color),
        fillColor: "#ffffff",
        fillOpacity: 1,
        className: "vehicle-ring",
      });
      marker.bindTooltip(v.label || v.route_id || "Vehicle", { direction: "top", offset: [0, -6] });
      marker.addTo(vehicleLayer);
      markersByTrip.set(v.trip_id, marker);
    } else {
      animateMarkerTo(v.trip_id, marker, latlng, POLL_MS - 120);
      marker.setStyle({ color: mutedColorForTheme(v.route_color), fillColor: "#ffffff" });
      marker.setTooltipContent(v.label || v.route_id || "Vehicle");
    }
  });

  const activeTypeLabel = [...activeVehicleTypes]
    .map((t) => VEHICLE_TYPE_LABELS[t])
    .filter(Boolean)
    .join("/");
  const simTag = filteredVehicles.some((v) => v.simulated) ? " | Simulated service window" : "";
  setStatus(
    `${payload.city.toUpperCase()} | ${filteredVehicles.length} active vehicles | ${activeTypeLabel || "None"}${simTag} | Updated ${new Date(
      payload.as_of
    ).toLocaleTimeString()}`
  );
}

async function loadVehicles(cityId) {
  const res = await fetch(`/api/vehicles?city=${encodeURIComponent(cityId)}`);
  if (!res.ok) throw new Error("Failed to load vehicles");
  const payload = await res.json();
  renderVehicles(payload);
}

async function switchCity(cityId) {
  if (activeCity === cityId && routeLayer) return;
  activeCity = cityId;
  paintToggle();
  closeLineFilterMenu();
  clearVehicles();
  lastVehiclesPayload = null;
  if (pollHandle) {
    clearInterval(pollHandle);
    pollHandle = null;
  }

  const city = cities.find((c) => c.id === cityId);
  if (city) map.setView(city.center, city.zoom, { animate: true });

  try {
    setStatus(`Loading ${city?.name || cityId}...`);
    await loadRoutes(cityId);
    await loadVehicles(cityId);
    pollHandle = setInterval(() => loadVehicles(cityId).catch(() => {}), POLL_MS);
  } catch (err) {
    setStatus(`Error loading ${city?.name || cityId}`);
    console.error(err);
  }
}

async function init() {
  const savedTheme = localStorage.getItem("map_theme");
  applyTheme(savedTheme === "dark" ? "dark" : "light");

  themeToggleEl.addEventListener("click", () => {
    applyTheme(theme === "dark" ? "light" : "dark");
  });

  lineFilterButtonEl.addEventListener("click", toggleLineFilterMenu);
  document.addEventListener("click", (event) => {
    if (!lineFilterWrapEl.contains(event.target)) closeLineFilterMenu();
  });

  try {
    const res = await fetch("/api/cities");
    if (!res.ok) throw new Error("Failed to load city list");
    cities = await res.json();
    if (!cities.length) throw new Error("No cities configured");
    activeCity = cities[0].id;
    buildToggle();
    await switchCity(activeCity);
  } catch (err) {
    setStatus("Initialization failed. See console for details.");
    console.error(err);
  }
}

init();
