import {
  BarChart3,
  Brain,
  FileText,
  Layers,
  Map,
  Menu,
  Settings,
  Satellite,
  ShieldAlert,
  SlidersHorizontal,
  Activity,
} from "lucide-react";

import { useEffect, useState } from "react";
import GoogleSatelliteMap from "./components/GoogleSatelliteMap";

import "./App.css";

const navigation = [
  {
    label: "Dashboard",
    icon: Activity,
    active: true,
  },
  {
    label: "Analyze Area",
    icon: SlidersHorizontal,
  },
  {
    label: "Change Map",
    icon: Map,
  },
  {
    label: "Change Zones",
    icon: Layers,
  },
  {
    label: "Temporal Analysis",
    icon: BarChart3,
  },
  {
    label: "Explainable AI",
    icon: Brain,
  },
  {
    label: "Reports",
    icon: FileText,
  },
];

function App() {
  const [zones, setZones] = useState([]);
  const [selectedZone, setSelectedZone] = useState(null);
  const [zonesLoading, setZonesLoading] = useState(true);
  const [zonesError, setZonesError] = useState(null);
  const [zoneEvidence, setZoneEvidence] = useState(null);
  const [evidenceLoading, setEvidenceLoading] = useState(false);
  const [evidenceError, setEvidenceError] = useState(null);

  useEffect(() => {
    const loadZones = async () => {
      try {
        setZonesLoading(true);
        setZonesError(null);

        const apiBase =
          import.meta.env.VITE_API_BASE_URL ||
          "http://127.0.0.1:8000";

        const response = await fetch(
          `${apiBase}/api/v1/analysis/zones`,
        );

        if (!response.ok) {
          throw new Error(
            `API request failed: ${response.status}`,
          );
        }

        const data = await response.json();

        setZones(data.features || []);
      } catch (error) {
        console.error(
          "Failed to load change zones:",
          error,
        );

        setZonesError(error.message);
      } finally {
        setZonesLoading(false);
      }
    };

    loadZones();
  }, []);

  useEffect(() => {
    if (!selectedZone) {
      setZoneEvidence(null);
      setEvidenceError(null);
      return;
    }

    const regionId =
      selectedZone.properties?.region_id;

    if (!regionId) {
      return;
    }

    const loadEvidence = async () => {
      try {
        setEvidenceLoading(true);
        setEvidenceError(null);

        const apiBase =
          import.meta.env.VITE_API_BASE_URL ||
          "http://127.0.0.1:8000";

        const response = await fetch(
          `${apiBase}/api/v1/analysis/zones/${regionId}/evidence`,
        );

        if (!response.ok) {
          throw new Error(
            `Evidence request failed: ${response.status}`,
          );
        }

        const data = await response.json();

        setZoneEvidence(data);
      } catch (error) {
        console.error(
          "Failed to load zone evidence:",
          error,
        );

        setEvidenceError(error.message);
        setZoneEvidence(null);
      } finally {
        setEvidenceLoading(false);
      }
    };

    loadEvidence();
  }, [selectedZone]);
  const selectedProperties =
    selectedZone?.properties || {};

  const selectedNdviEvidence =
    selectedProperties.ndvi_evidence || {};

  const selectedChangeEvidence =
    selectedProperties.change_evidence || {};

  const formatNumber = (value, decimals = 2) => {
    if (
      value === undefined ||
      value === null ||
      Number.isNaN(Number(value))
    ) {
      return "—";
    }

    return Number(value).toFixed(decimals);
  };

  const formatArea = (value) => {
    if (value === undefined || value === null) {
      return "—";
    }

    return `${Number(value).toLocaleString(
      undefined,
      {
        maximumFractionDigits: 1,
      },
    )} m²`;
  };

  const formatDirection = (value) => {
    if (!value) {
      return "—";
    }

    return value
      .replaceAll("_", " ")
      .replace(/\b\w/g, (letter) =>
        letter.toUpperCase(),
      );
  };

  const formatSeverity = (value) => {
    if (!value) {
      return "—";
    }

    return value.toUpperCase();
  };

  return (
    <div className="terrain-app">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <Satellite size={22} />
          </div>

          <div>
            <div className="brand-name">
              TERRAIN
            </div>

            <div className="brand-subtitle">
              CHANGE INTELLIGENCE
            </div>
          </div>
        </div>

        <div className="system-status">
          <span className="status-dot" />
          <span>SYSTEM ONLINE</span>
        </div>

        <nav className="navigation">
          {navigation.map((item) => {
            const Icon = item.icon;

            return (
              <button
                key={item.label}
                className={`nav-item ${
                  item.active ? "active" : ""
                }`}
              >
                <Icon size={18} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        <div className="sidebar-bottom">
          <button className="nav-item">
            <Settings size={18} />
            <span>Settings</span>
          </button>

          <div className="pipeline-status">
            <div className="pipeline-header">
              <span>Pipeline</span>

              <span className="pipeline-ready">
                READY
              </span>
            </div>

            <div className="pipeline-bar">
              <div className="pipeline-progress" />
            </div>

            <div className="pipeline-text">
              Sentinel-2 spectral engine
            </div>
          </div>
        </div>
      </aside>

      {/* Main content */}
      <main className="main-content">
        <header className="topbar">
          <div className="topbar-left">
            <button className="mobile-menu">
              <Menu size={20} />
            </button>

            <div>
              <div className="page-kicker">
                SATELLITE CHANGE INTELLIGENCE
              </div>

              <h1>Analysis Dashboard</h1>
            </div>
          </div>

          <div className="topbar-right">
            <div className="acquisition-status">
              <span className="status-dot" />

              <div>
                <strong>Sentinel-2</strong>
                <span>
                  Data pipeline operational
                </span>
              </div>
            </div>

            <button className="icon-button">
              <ShieldAlert size={19} />
            </button>
          </div>
        </header>

        <section className="dashboard-content">
          {/* Analysis controls */}
          <div className="control-card">
            <div className="control-heading">
              <div>
                <span className="section-label">
                  ANALYSIS CONFIGURATION
                </span>

                <h2>Monitor an area</h2>
              </div>

              <span className="ready-badge">
                READY
              </span>
            </div>

            <div className="control-grid">
              <div className="control-field">
                <label>Latitude</label>

                <input
                  type="text"
                  value="12.971912"
                  readOnly
                />
              </div>

              <div className="control-field">
                <label>Longitude</label>

                <input
                  type="text"
                  value="77.614378"
                  readOnly
                />
              </div>

              <div className="control-field">
                <label>Radius</label>

                <div className="input-with-unit">
                  <input
                    type="text"
                    value="500"
                    readOnly
                  />

                  <span>m</span>
                </div>
              </div>

              <div className="control-field">
                <label>Before</label>

                <input
                  type="text"
                  value="2020-03-29"
                  readOnly
                />
              </div>

              <div className="control-field">
                <label>After</label>

                <input
                  type="text"
                  value="2026-01-22"
                  readOnly
                />
              </div>

              <button className="analyze-button">
                <Satellite size={18} />
                Analyze Area
              </button>
            </div>
          </div>

          {/* Stats */}
          <div className="stats-grid">
            <div className="stat-card">
              <span className="stat-label">
                CHANGE ZONES
              </span>

              <strong>
                {zonesLoading ? "—" : zones.length}
              </strong>

              <span className="stat-description">
                Detected regions
              </span>
            </div>

            <div className="stat-card">
              <span className="stat-label">
                CHANGED AREA
              </span>

              <strong>5.00%</strong>

              <span className="stat-description">
                Of valid ROI
              </span>
            </div>

            <div className="stat-card">
              <span className="stat-label">
                ANALYSIS RADIUS
              </span>

              <strong>500 m</strong>

              <span className="stat-description">
                Area of interest
              </span>
            </div>

            <div className="stat-card">
              <span className="stat-label">
                TEMPORAL WINDOW
              </span>

              <strong>5.8 yr</strong>

              <span className="stat-description">
                2020 → 2026
              </span>
            </div>
          </div>

          {/* Change Map */}
          <div className="map-card">
            <div className="map-overlay">
              <div className="map-title">
                <Map size={18} />

                <div>
                  <strong>Change Map</strong>

                  <span>
                    Bengaluru · 500 m radius
                  </span>
                </div>
              </div>

              <div className="map-layer-control">
                <button className="layer-active">
                  Change Zones
                </button>

                <button>Satellite</button>

                <button>NDVI</button>
              </div>
            </div>

            {zonesLoading ? (
              <div className="map-loading">
                <div className="map-loading-spinner" />

                <span>
                  Loading detected change zones...
                </span>
              </div>
            ) : zonesError ? (
              <div className="map-error">
                <strong>
                  Unable to load change zones
                </strong>

                <span>{zonesError}</span>
              </div>
            ) : (
              <GoogleSatelliteMap
                zones={zones}
                onZoneSelect={setSelectedZone}
              />
            )}
          </div>

          {/* Selected Zone Intelligence */}
          {selectedZone && (
            <section className="zone-intelligence-card">
              <div className="zone-intelligence-header">
                <div>
                  <span className="section-label">
                    SELECTED CHANGE ZONE
                  </span>

                  <h3>
                    Zone{" "}
                    {selectedProperties.region_id ??
                      "—"}
                  </h3>

                  <span className="zone-direction">
                    {formatDirection(
                      selectedProperties.direction,
                    )}
                  </span>
                </div>

                <div className="zone-header-actions">
                  <span
                    className={`severity-badge severity-${
                      selectedProperties.severity ||
                      "unknown"
                    }`}
                  >
                    {formatSeverity(
                      selectedProperties.severity,
                    )}
                  </span>

                  <button
                    className="zone-close-button"
                    onClick={() =>
                      setSelectedZone(null)
                    }
                    aria-label="Close selected zone"
                  >
                    ×
                  </button>
                </div>
              </div>

              {/* Zone metrics */}
              <div className="zone-metrics-grid">
                <div className="zone-metric">
                  <span>CHANGED AREA</span>

                  <strong>
                    {formatArea(
                      selectedProperties.area_m2,
                    )}
                  </strong>
                </div>

                <div className="zone-metric">
                  <span>CONFIDENCE</span>

                  <strong>
                    {selectedProperties.confidence !==
                    undefined
                      ? `${formatNumber(
                          selectedProperties.confidence *
                            100,
                          1,
                        )}%`
                      : "—"}
                  </strong>
                </div>

                <div className="zone-metric">
                  <span>NDVI BEFORE</span>

                  <strong>
                    {formatNumber(
                      selectedNdviEvidence.before,
                      3,
                    )}
                  </strong>
                </div>

                <div className="zone-metric">
                  <span>NDVI AFTER</span>

                  <strong>
                    {formatNumber(
                      selectedNdviEvidence.after,
                      3,
                    )}
                  </strong>
                </div>

                <div className="zone-metric">
                  <span>NDVI CHANGE</span>

                  <strong
                    className={
                      Number(
                        selectedNdviEvidence.change,
                      ) < 0
                        ? "metric-negative"
                        : "metric-positive"
                    }
                  >
                    {selectedNdviEvidence.change !==
                    undefined
                      ? `${
                          Number(
                            selectedNdviEvidence.change,
                          ) > 0
                            ? "+"
                            : ""
                        }${formatNumber(
                          selectedNdviEvidence.change,
                          3,
                        )}`
                      : "—"}
                  </strong>
                </div>

                <div className="zone-metric">
                  <span>PIXELS</span>

                  <strong>
                    {selectedProperties.pixel_count ??
                      "—"}
                  </strong>
                </div>
              </div>
              {/* Visual Evidence */}
              <div className="zone-visual-evidence">
                <div className="zone-evidence-title">
                  <div>
                    <span className="section-label">
                      VISUAL EVIDENCE
                    </span>

                    <h4>
                      Sentinel-2 temporal comparison
                    </h4>
                  </div>

                  <span className="evidence-source">
                    Sentinel-2 · B04/B03/B02
                  </span>
                </div>

                {evidenceLoading ? (
                  <div className="evidence-loading">
                    <div className="map-loading-spinner" />

                    <span>
                      Generating zone evidence...
                    </span>
                  </div>
                ) : evidenceError ? (
                  <div className="evidence-error">
                    <strong>
                      Unable to generate evidence
                    </strong>

                    <span>{evidenceError}</span>
                  </div>
                ) : zoneEvidence?.images ? (
                  <div className="evidence-image-grid">
                    <div className="evidence-image-card">
                      <div className="evidence-image-header">
                        <strong>BEFORE</strong>
                        <span>2020-03-29</span>
                      </div>

                      <img
                        src={zoneEvidence.images.before}
                        alt="Sentinel-2 before imagery"
                      />
                    </div>

                    <div className="evidence-image-card">
                      <div className="evidence-image-header">
                        <strong>AFTER</strong>
                        <span>2026-01-22</span>
                      </div>

                      <img
                        src={zoneEvidence.images.after}
                        alt="Sentinel-2 after imagery"
                      />
                    </div>

                    <div className="evidence-image-card">
                      <div className="evidence-image-header">
                        <strong>CHANGE ZONE</strong>
                        <span>Detected region</span>
                      </div>

                      <img
                        src={zoneEvidence.images.overlay}
                        alt="Detected change zone overlay"
                      />
                    </div>
                  </div>
                ) : null}

                <div className="evidence-disclaimer">
                  RGB imagery is shown for visual interpretation.
                  Detection metrics are calculated from the
                  underlying multispectral Sentinel-2 data.
                </div>
              </div>

              {/* Multispectral Evidence */}
              {zoneEvidence?.band_profile && (
                <div className="multispectral-evidence">
                  <div className="zone-evidence-title">
                    <div>
                      <span className="section-label">
                        MULTISPECTRAL EVIDENCE
                      </span>

                      <h4>
                        Sentinel-2 band response
                      </h4>
                    </div>

                    <span className="evidence-source">
                      B02 · B03 · B04 · B08
                    </span>
                  </div>

                  <div className="band-table">
                    <div className="band-table-header">
                      <span>BAND</span>
                      <span>BEFORE</span>
                      <span>AFTER</span>
                      <span>Δ</span>
                    </div>

                    {[
                      ["B02", "Blue"],
                      ["B03", "Green"],
                      ["B04", "Red"],
                      ["B08", "NIR"],
                    ].map(([band, label]) => {
                      const values =
                        zoneEvidence.band_profile[band];

                      return (
                        <div
                          className="band-table-row"
                          key={band}
                        >
                          <div>
                            <strong>{band}</strong>
                            <span>{label}</span>
                          </div>

                          <span>
                            {formatNumber(
                              values?.before,
                              4,
                            )}
                          </span>

                          <span>
                            {formatNumber(
                              values?.after,
                              4,
                            )}
                          </span>

                          <span
                            className={
                              Number(values?.change) < 0
                                ? "metric-negative"
                                : "metric-positive"
                            }
                          >
                            {Number(values?.change) > 0
                              ? "+"
                              : ""}
                            {formatNumber(
                              values?.change,
                              4,
                            )}
                          </span>
                        </div>
                      );
                    })}
                  </div>

                  <div className="spectral-summary-grid">
                    <div>
                      <span>NDVI BEFORE</span>

                      <strong>
                        {formatNumber(
                          zoneEvidence.ndvi_profile
                            ?.before,
                          3,
                        )}
                      </strong>
                    </div>

                    <div>
                      <span>NDVI AFTER</span>

                      <strong>
                        {formatNumber(
                          zoneEvidence.ndvi_profile
                            ?.after,
                          3,
                        )}
                      </strong>
                    </div>

                    <div>
                      <span>NDVI CHANGE</span>

                      <strong
                        className={
                          Number(
                            zoneEvidence.ndvi_profile
                              ?.change,
                          ) < 0
                            ? "metric-negative"
                            : "metric-positive"
                        }
                      >
                        {Number(
                          zoneEvidence.ndvi_profile
                            ?.change,
                        ) > 0
                          ? "+"
                          : ""}
                        {formatNumber(
                          zoneEvidence.ndvi_profile
                            ?.change,
                          3,
                        )}
                      </strong>
                    </div>

                    <div>
                      <span>SPECTRAL CONTRAST</span>

                      <strong>
                        {formatNumber(
                          zoneEvidence.spectral_profile
                            ?.mean,
                          3,
                        )}
                      </strong>
                    </div>
                  </div>

                  <div className="evidence-disclaimer">
                    Band values represent mean surface reflectance
                    within the selected change zone. NDVI is
                    calculated per valid pixel from B08 and B04.
                  </div>
                </div>
              )}

              {/* Detection evidence */}
              <div className="zone-evidence-section">
                <div className="zone-evidence-header">
                  <div>
                    <span className="section-label">
                      DETECTION EVIDENCE
                    </span>

                    <h4>
                      Spectral change evidence
                    </h4>
                  </div>

                  <span className="evidence-score">
                    {formatNumber(
                      selectedChangeEvidence.mean_change_score,
                      3,
                    )}
                  </span>
                </div>

                <div className="evidence-track">
                  <div
                    className="evidence-fill"
                    style={{
                      width: `${Math.min(
                        Number(
                          selectedChangeEvidence
                            .mean_change_score || 0,
                        ) * 100,
                        100,
                      )}%`,
                    }}
                  />
                </div>

                <div className="zone-evidence-grid">
                  <div>
                    <span>
                      SPECTRAL CONTRAST
                    </span>

                    <strong>
                      {formatNumber(
                        selectedChangeEvidence
                          .mean_spectral_distance,
                        3,
                      )}
                    </strong>
                  </div>

                  <div>
                    <span>NDVI MAGNITUDE</span>

                    <strong>
                      {formatNumber(
                        selectedNdviEvidence.absolute_change,
                        3,
                      )}
                    </strong>
                  </div>

                  <div>
                    <span>DETECTION SCORE</span>

                    <strong>
                      {formatNumber(
                        selectedChangeEvidence
                          .mean_change_score,
                        3,
                      )}
                    </strong>
                  </div>
                </div>
              </div>

              {/* Interpretation */}
              <div className="zone-interpretation">
                <div>
                  <span className="section-label">
                    CURRENT INTERPRETATION
                  </span>

                  <p>
                    The numerical detector identifies
                    this region as{" "}
                    <strong>
                      {formatDirection(
                        selectedProperties.direction,
                      )}
                    </strong>{" "}
                    based on its spectral and NDVI
                    differences between the before and
                    after observations.
                  </p>
                </div>

                <div className="interpretation-status">
                  <span className="status-dot" />

                  <span>
                    Numerical evidence verified
                  </span>
                </div>
              </div>
            </section>
          )}

          {/* Lower cards */}
          <div className="lower-grid">
            {/* Detection Summary */}
            <div className="panel-card">
              <div className="panel-header">
                <div>
                  <span className="section-label">
                    DETECTION SUMMARY
                  </span>

                  <h3>Change Distribution</h3>
                </div>

                <BarChart3 size={19} />
              </div>

              <div className="chart-placeholder">
                <div className="chart-bar-row">
                  <span>Vegetation Loss</span>

                  <div className="chart-track">
                    <div
                      className="chart-fill"
                      style={{
                        width: "82%",
                      }}
                    />
                  </div>

                  <strong>25</strong>
                </div>

                <div className="chart-bar-row">
                  <span>Vegetation Gain</span>

                  <div className="chart-track">
                    <div
                      className="chart-fill"
                      style={{
                        width: "16%",
                      }}
                    />
                  </div>

                  <strong>3</strong>
                </div>

                <div className="chart-bar-row">
                  <span>Spectral Change</span>

                  <div className="chart-track">
                    <div
                      className="chart-fill"
                      style={{
                        width: "10%",
                      }}
                    />
                  </div>

                  <strong>3</strong>
                </div>

                <p className="chart-note">
                  Preview only — live chart data will
                  be connected to the analysis API.
                </p>
              </div>
            </div>

            {/* Pipeline */}
            <div className="panel-card">
              <div className="panel-header">
                <div>
                  <span className="section-label">
                    PIPELINE
                  </span>

                  <h3>Analysis Status</h3>
                </div>

                <Activity size={19} />
              </div>

              <div className="pipeline-list">
                <div className="pipeline-step complete">
                  <span />

                  <div>
                    <strong>
                      Sentinel-2 acquisition
                    </strong>

                    <small>Complete</small>
                  </div>
                </div>

                <div className="pipeline-step complete">
                  <span />

                  <div>
                    <strong>
                      Spectral change detection
                    </strong>

                    <small>
                      {zones.length} zones detected
                    </small>
                  </div>
                </div>

                <div className="pipeline-step complete">
                  <span />

                  <div>
                    <strong>
                      Geographic zone extraction
                    </strong>

                    <small>
                      {zones.length} polygons generated
                    </small>
                  </div>
                </div>

                <div className="pipeline-step pending">
                  <span />

                  <div>
                    <strong>
                      Semantic interpretation
                    </strong>

                    <small>
                      Qwen 2.5-VL
                    </small>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;