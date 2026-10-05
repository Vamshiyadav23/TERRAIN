import {
  BarChart3,
  Brain,
  BrainCircuit,
  Download,
  FileText,
  Layers,
  Map,
  MapPinned,
  Menu,
  Settings,
  Satellite,
  ShieldAlert,
  SlidersHorizontal,
  Activity,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import {
  GoogleMap,
  useJsApiLoader
} from "@react-google-maps/api";
import GoogleSatelliteMap from "./components/GoogleSatelliteMap";

import "./App.css";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ||
  "http://127.0.0.1:8000";

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

function AreaSelectorMap({ selectedArea, onAreaChange }) {
  const { isLoaded, loadError } = useJsApiLoader({
    googleMapsApiKey:
      import.meta.env.VITE_GOOGLE_MAPS_API_KEY || "",
  });
  
  const mapRef = useRef(null);
  const circleRef = useRef(null);
  const markerRef = useRef(null);

  const center = {
    lat: Number(selectedArea.lat),
    lng: Number(selectedArea.lng),
  };

  const radius = Number(selectedArea.radius_m);

  /*
   * ---------------------------------------------------------
   * UPDATE THE SINGLE ROI CIRCLE
   * ---------------------------------------------------------
   *
   * IMPORTANT:
   * We create ONE Google Maps Circle.
   * We never create another Circle on every click.
   *
   * When the user changes location/radius, we simply update:
   *
   *   circle.setCenter(...)
   *   circle.setRadius(...)
   *
   * This prevents the multiple-circle problem.
   */
  useEffect(() => {
    if (!mapRef.current) {
      return;
    }

    mapRef.current.panTo({
      lat: Number(selectedArea.lat),
      lng: Number(selectedArea.lng),
    });

  }, [
    selectedArea.lat,
    selectedArea.lng,
  ]);
  useEffect(() => {
    if (!isLoaded || !mapRef.current) {
      return;
    }

    const googleMaps = window.google.maps;

    /*
     * Create the circle ONLY ONCE.
     */
    if (!circleRef.current) {
      circleRef.current = new googleMaps.Circle({
        map: mapRef.current,

        center: center,

        radius: radius,

        fillOpacity: 0.10,

        strokeOpacity: 0.95,

        strokeWeight: 2,

        clickable: false,

        editable: false,

        draggable: false,

        zIndex: 10,
      });
    }

    /*
     * Update the EXISTING circle.
     */
    circleRef.current.setCenter(center);

    circleRef.current.setRadius(radius);

  }, [
    isLoaded,
    center.lat,
    center.lng,
    radius,
  ]);


  /*
   * ---------------------------------------------------------
   * UPDATE THE SINGLE MARKER
   * ---------------------------------------------------------
   */

  useEffect(() => {
    if (!isLoaded || !mapRef.current) {
      return;
    }

    const googleMaps = window.google.maps;

    /*
     * Create marker ONLY ONCE.
     */
    if (!markerRef.current) {
      markerRef.current = new googleMaps.Marker({
        map: mapRef.current,

        position: center,

        zIndex: 100,
      });
    }

    /*
     * Move existing marker.
     */
    markerRef.current.setPosition(center);

  }, [
    isLoaded,
    center.lat,
    center.lng,
  ]);


  /*
   * ---------------------------------------------------------
   * CLEANUP
   * ---------------------------------------------------------
   *
   * When leaving the page, completely remove the overlays.
   */

  useEffect(() => {
    return () => {

      if (circleRef.current) {
        circleRef.current.setMap(null);
        circleRef.current = null;
      }

      if (markerRef.current) {
        markerRef.current.setMap(null);
        markerRef.current = null;
      }

      mapRef.current = null;
    };
  }, []);


  /*
   * ---------------------------------------------------------
   * LOADING / ERROR
   * ---------------------------------------------------------
   */

  if (loadError) {
    return (
      <div className="map-error">
        <strong>
          Google Maps failed to load
        </strong>

        <span>
          {loadError.message}
        </span>
      </div>
    );
  }


  if (!isLoaded) {
    return (
      <div className="map-loading">
        <div className="map-loading-spinner" />

        <span>
          Loading satellite map...
        </span>
      </div>
    );
  }


  /*
   * ---------------------------------------------------------
   * MAP
   * ---------------------------------------------------------
   */

  return (
    <GoogleMap
      center={center}

      zoom={14}

      mapContainerStyle={{
        width: "100%",
        height: "520px",
      }}

      onLoad={(map) => {
        mapRef.current = map;

        /*
         * If the component loaded after the effects,
         * create the overlays immediately.
         */

        const googleMaps = window.google.maps;

        if (!circleRef.current) {
          circleRef.current =
            new googleMaps.Circle({
              map,

              center,

              radius,

              fillOpacity: 0.10,

              strokeOpacity: 0.95,

              strokeWeight: 2,

              clickable: false,

              editable: false,

              draggable: false,

              zIndex: 10,
            });
        }

        if (!markerRef.current) {
          markerRef.current =
            new googleMaps.Marker({
              map,

              position: center,

              zIndex: 100,
            });
        }
      }}

      onUnmount={() => {

        if (circleRef.current) {
          circleRef.current.setMap(null);
          circleRef.current = null;
        }

        if (markerRef.current) {
          markerRef.current.setMap(null);
          markerRef.current = null;
        }

        mapRef.current = null;
      }}

      onClick={(event) => {

        if (!event.latLng) {
          return;
        }

        const newLatitude =
          event.latLng.lat();

        const newLongitude =
          event.latLng.lng();

        /*
         * Update React state.
         *
         * The useEffect above will then move
         * the existing marker and circle.
         */

        onAreaChange({
          lat: newLatitude,

          lng: newLongitude,

          radius_m: radius,
        });
      }}

      options={{
        mapTypeId: "satellite",

        streetViewControl: false,

        fullscreenControl: true,

        mapTypeControl: false,

        clickableIcons: false,

        gestureHandling: "greedy",
      }}
    />
  );
}

function TemporalBarChart({
  data,
  series,
  yDomain,
  showZeroLine = false,
  valueFormatter = (value) => Number(value).toFixed(3),
  emptyMessage = "No chart data available",
}) {
  const validData = Array.isArray(data)
    ? data.filter((item) =>
        series.some((itemSeries) => {
          const value = Number(item?.[itemSeries.key]);
          return Number.isFinite(value);
        })
      )
    : [];

  if (validData.length === 0) {
    return (
      <div
        className="temporal-chart-empty"
        style={{
          minHeight: "320px",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          gap: "10px",
          color: "#6f8799",
        }}
      >
        <BarChart3 size={26} />
        <span>{emptyMessage}</span>
      </div>
    );
  }

  const chartHeight = 320;
  const topPadding = 18;
  const rightPadding = 28;
  const bottomPadding = 54;
  const leftPadding = 62;
  const plotHeight =
    chartHeight - topPadding - bottomPadding;

  const minimumWidth = Math.max(
    900,
    validData.length * (series.length > 1 ? 72 : 62)
  );

  const chartWidth = minimumWidth;
  const plotWidth =
    chartWidth - leftPadding - rightPadding;

  const numericValues = validData.flatMap((item) =>
    series
      .map((itemSeries) => Number(item?.[itemSeries.key]))
      .filter((value) => Number.isFinite(value))
  );

  let minValue =
    yDomain?.[0] ??
    Math.min(0, ...numericValues);

  let maxValue =
    yDomain?.[1] ??
    Math.max(1, ...numericValues);

  if (minValue === maxValue) {
    minValue -= 1;
    maxValue += 1;
  }

  const range = maxValue - minValue;
  const zeroY =
    topPadding +
    ((maxValue - 0) / range) * plotHeight;

  const yScale = (value) =>
    topPadding +
    ((maxValue - value) / range) * plotHeight;

  const tickCount = 5;
  const ticks = Array.from(
    { length: tickCount },
    (_, index) =>
      maxValue -
      (range * index) / (tickCount - 1)
  );

  const groupWidth = plotWidth / validData.length;
  const barGap = 5;
  const groupInnerWidth = Math.min(
    groupWidth * 0.72,
    58
  );
  const barWidth = Math.max(
    8,
    (groupInnerWidth -
      barGap * (series.length - 1)) /
      series.length
  );

  const barRadius = 3;

  const buildBarGeometry = (
    value,
    seriesIndex,
    dataIndex
  ) => {
    if (!Number.isFinite(value)) {
      return null;
    }

    const groupStart =
      leftPadding + dataIndex * groupWidth;

    const groupOffset =
      (groupWidth - groupInnerWidth) / 2;

    const x =
      groupStart +
      groupOffset +
      seriesIndex *
        (barWidth + barGap);

    const valueY = yScale(value);

    if (showZeroLine) {
      const baseline = yScale(0);

      return {
        x,
        y: Math.min(valueY, baseline),
        width: barWidth,
        height: Math.max(
          1,
          Math.abs(baseline - valueY)
        ),
      };
    }

    const baselineValue =
      minValue <= 0 && maxValue >= 0
        ? 0
        : minValue;

    const baseline = yScale(baselineValue);

    return {
      x,
      y: Math.min(valueY, baseline),
      width: barWidth,
      height: Math.max(
        1,
        Math.abs(baseline - valueY)
      ),
    };
  };

  return (
    <div
      className="temporal-chart-scroll"
      style={{
        width: "100%",
        overflowX: "auto",
        overflowY: "hidden",
      }}
    >
      <svg
        className="temporal-svg-chart"
        viewBox={`0 0 ${chartWidth} ${chartHeight}`}
        preserveAspectRatio="none"
        style={{
          display: "block",
          width: "100%",
          minWidth: `${chartWidth}px`,
          height: `${chartHeight}px`,
        }}
        role="img"
        aria-label="Temporal analysis chart"
      >
        <rect
          x="0"
          y="0"
          width={chartWidth}
          height={chartHeight}
          fill="transparent"
        />

        {ticks.map((tick, index) => {
          const y = yScale(tick);

          return (
            <g key={`grid-${index}`}>
              <line
                x1={leftPadding}
                x2={chartWidth - rightPadding}
                y1={y}
                y2={y}
                stroke="#1b2b39"
                strokeDasharray="3 3"
              />
              <text
                x={leftPadding - 10}
                y={y + 4}
                textAnchor="end"
                fill="#8093a4"
                fontSize="10"
              >
                {valueFormatter(tick)}
              </text>
            </g>
          );
        })}

        {showZeroLine &&
          minValue <= 0 &&
          maxValue >= 0 && (
            <line
              x1={leftPadding}
              x2={chartWidth - rightPadding}
              y1={zeroY}
              y2={zeroY}
              stroke="#627889"
              strokeDasharray="4 4"
              strokeWidth="1.2"
            />
          )}

        <line
          x1={leftPadding}
          x2={leftPadding}
          y1={topPadding}
          y2={chartHeight - bottomPadding}
          stroke="#42586a"
        />

        <line
          x1={leftPadding}
          x2={chartWidth - rightPadding}
          y1={chartHeight - bottomPadding}
          y2={chartHeight - bottomPadding}
          stroke="#42586a"
        />

        {validData.map((item, dataIndex) => (
          <g key={`group-${item.zone}-${dataIndex}`}>
            {series.map((itemSeries, seriesIndex) => {
              const value = Number(
                item?.[itemSeries.key]
              );

              const geometry =
                buildBarGeometry(
                  value,
                  seriesIndex,
                  dataIndex
                );

              if (!geometry) {
                return null;
              }

              return (
                <g
                  key={`${item.zone}-${itemSeries.key}`}
                >
                  <rect
                    {...geometry}
                    rx={barRadius}
                    ry={barRadius}
                    fill={itemSeries.fill}
                  >
                    <title>
                      {`${item.zone} · ${
                        itemSeries.name
                      }: ${valueFormatter(value)}`}
                    </title>
                  </rect>
                </g>
              );
            })}

            <text
              x={
                leftPadding +
                dataIndex * groupWidth +
                groupWidth / 2
              }
              y={chartHeight - 24}
              textAnchor="middle"
              fill="#8093a4"
              fontSize="10"
            >
              {item.zone}
            </text>
          </g>
        ))}

        <text
          x="16"
          y={chartHeight / 2}
          transform={`rotate(-90 16 ${
            chartHeight / 2
          })`}
          textAnchor="middle"
          fill="#64788b"
          fontSize="10"
        >
          Value
        </text>

        {series.length > 0 && (
          <g
            transform={`translate(${leftPadding}, ${
              chartHeight - 8
            })`}
          >
            {series.map((itemSeries, index) => {
              const legendX =
                index * 150;

              return (
                <g
                  key={`legend-${itemSeries.key}`}
                  transform={`translate(${legendX}, 0)`}
                >
                  <rect
                    width="11"
                    height="11"
                    rx="2"
                    fill={itemSeries.fill}
                  />
                  <text
                    x="17"
                    y="10"
                    fill="#8ca0b1"
                    fontSize="11"
                  >
                    {itemSeries.name}
                  </text>
                </g>
              );
            })}
          </g>
        )}
      </svg>
    </div>
  );
}

function BeforeAfterSlider({
  before,
  after,
  beforeLabel = "BEFORE",
  afterLabel = "AFTER",
}) {
  const [position, setPosition] = useState(50);

  if (!before || !after) {
    return null;
  }

  return (
    <div className="before-after-slider">
      <div className="before-after-viewport">
        <img
          className="before-after-image before-after-image-after"
          src={after}
          alt="Sentinel-2 after imagery"
          draggable="false"
        />

        <img
          className="before-after-image before-after-image-before"
          src={before}
          alt="Sentinel-2 before imagery"
          draggable="false"
          style={{
            clipPath: `inset(0 ${100 - position}% 0 0)`,
          }}
        />

        <div
          className="before-after-divider"
          style={{ left: `${position}%` }}
          aria-hidden="true"
        >
          <div className="before-after-handle">
            <span>‹</span>
            <span>›</span>
          </div>
        </div>

        <div className="before-after-label before-after-label-left">
          {beforeLabel}
        </div>

        <div className="before-after-label before-after-label-right">
          {afterLabel}
        </div>

        <input
          className="before-after-range"
          type="range"
          min="0"
          max="100"
          step="1"
          value={position}
          onChange={(event) =>
            setPosition(Number(event.target.value))
          }
          aria-label="Before and after image comparison slider"
        />
      </div>

      <div className="before-after-slider-footer">
        <span>BEFORE</span>
        <span>{position}% comparison</span>
        <span>AFTER</span>
      </div>
    </div>
  );
}

function App() {
  const [activePage, setActivePage] = useState("Dashboard");
  const [selectedArea, setSelectedArea] = useState({ lat: 12.971912, lng: 77.614378, radius_m: 1000 });
  const [analysisRunning, setAnalysisRunning] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [analysisSummary, setAnalysisSummary] = useState(null);
  const [analysisError, setAnalysisError] = useState(null);
  const [beforeDate, setBeforeDate] = useState("2020-03-01/2020-04-30");
  const [afterDate, setAfterDate] = useState("2025-12-01/2026-01-31");

  const [zones, setZones] = useState([]);
  const [selectedZone, setSelectedZone] = useState(null);
  const [zonesLoading, setZonesLoading] = useState(true);
  const [zonesError, setZonesError] = useState(null);

  const [zoneEvidence, setZoneEvidence] = useState(null);
  const [evidenceLoading, setEvidenceLoading] = useState(false);
  const [evidenceError, setEvidenceError] = useState(null);

  const [semanticData, setSemanticData] = useState(null);
  const [semanticLoading, setSemanticLoading] = useState(false);
  const [semanticError, setSemanticError] = useState(null);

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
    const loadAnalysisSummary = async () => {
      try {
        const response = await fetch(
          `${API_BASE_URL}/api/v1/analysis/summary`
        );

        if (!response.ok) {
          throw new Error(
            `Summary request failed: ${response.status}`
          );
        }

        const data = await response.json();

        setAnalysisSummary(data);
      } catch (error) {
        console.error(
          "Failed to load analysis summary:",
          error
        );

        setAnalysisSummary(null);
      }
    };

    loadAnalysisSummary();
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

  useEffect(() => {
    if (!selectedZone) {
      setSemanticData(null);
      setSemanticError(null);
      return;
    }

    const regionId =
      selectedZone.properties?.region_id ??
      selectedZone.properties?.regionId ??
      selectedZone.id;

    if (!regionId) {
      setSemanticData(null);
      setSemanticError("Zone ID is unavailable.");
      return;
    }

    let cancelled = false;

    async function fetchSemanticAnalysis() {
      try {
        setSemanticLoading(true);
        setSemanticError(null);
        setSemanticData(null);

        const response = await fetch(
          `${API_BASE_URL}/api/v1/analysis/zones/${regionId}/semantic`,
        );

        if (!response.ok) {
          throw new Error(
            `Semantic analysis failed (${response.status})`,
          );
        }

        const data = await response.json();

        if (!cancelled) {
          setSemanticData(data);
        }
      } catch (error) {
        if (!cancelled) {
          setSemanticError(
            error.message ||
              "Unable to load semantic analysis.",
          );
        }
      } finally {
        if (!cancelled) {
          setSemanticLoading(false);
        }
      }
    }

    fetchSemanticAnalysis();

    return () => {
      cancelled = true;
    };
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

  const formatSemanticClass = (value) => {
    if (!value) return "Unknown";

    return value
      .replaceAll("_", " ")
      .replace(/\b\w/g, (char) =>
        char.toUpperCase(),
      );
  };

  const changeDistribution = zones.reduce(
  (counts, zone) => {
    const direction =
      zone.properties?.direction;

    if (direction === "vegetation_loss") {
      counts.vegetation_loss += 1;
    } else if (direction === "vegetation_gain") {
      counts.vegetation_gain += 1;
    } else if (direction === "spectral_change") {
      counts.spectral_change += 1;
    } else {
      counts.other += 1;
    }

    return counts;
  },
  {
    vegetation_loss: 0,
    vegetation_gain: 0,
    spectral_change: 0,
    other: 0,
  },
);

const totalClassifiedZones =
  changeDistribution.vegetation_loss +
  changeDistribution.vegetation_gain +
  changeDistribution.spectral_change +
  changeDistribution.other;

const getDistributionWidth = (count) => {
  if (totalClassifiedZones === 0) {
    return 0;
  }

  return Math.max(
    (count / totalClassifiedZones) * 100,
    count > 0 ? 4 : 0,
  );
};
const toFiniteNumber = (value) => {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
};

const temporalChartData = zones
  .map((zone) => {
    const p = zone.properties || {};
    const ndvi = p.ndvi_evidence || {};
    const change = p.change_evidence || {};

    const ndviBefore = toFiniteNumber(
      ndvi.before ?? ndvi.ndvi_before ?? p.ndvi_before
    );
    const ndviAfter = toFiniteNumber(
      ndvi.after ?? ndvi.ndvi_after ?? p.ndvi_after
    );
    const ndviChange = toFiniteNumber(
      ndvi.change ?? ndvi.ndvi_change ?? p.ndvi_change
    );
    const spectral = toFiniteNumber(
      change.mean_spectral_distance ??
        change.spectral_contrast ??
        p.mean_spectral_distance ??
        p.spectral_contrast
    );

    return {
      zone: `Z${p.region_id ?? "—"}`,
      regionId: p.region_id,
      ndviBefore,
      ndviAfter,
      ndviChange,
      spectral,
    };
  })
  .filter(
    (item) =>
      item.ndviBefore !== null ||
      item.ndviAfter !== null ||
      item.ndviChange !== null ||
      item.spectral !== null
  );

const temporalNdviComparisonData = temporalChartData.filter(
  (item) =>
    item.ndviBefore !== null || item.ndviAfter !== null
);

const temporalNdviChangeData = temporalChartData.filter(
  (item) => item.ndviChange !== null
);

const temporalSpectralData = temporalChartData.filter(
  (item) => item.spectral !== null
);

const directionChartData = [
  {
    direction: "Vegetation Loss",
    count: changeDistribution.vegetation_loss,
  },
  {
    direction: "Vegetation Gain",
    count: changeDistribution.vegetation_gain,
  },
  {
    direction: "Spectral Change",
    count: changeDistribution.spectral_change,
  },
  {
    direction: "Other / Uncertain",
    count: changeDistribution.other,
  },
];

const temporalDataAvailable =
  temporalChartData.length > 0;

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
                className={`nav-item ${activePage === item.label ? "active" : ""}`}
                onClick={() => setActivePage(item.label)}
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

              <h1>{activePage === "Dashboard" ? "Analysis Dashboard" : activePage}</h1>
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

        {activePage === "Dashboard" && (
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
                  value={Number(selectedArea.lat).toFixed(6)}
                  readOnly
                />
              </div>

              <div className="control-field">
                <label>Longitude</label>

                <input
                  type="text"
                  value={Number(selectedArea.lng).toFixed(6)}
                  readOnly
                />
              </div>

              <div className="control-field">

                <label>Radius</label>

                <div className="input-with-unit">

                  <input
                    type="text"
                    value={selectedArea.radius_m}
                    readOnly
                  />

                  <span>m</span>

                </div>

              </div>

              <div className="control-field">
                <label>Before</label>

                <input
                  type="text"
                  value={beforeDate.split("/")[0]}
                  readOnly
                />
              </div>

              <div className="control-field">
                <label>After</label>

                <input
                  type="text"
                  value={afterDate.split("/")[0]}
                  readOnly
                />
              </div>

              <button
                className="analyze-button"
                onClick={() => setActivePage("Analyze Area")}
              >
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

              <strong>
                {analysisResult?.summary?.changed_percentage != null
                  ? `${Number(
                      analysisResult.summary.changed_percentage
                    ).toFixed(2)}%`
                  : "—"}
              </strong>

              <span className="stat-description">
                Of valid ROI
              </span>
            </div>

            <div className="stat-card">
              <span className="stat-label">
                ANALYSIS RADIUS
              </span>

              <strong>
                {Number(selectedArea.radius_m).toLocaleString()} m
              </strong>

              <span className="stat-description">
                Area of interest
              </span>
            </div>

            <div className="stat-card">
              <span className="stat-label">
                TEMPORAL WINDOW
              </span>

              <strong>
                {(() => {
                  const before = new Date(
                    `${beforeDate.split("/")[0]}T00:00:00`
                  );
                  const after = new Date(
                    `${afterDate.split("/")[0]}T00:00:00`
                  );
                  const years =
                    (after - before) /
                    (365.25 * 24 * 60 * 60 * 1000);
                  return Number.isFinite(years)
                    ? `${years.toFixed(1)} yr`
                    : "—";
                })()}
              </strong>

              <span className="stat-description">
                {beforeDate.split("/")[0].slice(0, 4)} →{" "}
                {afterDate.split("/")[0].slice(0, 4)}
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

                <button>
                  Satellite
                </button>

                <button>
                  NDVI
                </button>

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
                  <button
                    className="analyze-button"
                    onClick={() => setActivePage("Explainable AI")}
                  >
                    <Brain size={17} />
                    Explain Zone
                  </button>

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

              {/* =====================================================
                  AI SEMANTIC INTERPRETATION
                  ===================================================== */}
              <div className="semantic-panel">

                <div className="semantic-panel-header">

                  <div className="semantic-panel-title">

                    <Brain size={20} />

                    <div>

                      <p>
                        AI SEMANTIC INTERPRETATION
                      </p>

                      <h3>
                        What changed?
                      </h3>

                    </div>

                  </div>

                  {semanticData?.semantic_interpretation && (
                    <span className="semantic-model-badge">
                      Qwen 2.5-VL
                    </span>
                  )}

                </div>

                {/* Loading */}
                {semanticLoading && (
                  <div className="semantic-loading">

                    <div className="semantic-skeleton short" />

                    <div className="semantic-skeleton medium" />

                    <div className="semantic-skeleton" />

                  </div>
                )}

                {/* Error */}
                {semanticError && !semanticLoading && (
                  <div className="semantic-error">
                    {semanticError}
                  </div>
                )}

                {/* Results */}
                {semanticData?.semantic_interpretation &&
                  !semanticLoading && (
                    <>

                      {/* Semantic metrics */}
                      <div className="semantic-metrics">

                        {/* Classification */}
                        <div className="semantic-metric">

                          <div className="semantic-metric-label">
                            Classification
                          </div>

                          <div className="semantic-classification">

                            <span className="semantic-classification-dot" />

                            <span className="semantic-metric-value">
                              {formatSemanticClass(
                                semanticData
                                  .semantic_interpretation
                                  .classification,
                              )}
                            </span>

                          </div>

                        </div>

                        {/* Confidence */}
                        <div className="semantic-metric">

                          <div className="semantic-metric-label">
                            Semantic Confidence
                          </div>

                          <div className="semantic-metric-value">

                            {Math.round(
                              semanticData
                                .semantic_interpretation
                                .confidence * 100,
                            )}
                            %

                          </div>

                        </div>

                        {/* Spectral consistency */}
                        <div className="semantic-metric">

                          <div className="semantic-metric-label">
                            Spectral Consistency
                          </div>

                          <div className="semantic-metric-value success">

                            {formatSemanticClass(
                              semanticData
                                .semantic_interpretation
                                .spectral_consistency,
                            )}

                          </div>

                        </div>

                        {/* Human review */}
                        <div className="semantic-metric">

                          <div className="semantic-metric-label">
                            Human Review
                          </div>

                          <div
                            className={`semantic-review ${
                              semanticData
                                .semantic_interpretation
                                .needs_review
                                ? "required"
                                : "ok"
                            }`}
                          >
                            {semanticData
                              .semantic_interpretation
                              .needs_review
                              ? "⚠ REVIEW REQUIRED"
                              : "✓ NOT REQUIRED"}
                          </div>

                        </div>

                      </div>

                      {/* Interpretation */}
                      <div className="semantic-section">

                        <div className="semantic-section-title">
                          Interpretation
                        </div>

                        <p className="semantic-summary">
                          {
                            semanticData
                              .semantic_interpretation
                              .summary
                          }
                        </p>

                      </div>

                      {/* Evidence */}
                      {semanticData
                        .semantic_interpretation
                        .visual_evidence
                        ?.length > 0 && (
                        <div className="semantic-section">

                          <div className="semantic-section-title">
                            Evidence
                          </div>

                          <ul className="semantic-evidence-list">

                            {semanticData
                              .semantic_interpretation
                              .visual_evidence
                              .map(
                                (item, index) => (
                                  <li key={index}>
                                    {item}
                                  </li>
                                ),
                              )}

                          </ul>

                        </div>
                      )}

                      {/* Alternative explanations */}
                      {semanticData
                        .semantic_interpretation
                        .alternative_explanations
                        ?.length > 0 && (
                        <div className="semantic-section">

                          <div className="semantic-section-title">
                            Alternative Explanations
                          </div>

                          <ul className="semantic-alternative-list">

                            {semanticData
                              .semantic_interpretation
                              .alternative_explanations
                              .map(
                                (item, index) => (
                                  <li key={index}>
                                    {item}
                                  </li>
                                ),
                              )}

                          </ul>

                        </div>
                      )}

                      {/* Review warning */}
                      {semanticData
                        .semantic_interpretation
                        .needs_review && (
                        <div className="semantic-review-warning">

                          <strong>
                            Manual review recommended
                          </strong>

                          <span>
                            The numerical evidence indicates
                            change, but the semantic
                            interpretation remains ambiguous.
                          </span>

                        </div>
                      )}

                    </>
                  )}

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

                    <span>
                      {evidenceError}
                    </span>

                  </div>
                ) : zoneEvidence?.images?.before &&
                  zoneEvidence?.images?.after ? (
                  <div className="evidence-comparison-layout">

                    <BeforeAfterSlider
                      before={zoneEvidence.images.before}
                      after={zoneEvidence.images.after}
                      beforeLabel="BEFORE"
                      afterLabel="AFTER"
                    />

                    {zoneEvidence.images.overlay && (
                      <div className="evidence-image-card evidence-overlay-card">

                        <div className="evidence-image-header">

                          <strong>
                            CHANGE ZONE
                          </strong>

                          <span>
                            Detected region
                          </span>

                        </div>

                        <img
                          src={zoneEvidence.images.overlay}
                          alt="Detected change zone overlay"
                        />

                      </div>
                    )}

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

                      <span>
                        NDVI BEFORE
                      </span>

                      <strong>
                        {formatNumber(
                          zoneEvidence
                            .ndvi_profile
                            ?.before,
                          3,
                        )}
                      </strong>

                    </div>

                    <div>

                      <span>
                        NDVI AFTER
                      </span>

                      <strong>
                        {formatNumber(
                          zoneEvidence
                            .ndvi_profile
                            ?.after,
                          3,
                        )}
                      </strong>

                    </div>

                    <div>

                      <span>
                        NDVI CHANGE
                      </span>

                      <strong
                        className={
                          Number(
                            zoneEvidence
                              .ndvi_profile
                              ?.change,
                          ) < 0
                            ? "metric-negative"
                            : "metric-positive"
                        }
                      >
                        {Number(
                          zoneEvidence
                            .ndvi_profile
                            ?.change,
                        ) > 0
                          ? "+"
                          : ""}

                        {formatNumber(
                          zoneEvidence
                            .ndvi_profile
                            ?.change,
                          3,
                        )}
                      </strong>

                    </div>

                    <div>

                      <span>
                        SPECTRAL CONTRAST
                      </span>

                      <strong>
                        {formatNumber(
                          zoneEvidence
                            .spectral_profile
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
                      selectedChangeEvidence
                        .mean_change_score,
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

                    <span>
                      NDVI MAGNITUDE
                    </span>

                    <strong>
                      {formatNumber(
                        selectedNdviEvidence
                          .absolute_change,
                        3,
                      )}
                    </strong>

                  </div>

                  <div>

                    <span>
                      DETECTION SCORE
                    </span>

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

                  <h3>
                    Change Distribution
                  </h3>

                </div>

                <BarChart3 size={19} />

              </div>

              <div className="chart-placeholder">

              {/* Vegetation Loss */}
              <div className="chart-bar-row">

                <span>
                  Vegetation Loss
                </span>

                <div className="chart-track">

                  <div
                    className="chart-fill"
                    style={{
                      width: `${getDistributionWidth(
                        changeDistribution.vegetation_loss,
                      )}%`,
                    }}
                  />

                </div>

                <strong>
                  {changeDistribution.vegetation_loss}
                </strong>

              </div>

              {/* Vegetation Gain */}
              <div className="chart-bar-row">

                <span>
                  Vegetation Gain
                </span>

                <div className="chart-track">

                  <div
                    className="chart-fill"
                    style={{
                      width: `${getDistributionWidth(
                        changeDistribution.vegetation_gain,
                      )}%`,
                    }}
                  />

                </div>

                <strong>
                  {changeDistribution.vegetation_gain}
                </strong>

              </div>

              {/* Spectral Change */}
              <div className="chart-bar-row">

                <span>
                  Spectral Change
                </span>

                <div className="chart-track">

                  <div
                    className="chart-fill"
                    style={{
                      width: `${getDistributionWidth(
                        changeDistribution.spectral_change,
                      )}%`,
                    }}
                  />

                </div>

                <strong>
                  {changeDistribution.spectral_change}
                </strong>

              </div>

              {/* Other / uncertain */}
              {changeDistribution.other > 0 && (
                <div className="chart-bar-row">

                  <span>
                    Other / Uncertain
                  </span>

                  <div className="chart-track">

                    <div
                      className="chart-fill"
                      style={{
                        width: `${getDistributionWidth(
                          changeDistribution.other,
                        )}%`,
                      }}
                    />

                  </div>

                  <strong>
                    {changeDistribution.other}
                  </strong>

                </div>
              )}

              <p className="chart-note">
                Distribution calculated from {zones.length} detected
                change zones.
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

                  <h3>
                    Analysis Status
                  </h3>

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

                    <small>
                      Complete
                    </small>

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

                <div
                  className={`pipeline-step ${
                    semanticData
                      ? "complete"
                      : "pending"
                  }`}
                >

                  <span />

                  <div>

                    <strong>
                      Semantic interpretation
                    </strong>

                    <small>
                      {semanticLoading
                        ? "Qwen 2.5-VL analyzing..."
                        : semanticData
                          ? "Complete"
                          : "Select a zone"}
                    </small>

                  </div>

                </div>

              </div>

            </div>

          </div>

        </section>
        )}

        {activePage === "Analyze Area" && (
          <section className="dashboard-content page-section">
            <div className="page-section-header">
              <div>
                <span className="section-label">AREA SELECTION</span>
                <h2>Select an area to analyze</h2>
                <p>Click on the satellite map to choose the analysis center, then select the radius and temporal windows.</p>
              </div>
              <span className="ready-badge">SATELLITE ROI</span>
            </div>

            <div className="map-card area-selector-card">
              <AreaSelectorMap selectedArea={selectedArea} onAreaChange={setSelectedArea} />
            </div>

            <div className="control-card">
              <div className="control-grid">
                <div className="control-field">
                  <label>Selected center</label>
                  <input readOnly value={`${selectedArea.lat.toFixed(6)}, ${selectedArea.lng.toFixed(6)}`} />
                </div>
                <div className="control-field">
                  <label>Radius</label>
                  <select value={selectedArea.radius_m} onChange={(e) => setSelectedArea({ ...selectedArea, radius_m: Number(e.target.value) })}>
                    <option value={250}>250 m</option>
                    <option value={500}>500 m</option>
                    <option value={1000}>1 km</option>
                    <option value={1500}>1.5 km</option>
                    <option value={2000}>2 km</option>
                    <option value={3000}>3 km</option>
                    <option value={5000}>5 km</option>
                  </select>
                </div>
                <div className="control-field">
                  <label>Before period</label>
                  <input value={beforeDate} onChange={(e) => setBeforeDate(e.target.value)} placeholder="2020-03-01/2020-04-30" />
                </div>
                <div className="control-field">
                  <label>After period</label>
                  <input value={afterDate} onChange={(e) => setAfterDate(e.target.value)} placeholder="2025-12-01/2026-01-31" />
                </div>
                <button className="analyze-button" disabled={analysisRunning} onClick={async () => {
                  try {
                    setAnalysisRunning(true);
                    setAnalysisError(null);
                    setAnalysisResult(null);
                    const response = await fetch(`${API_BASE_URL}/analyze-area`, {
                      method: "POST",
                      headers: { "Content-Type": "application/json" },
                      body: JSON.stringify({
                        latitude: selectedArea.lat,
                        longitude: selectedArea.lng,
                        radius_m: selectedArea.radius_m,
                        before_date: beforeDate,
                        after_date: afterDate,
                      }),
                    });
                    const data = await response.json();
                    if (!response.ok || !data.success) throw new Error(data.detail || `Analysis failed (${response.status})`);
                    setAnalysisResult(data);
                    setAnalysisSummary(data.summary);
                    const zonesResponse = await fetch(`${API_BASE_URL}/api/v1/analysis/zones?t=${Date.now()}`);
                    if (!zonesResponse.ok) throw new Error(`Zone refresh failed (${zonesResponse.status})`);
                    const zonesData = await zonesResponse.json();
                    setZones(zonesData.features || []);
                    setSelectedZone(null);
                    setActivePage("Dashboard");
                  } catch (error) {
                    console.error("Satellite analysis failed:", error);
                    setAnalysisError(error.message || "Satellite analysis failed");
                  } finally {
                    setAnalysisRunning(false);
                  }
                }}>
                  <Satellite size={18} />
                  {analysisRunning ? "Running Sentinel-2 Analysis..." : "Analyze Selected Area"}
                </button>
              </div>

              <p className="area-selector-hint">Click the map to move the ROI center. Coordinates are calculated automatically.</p>

              {analysisRunning && <div className="map-loading"><div className="map-loading-spinner" /><span>Acquiring Sentinel-2 imagery and running the spectral change engine. This may take a few minutes.</span></div>}
              {analysisError && <div className="map-error"><strong>Satellite analysis failed</strong><span>{analysisError}</span></div>}
              {analysisResult?.summary && <div className="stats-grid analysis-result-grid">
                <div className="stat-card"><span className="stat-label">CHANGE ZONES</span><strong>{analysisResult.summary.zone_count}</strong><span className="stat-description">Newly detected zones</span></div>
                <div className="stat-card"><span className="stat-label">CHANGED PIXELS</span><strong>{analysisResult.summary.changed_pixels.toLocaleString()}</strong><span className="stat-description">Within valid ROI</span></div>
                <div className="stat-card"><span className="stat-label">CHANGED AREA</span><strong>{Number(analysisResult.summary.changed_percentage).toFixed(2)}%</strong><span className="stat-description">Of valid ROI</span></div>
                <div className="stat-card"><span className="stat-label">THRESHOLD</span><strong>{Number(analysisResult.summary.threshold).toFixed(3)}</strong><span className="stat-description">Adaptive detector</span></div>
              </div>}
            </div>
          </section>
        )}

        {activePage === "Change Map" && (
          <section className="dashboard-content page-section">
            <div className="page-section-header"><div><span className="section-label">GEOSPATIAL INTELLIGENCE</span><h2>Change Map</h2><p>Verified change-zone polygons generated by the spectral engine.</p></div></div>
            <div className="map-card page-map-card"><GoogleSatelliteMap zones={zones} onZoneSelect={setSelectedZone} /></div>
          </section>
        )}

        {activePage === "Change Zones" && (
          <section className="dashboard-content page-section">
            <div className="page-section-header"><div><span className="section-label">ZONE INVENTORY</span><h2>Detected Change Zones</h2><p>{zones.length} geographic zones currently loaded from the verified GeoJSON output.</p></div></div>
            <div className="panel-card zone-table-card">
              <div className="zone-table-header"><span>ZONE</span><span>DIRECTION</span><span>SEVERITY</span><span>AREA</span><span>CONFIDENCE</span></div>
              {zones.map((zone) => { const p = zone.properties || {}; return <button key={p.region_id} className="zone-table-row" onClick={() => { setSelectedZone(zone); setActivePage("Dashboard"); }}><strong>Zone {p.region_id}</strong><span>{formatDirection(p.direction)}</span><span>{formatSeverity(p.severity)}</span><span>{formatArea(p.area_m2)}</span><span>{p.confidence != null ? `${(Number(p.confidence) * 100).toFixed(1)}%` : "—"}</span></button>; })}
            </div>
          </section>
        )}

        {activePage === "Temporal Analysis" && (
          <section className="dashboard-content page-section temporal-page">
            <div className="page-section-header">
              <div>
                <span className="section-label">TEMPORAL ANALYSIS</span>
                <h2>Before / After Comparison</h2>
                <p>
                  Quantitative temporal evidence derived from the current
                  Sentinel-2 change detection results.
                </p>
              </div>

              <span className="ready-badge">
                {temporalDataAvailable ? "LIVE ANALYSIS DATA" : "NO DATA"}
              </span>
            </div>

            <div className="temporal-timeline-card">
              <div className="temporal-timeline">
                <div className="temporal-line" />

                <div className="temporal-point">
                  <span className="temporal-point-marker" />
                  <div>
                    <div className="temporal-point-label">BEFORE</div>
                    <div className="temporal-point-date">
                      {beforeDate.split("/")[0] || "—"}
                    </div>
                    <div className="temporal-point-subtitle">
                      Sentinel-2 acquisition window
                    </div>
                  </div>
                </div>

                <div className="temporal-duration">
                  TEMPORAL
                  <br />
                  COMPARISON
                </div>

                <div className="temporal-point after">
                  <div>
                    <div className="temporal-point-label">AFTER</div>
                    <div className="temporal-point-date">
                      {afterDate.split("/")[0] || "—"}
                    </div>
                    <div className="temporal-point-subtitle">
                      Sentinel-2 acquisition window
                    </div>
                  </div>
                  <span className="temporal-point-marker" />
                </div>
              </div>
            </div>

            <div className="temporal-metrics-grid">
              <div className="temporal-metric">
                <div className="temporal-metric-label">DETECTED ZONES</div>
                <strong className="temporal-metric-value">
                  {zones.length}
                </strong>
                <span className="temporal-metric-subtitle">
                  Current detection
                </span>
              </div>

              <div className="temporal-metric">
                <div className="temporal-metric-label">VEGETATION LOSS</div>
                <strong className="temporal-metric-value negative">
                  {zones.filter(
                    (z) => z.properties?.direction === "vegetation_loss"
                  ).length}
                </strong>
                <span className="temporal-metric-subtitle">
                  Zones with negative NDVI direction
                </span>
              </div>

              <div className="temporal-metric">
                <div className="temporal-metric-label">VEGETATION GAIN</div>
                <strong className="temporal-metric-value positive">
                  {zones.filter(
                    (z) => z.properties?.direction === "vegetation_gain"
                  ).length}
                </strong>
                <span className="temporal-metric-subtitle">
                  Zones with positive NDVI direction
                </span>
              </div>

              <div className="temporal-metric">
                <div className="temporal-metric-label">SPECTRAL CHANGE</div>
                <strong className="temporal-metric-value">
                  {zones.filter(
                    (z) => z.properties?.direction === "spectral_change"
                  ).length}
                </strong>
                <span className="temporal-metric-subtitle">
                  Zones requiring spectral interpretation
                </span>
              </div>
            </div>

            {!temporalDataAvailable ? (
              <div className="empty-page-card">
                <BarChart3 size={32} />
                <h2>No temporal evidence available</h2>
                <p>
                  Run a satellite analysis first. These charts are populated
                  only from the verified change-zone evidence returned by the
                  backend.
                </p>
                <button
                  className="analyze-button"
                  onClick={() => setActivePage("Analyze Area")}
                >
                  <Satellite size={18} />
                  Analyze Area
                </button>
              </div>
            ) : (
              <>
                <div className="temporal-analysis-card">
                  <div className="page-card-header">
                    <div>
                      <span className="section-label">VEGETATION RESPONSE</span>
                      <h3>NDVI Before vs After by Zone</h3>
                    </div>
                    <Activity size={18} />
                  </div>

                  <TemporalBarChart
                    data={temporalNdviComparisonData}
                    series={[
                      {
                        key: "ndviBefore",
                        name: "Before NDVI",
                        fill: "#4fb7d8",
                      },
                      {
                        key: "ndviAfter",
                        name: "After NDVI",
                        fill: "#55d69b",
                      },
                    ]}
                    yDomain={[-1, 1]}
                  />

                  <div className="temporal-data-note">
                    NDVI is calculated from the red and near-infrared
                    Sentinel-2 bands. Values are derived from detected zone
                    evidence rather than hardcoded dashboard values.
                  </div>
                </div>

                <div className="temporal-analysis-card">
                  <div className="page-card-header">
                    <div>
                      <span className="section-label">TEMPORAL DELTA</span>
                      <h3>NDVI Change by Zone</h3>
                    </div>
                    <BarChart3 size={18} />
                  </div>

                  <TemporalBarChart
                    data={temporalNdviChangeData}
                    series={[
                      {
                        key: "ndviChange",
                        name: "NDVI Change",
                        fill: "#65d7ff",
                      },
                    ]}
                    yDomain={[-1, 1]}
                    showZeroLine
                  />

                  <div className="temporal-data-note">
                    Negative values indicate a decrease in NDVI between the
                    acquisition periods; positive values indicate an increase.
                  </div>
                </div>

                <div className="temporal-analysis-card">
                  <div className="page-card-header">
                    <div>
                      <span className="section-label">SPECTRAL RESPONSE</span>
                      <h3>Spectral Contrast by Zone</h3>
                    </div>
                    <Layers size={18} />
                  </div>

                  <TemporalBarChart
                    data={temporalSpectralData}
                    series={[
                      {
                        key: "spectral",
                        name: "Spectral Contrast",
                        fill: "#8ba9ff",
                      },
                    ]}
                    valueFormatter={(value) =>
                      Number(value).toFixed(3)
                    }
                  />

                  <div className="temporal-data-note">
                    Spectral contrast represents the measured physical
                    spectral difference between the before and after
                    observations within each detected zone.
                  </div>
                </div>

                <div className="temporal-analysis-card">
                  <div className="page-card-header">
                    <div>
                      <span className="section-label">CHANGE DISTRIBUTION</span>
                      <h3>Detected Change Direction</h3>
                    </div>
                    <BarChart3 size={18} />
                  </div>

                  <TemporalBarChart
                    data={directionChartData}
                    series={[
                      {
                        key: "count",
                        name: "Detected Zones",
                        fill: "#55d69b",
                      },
                    ]}
                    yDomain={[0, Math.max(
                      1,
                      ...directionChartData.map(
                        (item) => Number(item.count) || 0
                      )
                    )]}
                    valueFormatter={(value) =>
                      String(Math.round(Number(value)))
                    }
                  />

                  <div className="temporal-data-note">
                    Counts are calculated directly from the direction assigned
                    to each detected geographic zone.
                  </div>
                </div>
              </>
            )}
          </section>
        )}

        {activePage === "Explainable AI" && (
  <section className="dashboard-content page-section">

    {!selectedZone ? (

      <div className="empty-page-card">

        <Brain size={32} />

        <h2>Explainable AI</h2>

        <p>
          Select a detected change zone from the Dashboard,
          Change Map, or Change Zones page to inspect its
          numerical and semantic evidence.
        </p>

        <button
          className="analyze-button"
          onClick={() => setActivePage("Dashboard")}
        >
          Return to Dashboard
        </button>

      </div>

    ) : (

      <>

        {/* =====================================================
            XAI HEADER
            ===================================================== */}

        <div className="page-section-header">

          <div>

            <span className="section-label">
              EXPLAINABLE AI
            </span>

            <h2>
              Zone {selectedProperties.region_id ?? "—"} Intelligence
            </h2>

            <p>
              Numerical remote-sensing evidence combined with
              visual interpretation and Qwen semantic analysis.
            </p>

          </div>

          <div className="zone-header-actions">

            <span
              className={`severity-badge severity-${
                selectedProperties.severity || "unknown"
              }`}
            >
              {formatSeverity(selectedProperties.severity)}
            </span>

            <button
              className="analyze-button"
              onClick={() => setActivePage("Dashboard")}
            >
              Return to Dashboard
            </button>

          </div>

        </div>


        {/* =====================================================
            ZONE OVERVIEW
            ===================================================== */}

        <div className="zone-intelligence-card">

          <div className="zone-intelligence-header">

            <div>

              <span className="section-label">
                DETECTED CHANGE ZONE
              </span>

              <h3>
                Zone {selectedProperties.region_id ?? "—"}
              </h3>

              <span className="zone-direction">
                {formatDirection(selectedProperties.direction)}
              </span>

            </div>

          </div>


          <div className="zone-metrics-grid">

            <div className="zone-metric">
              <span>CHANGED AREA</span>

              <strong>
                {formatArea(selectedProperties.area_m2)}
              </strong>
            </div>


            <div className="zone-metric">
              <span>PIXELS</span>

              <strong>
                {selectedProperties.pixel_count ?? "—"}
              </strong>
            </div>


            <div className="zone-metric">
              <span>DETECTOR CONFIDENCE</span>

              <strong>
                {selectedProperties.confidence != null
                  ? `${(
                      Number(selectedProperties.confidence) * 100
                    ).toFixed(1)}%`
                  : "—"}
              </strong>
            </div>


            <div className="zone-metric">
              <span>SEVERITY</span>

              <strong>
                {formatSeverity(selectedProperties.severity)}
              </strong>
            </div>


            <div className="zone-metric">
              <span>NDVI CHANGE</span>

              <strong
                className={
                  Number(selectedNdviEvidence.change) < 0
                    ? "metric-negative"
                    : "metric-positive"
                }
              >
                {selectedNdviEvidence.change != null
                  ? `${
                      Number(selectedNdviEvidence.change) > 0
                        ? "+"
                        : ""
                    }${Number(
                      selectedNdviEvidence.change
                    ).toFixed(3)}`
                  : "—"}
              </strong>
            </div>


            <div className="zone-metric">
              <span>SPECTRAL CONTRAST</span>

              <strong>
                {selectedChangeEvidence.mean_spectral_distance != null
                  ? Number(
                      selectedChangeEvidence.mean_spectral_distance
                    ).toFixed(3)
                  : "—"}
              </strong>
            </div>

          </div>

        </div>


        {/* =====================================================
            TEMPORAL VISUAL EVIDENCE
            ===================================================== */}

        <div className="zone-intelligence-card">

          <div className="zone-evidence-title">

            <div>

              <span className="section-label">
                VISUAL EVIDENCE
              </span>

              <h4>
                Before / After Satellite Comparison
              </h4>

            </div>

            <span className="evidence-source">
              Sentinel-2 · RGB
            </span>

          </div>


          {evidenceLoading && (

            <div className="evidence-loading">

              <div className="map-loading-spinner" />

              <span>
                Generating zone evidence...
              </span>

            </div>

          )}


          {evidenceError && !evidenceLoading && (

            <div className="evidence-error">

              <strong>
                Unable to generate visual evidence
              </strong>

              <span>
                {evidenceError}
              </span>

            </div>

          )}


          {zoneEvidence?.images && !evidenceLoading && (

            <div className="evidence-image-grid">

              <div className="evidence-image-card">

                <div className="evidence-image-header">

                  <strong>
                    BEFORE
                  </strong>

                  <span>
                    {beforeDate}
                  </span>

                </div>

                <img
                  src={zoneEvidence.images.before}
                  alt="Sentinel-2 before imagery for selected zone"
                />

              </div>


              <div className="evidence-image-card">

                <div className="evidence-image-header">

                  <strong>
                    AFTER
                  </strong>

                  <span>
                    {afterDate}
                  </span>

                </div>

                <img
                  src={zoneEvidence.images.after}
                  alt="Sentinel-2 after imagery for selected zone"
                />

              </div>


              <div className="evidence-image-card">

                <div className="evidence-image-header">

                  <strong>
                    CHANGE OVERLAY
                  </strong>

                  <span>
                    Detected region
                  </span>

                </div>

                <img
                  src={zoneEvidence.images.overlay}
                  alt="Detected change zone overlay"
                />

              </div>

            </div>

          )}


          <div className="evidence-disclaimer">

            RGB imagery is used for visual interpretation.
            Change detection is calculated from multispectral
            Sentinel-2 measurements.

          </div>

        </div>


        {/* =====================================================
            MULTISPECTRAL EVIDENCE
            ===================================================== */}

        {zoneEvidence?.band_profile && (

          <div className="zone-intelligence-card">

            <div className="zone-evidence-title">

              <div>

                <span className="section-label">
                  MULTISPECTRAL EVIDENCE
                </span>

                <h4>
                  Spectral response inside the change zone
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
                  zoneEvidence.band_profile[band] || {};

                const change =
                  Number(values.change);

                return (

                  <div
                    className="band-table-row"
                    key={band}
                  >

                    <div>

                      <strong>
                        {band}
                      </strong>

                      <span>
                        {label}
                      </span>

                    </div>

                    <span>
                      {formatNumber(values.before, 4)}
                    </span>

                    <span>
                      {formatNumber(values.after, 4)}
                    </span>

                    <span
                      className={
                        change < 0
                          ? "metric-negative"
                          : "metric-positive"
                      }
                    >
                      {change > 0 ? "+" : ""}
                      {formatNumber(change, 4)}
                    </span>

                  </div>

                );

              })}

            </div>


            <div className="spectral-summary-grid">

              <div>

                <span>
                  NDVI BEFORE
                </span>

                <strong>
                  {formatNumber(
                    zoneEvidence.ndvi_profile?.before,
                    3
                  )}
                </strong>

              </div>


              <div>

                <span>
                  NDVI AFTER
                </span>

                <strong>
                  {formatNumber(
                    zoneEvidence.ndvi_profile?.after,
                    3
                  )}
                </strong>

              </div>


              <div>

                <span>
                  NDVI CHANGE
                </span>

                <strong
                  className={
                    Number(
                      zoneEvidence.ndvi_profile?.change
                    ) < 0
                      ? "metric-negative"
                      : "metric-positive"
                  }
                >
                  {Number(
                    zoneEvidence.ndvi_profile?.change
                  ) > 0
                    ? "+"
                    : ""}

                  {formatNumber(
                    zoneEvidence.ndvi_profile?.change,
                    3
                  )}
                </strong>

              </div>


              <div>

                <span>
                  SPECTRAL CONTRAST
                </span>

                <strong>
                  {formatNumber(
                    zoneEvidence.spectral_profile?.mean,
                    3
                  )}
                </strong>

              </div>

            </div>


            <div className="evidence-disclaimer">

              Band values represent mean surface reflectance
              within the detected zone. NDVI is calculated from
              B08 and B04.

            </div>

          </div>

        )}


        {/* =====================================================
            DETECTION EVIDENCE
            ===================================================== */}

        <div className="zone-intelligence-card">

          <div className="zone-evidence-title">

            <div>

              <span className="section-label">
                DETECTION EVIDENCE
              </span>

              <h4>
                Why was this zone detected?
              </h4>

            </div>

            <span className="evidence-source">
              Numerical detector
            </span>

          </div>


          <div className="zone-metrics-grid">

            <div className="zone-metric">

              <span>
                NDVI BEFORE
              </span>

              <strong>
                {formatNumber(
                  selectedNdviEvidence.before,
                  3
                )}
              </strong>

            </div>


            <div className="zone-metric">

              <span>
                NDVI AFTER
              </span>

              <strong>
                {formatNumber(
                  selectedNdviEvidence.after,
                  3
                )}
              </strong>

            </div>


            <div className="zone-metric">

              <span>
                NDVI DELTA
              </span>

              <strong
                className={
                  Number(selectedNdviEvidence.change) < 0
                    ? "metric-negative"
                    : "metric-positive"
                }
              >
                {selectedNdviEvidence.change != null
                  ? `${
                      Number(selectedNdviEvidence.change) > 0
                        ? "+"
                        : ""
                    }${Number(
                      selectedNdviEvidence.change
                    ).toFixed(3)}`
                  : "—"}
              </strong>

            </div>


            <div className="zone-metric">

              <span>
                SPECTRAL DISTANCE
              </span>

              <strong>
                {formatNumber(
                  selectedChangeEvidence.mean_spectral_distance,
                  3
                )}
              </strong>

            </div>

          </div>

        </div>


        {/* =====================================================
            QWEN SEMANTIC INTERPRETATION
            ===================================================== */}

        <div className="semantic-panel">

          <div className="semantic-panel-header">

            <div className="semantic-panel-title">

              <Brain size={21} />

              <div>

                <p>
                  AI SEMANTIC INTERPRETATION
                </p>

                <h3>
                  What does the detected change mean?
                </h3>

              </div>

            </div>


            <span className="semantic-model-badge">
              Qwen 2.5-VL
            </span>

          </div>


          {semanticLoading && (

            <div className="semantic-loading">

              <div className="semantic-skeleton short" />

              <div className="semantic-skeleton medium" />

              <div className="semantic-skeleton" />

            </div>

          )}


          {semanticError && !semanticLoading && (

            <div className="semantic-error">

              {semanticError}

            </div>

          )}


          {semanticData?.semantic_interpretation &&
            !semanticLoading && (

            <>

              <div className="semantic-metrics">

                <div className="semantic-metric">

                  <div className="semantic-metric-label">
                    Classification
                  </div>

                  <div className="semantic-classification">

                    <span className="semantic-classification-dot" />

                    <span className="semantic-metric-value">
                      {formatSemanticClass(
                        semanticData.semantic_interpretation
                          .classification
                      )}
                    </span>

                  </div>

                </div>


                <div className="semantic-metric">

                  <div className="semantic-metric-label">
                    Semantic Confidence
                  </div>

                  <div className="semantic-metric-value">

                    {Math.round(
                      semanticData.semantic_interpretation
                        .confidence * 100
                    )}
                    %

                  </div>

                </div>


                <div className="semantic-metric">

                  <div className="semantic-metric-label">
                    Spectral Consistency
                  </div>

                  <div className="semantic-metric-value success">

                    {formatSemanticClass(
                      semanticData.semantic_interpretation
                        .spectral_consistency
                    )}

                  </div>

                </div>


                <div className="semantic-metric">

                  <div className="semantic-metric-label">
                    Review Status
                  </div>

                  <div
                    className={`semantic-review ${
                      semanticData.semantic_interpretation
                        .needs_review
                        ? "required"
                        : "ok"
                    }`}
                  >
                    {semanticData.semantic_interpretation
                      .needs_review
                      ? "⚠ REVIEW REQUIRED"
                      : "✓ NOT REQUIRED"}
                  </div>

                </div>

              </div>


              <div className="semantic-section">

                <div className="semantic-section-title">
                  Interpretation
                </div>

                <p className="semantic-summary">
                  {
                    semanticData.semantic_interpretation
                      .summary
                  }
                </p>

              </div>


              {semanticData.semantic_interpretation
                .visual_evidence?.length > 0 && (

                <div className="semantic-section">

                  <div className="semantic-section-title">
                    Visual Evidence
                  </div>

                  <ul className="semantic-evidence-list">

                    {semanticData.semantic_interpretation
                      .visual_evidence
                      .map((item, index) => (

                        <li key={index}>
                          {item}
                        </li>

                      ))}

                  </ul>

                </div>

              )}


              {semanticData.semantic_interpretation
                .alternative_explanations?.length > 0 && (

                <div className="semantic-section">

                  <div className="semantic-section-title">
                    Alternative Explanations
                  </div>

                  <ul className="semantic-alternative-list">

                    {semanticData.semantic_interpretation
                      .alternative_explanations
                      .map((item, index) => (

                        <li key={index}>
                          {item}
                        </li>

                      ))}

                  </ul>

                </div>

              )}


              {semanticData.semantic_interpretation
                .needs_review && (

                <div className="semantic-review-warning">

                  <strong>
                    Manual review recommended
                  </strong>

                  <span>
                    The numerical evidence confirms that
                    the zone contains measurable change,
                    but the semantic interpretation remains
                    ambiguous.
                  </span>

                </div>

              )}

            </>

          )}

        </div>


        {/* =====================================================
            XAI METHODOLOGY
            ===================================================== */}

        <div className="panel-card">

          <div className="panel-header">

            <div>

              <span className="section-label">
                EXPLAINABILITY METHOD
              </span>

              <h3>
                How TERRAIN reached this conclusion
              </h3>

            </div>

            <Brain size={19} />

          </div>


          <div className="pipeline-list">

            <div className="pipeline-step complete">

              <span />

              <div>

                <strong>
                  1. Multispectral comparison
                </strong>

                <small>
                  Sentinel-2 B02, B03, B04 and B08 were
                  compared between the two acquisition periods.
                </small>

              </div>

            </div>


            <div className="pipeline-step complete">

              <span />

              <div>

                <strong>
                  2. Spectral change detection
                </strong>

                <small>
                  Physical spectral distance and NDVI change
                  were combined to identify changed pixels.
                </small>

              </div>

            </div>


            <div className="pipeline-step complete">

              <span />

              <div>

                <strong>
                  3. Geographic zone extraction
                </strong>

                <small>
                  Connected changed pixels were grouped into
                  spatially meaningful geographic zones.
                </small>

              </div>

            </div>


            <div className="pipeline-step complete">

              <span />

              <div>

                <strong>
                  4. Numerical evidence
                </strong>

                <small>
                  Area, NDVI delta, spectral contrast,
                  direction and detector confidence were
                  calculated for the selected zone.
                </small>

              </div>

            </div>


            <div
              className={`pipeline-step ${
                semanticData
                  ? "complete"
                  : semanticLoading
                    ? "complete"
                    : "pending"
              }`}
            >

              <span />

              <div>

                <strong>
                  5. Qwen semantic interpretation
                </strong>

                <small>
                  {semanticLoading
                    ? "Qwen 2.5-VL is interpreting the detected zone..."
                    : semanticData
                      ? "Visual interpretation completed and validated against numerical evidence."
                      : "Waiting for semantic interpretation."}
                </small>

              </div>

            </div>

          </div>

        </div>

      </>

    )}

  </section>
)}

        {activePage === "Reports" && (
        <section className="dashboard-content page-section">
          <div className="page-section-header">
            <div>
              <span className="section-label">ANALYSIS REPORTING</span>
              <h2>Satellite Change Report</h2>
              <p>
                Verified TERRAIN analysis outputs, change-zone evidence,
                temporal metrics, and semantic interpretation.
              </p>
            </div>

            <span className="ready-badge">REPORT READY</span>
          </div>

          {/* REPORT SUMMARY */}
          <div className="stats-grid">
            <div className="stat-card">
              <span className="stat-label">ROI PIXELS</span>
              <strong>
                {analysisSummary?.roi_pixels?.toLocaleString() || "—"}
              </strong>
              <small>Valid analysis region</small>
            </div>

            <div className="stat-card">
              <span className="stat-label">CHANGED PIXELS</span>
              <strong>
                {analysisSummary?.changed_pixels?.toLocaleString() || "—"}
              </strong>
              <small>Detected spectral change</small>
            </div>

            <div className="stat-card">
              <span className="stat-label">CHANGED AREA</span>
              <strong>
                {analysisSummary?.changed_percentage != null
                  ? `${analysisSummary.changed_percentage.toFixed(2)}%`
                  : "—"}
              </strong>
              <small>Of valid ROI</small>
            </div>

            <div className="stat-card">
              <span className="stat-label">CHANGE ZONES</span>
              <strong>
                {analysisSummary?.zone_count ?? zones.length}
              </strong>
              <small>Geographic zones detected</small>
            </div>
          </div>

          {/* REPORT CONTENT */}
          <div className="panel-card">
            <div className="panel-header">
              <div>
                <span className="section-label">REPORT CONTENT</span>
                <h3>Verified analysis package</h3>
              </div>

              <FileText size={19} />
            </div>

            <div className="pipeline-list">

              <div className="pipeline-step complete">
                <span />
                <div>
                  <strong>Satellite analysis</strong>
                  <small>
                    Sentinel-2 before/after observations and spectral
                    change detection are included.
                  </small>
                </div>
              </div>

              <div className="pipeline-step complete">
                <span />
                <div>
                  <strong>Change-zone inventory</strong>
                  <small>
                    {zones.length} geographic zones are currently loaded
                    from the verified GeoJSON output.
                  </small>
                </div>
              </div>

              <div className="pipeline-step complete">
                <span />
                <div>
                  <strong>Temporal evidence</strong>
                  <small>
                    NDVI change and spectral contrast are available for
                    the detected zones.
                  </small>
                </div>
              </div>

              <div
                className={`pipeline-step ${
                  selectedZone && semanticData
                    ? "complete"
                    : "pending"
                }`}
              >
                <span />
                <div>
                  <strong>Semantic interpretation</strong>
                  <small>
                    {selectedZone && semanticData
                      ? "Selected-zone Qwen interpretation is available."
                      : "Semantic interpretation can be generated from the Dashboard for a selected zone."}
                  </small>
                </div>
              </div>
            </div>
          </div>

          {/* TOP ZONES */}
          <div className="panel-card">
            <div className="panel-header">
              <div>
                <span className="section-label">CHANGE ZONES</span>
                <h3>Highest-area detected zones</h3>
              </div>

              <MapPinned size={19} />
            </div>

            {zones.length > 0 ? (
              <div className="data-table-wrapper">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>ZONE</th>
                      <th>DIRECTION</th>
                      <th>SEVERITY</th>
                      <th>AREA</th>
                      <th>CONFIDENCE</th>
                    </tr>
                  </thead>

                  <tbody>
                    {[...zones]
                      .sort(
                        (a, b) =>
                          (b.properties?.area_m2 || 0) -
                          (a.properties?.area_m2 || 0)
                      )
                      .slice(0, 8)
                      .map((zone) => {
                        const p = zone.properties || {};

                        return (
                          <tr key={p.region_id}>
                            <td>
                              <strong>
                                Zone {p.region_id ?? "—"}
                              </strong>
                            </td>

                            <td>
                              {String(
                                p.direction || "unknown"
                              ).replaceAll("_", " ")}
                            </td>

                            <td>
                              {p.severity || "—"}
                            </td>

                            <td>
                              {p.area_m2 != null
                                ? `${Number(
                                    p.area_m2
                                  ).toFixed(1)} m²`
                                : "—"}
                            </td>

                            <td>
                              {p.detector_confidence != null
                                ? `${(
                                    Number(
                                      p.detector_confidence
                                    ) * 100
                                  ).toFixed(1)}%`
                                : "—"}
                            </td>
                          </tr>
                        );
                      })}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="evidence-disclaimer">
                No verified change zones are currently available.
              </div>
            )}
          </div>

          {/* METHODOLOGY */}
          <div className="panel-card">
            <div className="panel-header">
              <div>
                <span className="section-label">METHODOLOGY</span>
                <h3>Evidence used by the report</h3>
              </div>

              <BrainCircuit size={19} />
            </div>

            <div className="pipeline-list">
              <div className="pipeline-step complete">
                <span />
                <div>
                  <strong>Sentinel-2 multispectral imagery</strong>
                  <small>
                    B02, B03, B04 and B08 spectral observations.
                  </small>
                </div>
              </div>

              <div className="pipeline-step complete">
                <span />
                <div>
                  <strong>Temporal spectral analysis</strong>
                  <small>
                    Before/after spectral distance and NDVI change.
                  </small>
                </div>
              </div>

              <div className="pipeline-step complete">
                <span />
                <div>
                  <strong>Adaptive change detection</strong>
                  <small>
                    Pixel-level change likelihood followed by
                    connected-component region extraction.
                  </small>
                </div>
              </div>

              <div className="pipeline-step complete">
                <span />
                <div>
                  <strong>Semantic interpretation</strong>
                  <small>
                    Qwen interprets already-detected change zones using
                    numerical evidence and visual evidence.
                  </small>
                </div>
              </div>
            </div>
          </div>

          {/* DOWNLOAD */}
          <div className="empty-page-card">
            <FileText size={32} />

            <h2>Professional PDF report</h2>

            <p>
              Generate and download the authoritative TERRAIN satellite
              change analysis report containing the current analysis
              metrics, change-zone inventory, evidence and methodology.
            </p>

            <div
              style={{
                display: "flex",
                gap: "12px",
                justifyContent: "center",
                flexWrap: "wrap",
                marginTop: "18px",
              }}
            >
              <button
                className="analyze-button"
                onClick={() =>
                  window.open(
                    `${API_BASE_URL}/api/v1/analysis/report/pdf`,
                    "_blank"
                  )
                }
              >
                <Download size={18} />
                Download PDF Report
              </button>

              <button
                className="secondary-button"
                onClick={() => setActivePage("Temporal Analysis")}
              >
                <BarChart3 size={18} />
                View Temporal Evidence
              </button>
            </div>

            <div className="evidence-disclaimer">
              The PDF is generated from the backend's verified spectral
              analysis output. No dashboard-only or fabricated metrics are
              introduced into the report.
            </div>
          </div>
        </section>
      )}

      </main>

    </div>
  );
}

export default App;