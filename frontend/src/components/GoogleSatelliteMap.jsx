import {
  GoogleMap,
  Polygon,
  useJsApiLoader,
} from "@react-google-maps/api";

const CENTER = {
  lat: 12.971912,
  lng: 77.614378,
};

const MAP_CONTAINER_STYLE = {
  width: "100%",
  height: "100%",
};

const MAP_OPTIONS = {
  mapTypeId: "satellite",
  disableDefaultUI: true,
  zoomControl: true,
  fullscreenControl: true,
  gestureHandling: "greedy",
};

function getSeverityStyle(severity) {
  switch (severity) {
    case "critical":
      return {
        fillColor: "#ff3b30",
        strokeColor: "#ff6258",
      };

    case "high":
      return {
        fillColor: "#ff8a00",
        strokeColor: "#ff9f33",
      };

    case "medium":
      return {
        fillColor: "#ffd60a",
        strokeColor: "#ffe45c",
      };

    case "low":
      return {
        fillColor: "#30d158",
        strokeColor: "#63df7c",
      };

    default:
      return {
        fillColor: "#00bfff",
        strokeColor: "#4dd4ff",
      };
  }
}

function GoogleSatelliteMap({
  zones = [],
  onZoneSelect,
}) {
  const { isLoaded, loadError } = useJsApiLoader({
    googleMapsApiKey:
      import.meta.env.VITE_GOOGLE_MAPS_API_KEY,
  });

  if (loadError) {
    return (
      <div className="map-error">
        <strong>Google Maps failed to load</strong>

        <span>
          Check your Google Maps API key and enabled APIs.
        </span>
      </div>
    );
  }

  if (!isLoaded) {
    return (
      <div className="map-loading">
        <div className="map-loading-spinner" />

        <span>
          Loading satellite imagery...
        </span>
      </div>
    );
  }

  return (
    <GoogleMap
      mapContainerStyle={MAP_CONTAINER_STYLE}
      center={CENTER}
      zoom={15}
      options={MAP_OPTIONS}
    >
      {zones.map((zone) => {
        const geometry = zone.geometry;

        if (!geometry) {
          return null;
        }

        const properties = zone.properties || {};

        const style = getSeverityStyle(
          properties.severity,
        );

        let polygons = [];

        if (geometry.type === "Polygon") {
          polygons = [geometry.coordinates];
        }

        if (geometry.type === "MultiPolygon") {
          polygons = geometry.coordinates.map(
            (polygon) => polygon,
          );
        }

        return polygons.map(
          (polygon, polygonIndex) => {
            const outerRing = polygon[0];

            const paths = outerRing.map(
              ([lng, lat]) => ({
                lat,
                lng,
              }),
            );

            return (
              <Polygon
                key={`${properties.region_id}-${polygonIndex}`}
                paths={paths}
                options={{
                  fillColor: style.fillColor,
                  fillOpacity: 0.35,
                  strokeColor: style.strokeColor,
                  strokeOpacity: 0.95,
                  strokeWeight: 2,
                  clickable: true,
                }}
                onClick={() =>
                  onZoneSelect?.(zone)
                }
              />
            );
          },
        );
      })}
    </GoogleMap>
  );
}

export default GoogleSatelliteMap;