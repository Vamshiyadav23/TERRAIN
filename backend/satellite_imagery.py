from pystac_client import Client


SENTINEL_STAC_URL = "https://earth-search.aws.element84.com/v1"


def connect_to_stac() -> Client:
    """
    Connect to the Earth Search STAC API.
    """
    return Client.open(SENTINEL_STAC_URL)


def search_sentinel2(
    latitude: float,
    longitude: float,
    radius_m: float,
    date_range: str,
    max_cloud_cover: float = 20.0,
):
    """
    Search Sentinel-2 L2A scenes around a location.

    Returns matching STAC items ordered by cloud cover.
    """

    client = connect_to_stac()

    radius_deg = radius_m / 111_000

    bbox = [
        longitude - radius_deg,
        latitude - radius_deg,
        longitude + radius_deg,
        latitude + radius_deg,
    ]

    search = client.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=date_range,
        query={
            "eo:cloud_cover": {
                "lt": max_cloud_cover
            }
        },
        max_items=20,
    )

    items = list(search.items())

    items.sort(
        key=lambda item: (
            item.properties.get("eo:cloud_cover", 100.0)
        )
    )

    return items