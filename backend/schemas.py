from pydantic import BaseModel, Field


class LocationRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    radius_m: float = Field(default=500, gt=0, le=5000)

    before_date: str
    after_date: str


class SceneMetadata(BaseModel):
    scene_id: str
    date: str
    cloud_cover: float | None = None


class AcquisitionResult(BaseModel):
    latitude: float
    longitude: float
    radius_m: float

    before: SceneMetadata
    after: SceneMetadata

    bands: list[str]