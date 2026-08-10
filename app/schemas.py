from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class FeaturePayload(BaseModel):
    stress_score: float = Field(..., ge=0)
    academic_workload_score: float = Field(..., ge=0)
    study_time_score: float = Field(..., ge=0)
    sleep_hours_score: float = Field(..., ge=0)
    mfbi_score: Optional[float] = Field(default=None, ge=0, le=1)
    stress_trend: Optional[float] = None
    workload_trend: Optional[float] = None
    study_trend: Optional[float] = None
    sleep_trend: Optional[float] = None


class PredictionRequest(BaseModel):
    student_id: Optional[str] = None
    features: Optional[FeaturePayload] = None
    mode: Literal["same_week", "next_week", "auto"] = "auto"
    # Optional history for early-warning trend analysis (oldest → newest)
    history_levels: Optional[list[str]] = None
    history_mfbi: Optional[list[float]] = None

    @model_validator(mode="after")
    def require_student_or_features(self):
        if not self.student_id and not self.features:
            raise ValueError("Provide student_id and/or features")
        return self


class EarlyWarningRequest(BaseModel):
    student_id: Optional[str] = None
    features: Optional[FeaturePayload] = None
    history_levels: Optional[list[str]] = None
    history_mfbi: Optional[list[float]] = None

    @model_validator(mode="after")
    def require_student_or_features(self):
        if not self.student_id and not self.features:
            raise ValueError("Provide student_id and/or features")
        return self
