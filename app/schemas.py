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


class RecommendationRequest(BaseModel):
    """RAG recommendation input. Risk must already come from MFBI + ML."""

    student_id: Optional[str] = None
    monitoring_id: Optional[int] = None
    mfbi_id: Optional[int] = None
    prediction_id: Optional[int] = None
    stress_score: Optional[float] = Field(default=None, ge=0)
    academic_workload_score: Optional[float] = Field(default=None, ge=0)
    study_time_score: Optional[float] = Field(default=None, ge=0)
    sleep_hours_score: Optional[float] = Field(default=None, ge=0)
    # Spec aliases (same values, different names)
    stress_level: Optional[float] = Field(default=None, ge=0)
    academic_workload: Optional[float] = Field(default=None, ge=0)
    study_time: Optional[float] = Field(default=None, ge=0)
    sleep_hours: Optional[float] = Field(default=None, ge=0)
    mfbi_score: Optional[float] = Field(default=None, ge=0, le=1)
    risk_level: Optional[str] = None
    prediction_model: Optional[str] = None
    selected_model: Optional[str] = None

    def resolved_scores(self) -> dict[str, float]:
        stress = self.stress_score if self.stress_score is not None else self.stress_level
        workload = (
            self.academic_workload_score
            if self.academic_workload_score is not None
            else self.academic_workload
        )
        study = self.study_time_score if self.study_time_score is not None else self.study_time
        sleep = (
            self.sleep_hours_score
            if self.sleep_hours_score is not None
            else self.sleep_hours
        )
        if None in (stress, workload, study, sleep):
            raise ValueError(
                "Provide stress, academic workload, study time, and sleep scores"
            )
        return {
            "stress_score": float(stress),
            "academic_workload_score": float(workload),
            "study_time_score": float(study),
            "sleep_hours_score": float(sleep),
        }
