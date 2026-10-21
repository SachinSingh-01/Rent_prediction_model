from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path


_PROCESS_STARTED = time.perf_counter()
_DIAGNOSTIC_LOGGER = logging.getLogger("rentwise.diagnostics")
_DIAGNOSTIC_LOGGER.setLevel(logging.INFO)
_DIAGNOSTIC_LOGGER.propagate = False
if not _DIAGNOSTIC_LOGGER.handlers:
    _DIAGNOSTIC_HANDLER = logging.StreamHandler(sys.stdout)
    _DIAGNOSTIC_HANDLER.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    _DIAGNOSTIC_LOGGER.addHandler(_DIAGNOSTIC_HANDLER)


def _memory_diagnostics() -> dict[str, int | str]:
    """Return best-effort Linux memory readings without affecting app startup."""
    metrics: dict[str, int | str] = {
        "rss_bytes": "unavailable",
        "cgroup_memory_current_bytes": "unavailable",
        "cgroup_memory_limit_bytes": "unavailable",
    }
    if not sys.platform.startswith("linux"):
        return metrics

    try:
        status_lines = Path("/proc/self/status").read_text(encoding="ascii").splitlines()
        rss_line = next(line for line in status_lines if line.startswith("VmRSS:"))
        metrics["rss_bytes"] = int(rss_line.split()[1]) * 1024
    except (OSError, StopIteration, ValueError, IndexError):
        pass

    cgroup_files = (
        ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.max"),
        (
            "/sys/fs/cgroup/memory/memory.usage_in_bytes",
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
        ),
    )
    for current_path, limit_path in cgroup_files:
        try:
            current_value = Path(current_path).read_text(encoding="ascii").strip()
            limit_value = Path(limit_path).read_text(encoding="ascii").strip()
            metrics["cgroup_memory_current_bytes"] = int(current_value)
            metrics["cgroup_memory_limit_bytes"] = (
                limit_value if limit_value == "max" else int(limit_value)
            )
            break
        except (OSError, ValueError):
            continue
    return metrics


def _log_diagnostic(stage: str, event: str, **fields: object) -> None:
    """Write safe, flushed stage metadata; never include request or dataset values."""
    try:
        details = {
            "pid": os.getpid(),
            "process_elapsed_ms": round((time.perf_counter() - _PROCESS_STARTED) * 1000, 3),
            **_memory_diagnostics(),
            **fields,
        }
        serialized = " ".join(f"{key}={value}" for key, value in details.items())
        _DIAGNOSTIC_LOGGER.info("stage=%s event=%s %s", stage, event, serialized)
    except Exception:
        # Diagnostics must never interfere with application behavior.
        pass


@contextmanager
def _diagnostic_stage(stage: str):
    started = time.perf_counter()
    _log_diagnostic(stage, "start")
    try:
        yield
    except BaseException as error:
        _log_diagnostic(
            stage,
            "failed",
            elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
            error_type=type(error).__name__,
        )
        raise
    else:
        _log_diagnostic(
            stage,
            "complete",
            elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
        )


def _installed_version(distribution: str) -> str:
    try:
        return version(distribution)
    except Exception:
        return "unavailable"


_log_diagnostic(
    "runtime",
    "versions",
    python_version=sys.version.split()[0],
    numpy_version=_installed_version("numpy"),
    pandas_version=_installed_version("pandas"),
    scipy_version=_installed_version("scipy"),
    scikit_learn_version=_installed_version("scikit-learn"),
)

with _diagnostic_stage("import_numpy"):
    import numpy as np
with _diagnostic_stage("import_pandas"):
    import pandas as pd
with _diagnostic_stage("import_fastapi_and_pydantic"):
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, ConfigDict, Field, model_validator
with _diagnostic_stage("import_scikit_learn"):
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "House_prediction.csv"

NUMERIC_FEATURES = [
    "BHK",
    "Size",
    "Bathroom",
    "Floor_Number",
    "Total_Floors",
    "Posted_month",
    "Locality_Frequency",
]
CATEGORICAL_FEATURES = [
    "Area Type",
    "City",
    "Furnishing Status",
    "Tenant Preferred",
    "Point of Contact",
]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


class PredictionRequest(BaseModel):
    """The request fields sent by the Rentwise Next.js form."""

    model_config = ConfigDict(extra="forbid")

    city: str = Field(min_length=1)
    locality: str = Field(min_length=1)
    bhk: int = Field(gt=0)
    size: float = Field(gt=0)
    bathrooms: int = Field(gt=0)
    floor: int = Field(ge=0)
    total_floors: int = Field(gt=0)
    area_type: str = Field(min_length=1)
    furnishing_status: str = Field(min_length=1)
    tenant_preferred: str = Field(min_length=1)
    point_of_contact: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_floor_range(self) -> PredictionRequest:
        if self.floor > self.total_floors:
            raise ValueError("floor cannot be greater than total_floors")
        return self


@dataclass(frozen=True)
class ModelBundle:
    preprocessor: ColumnTransformer
    estimator: GradientBoostingRegressor
    locality_frequency: dict[str, int]


def _convert_floor(value: str) -> int:
    """Match the notebook's Ground/basement floor-number conversion."""
    if value == "Ground":
        return 0
    if value == "Upper Basement":
        return -1
    if value == "Lower Basement":
        return -2
    return int(value)


def _split_floor(value: str) -> tuple[float, float]:
    """Return the notebook's Floor_Number and Total_Floors features."""
    parts = value.split(" out of ")
    if len(parts) == 1:
        return float(_convert_floor(parts[0])), np.nan

    floor_token, total_token = parts[0], parts[1]
    if floor_token.isdigit() and total_token.isdigit():
        if int(floor_token) > int(total_token):
            return np.nan, np.nan

    return float(_convert_floor(floor_token)), float(int(total_token))


def _training_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Recreate the feature columns and target used by the final notebook model."""
    required_columns = {
        "Posted On",
        "BHK",
        "Rent",
        "Size",
        "Floor",
        "Area Type",
        "Area Locality",
        "City",
        "Furnishing Status",
        "Tenant Preferred",
        "Bathroom",
        "Point of Contact",
    }
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")

    with _diagnostic_stage("feature_engineering"):
        with _diagnostic_stage("feature_engineering_frame_copy"):
            prepared = frame.copy()

        with _diagnostic_stage("feature_engineering_posted_on_conversion"):
            prepared["Posted On"] = pd.to_datetime(prepared["Posted On"], errors="raise")

        with _diagnostic_stage("feature_engineering_month_extraction"):
            prepared["Posted_month"] = prepared["Posted On"].dt.month

        with _diagnostic_stage("feature_engineering_floor_mapping"):
            floor_features = prepared["Floor"].map(_split_floor)

        with _diagnostic_stage("feature_engineering_floor_assignments"):
            prepared["Floor_Number"] = floor_features.map(lambda features: features[0])
            prepared["Total_Floors"] = floor_features.map(lambda features: features[1])

        with _diagnostic_stage("feature_engineering_feature_selection_copy"):
            features = prepared[
                [
                    "BHK",
                    "Size",
                    "Bathroom",
                    "Floor_Number",
                    "Total_Floors",
                    "Posted_month",
                    "Area Type",
                    "City",
                    "Furnishing Status",
                    "Tenant Preferred",
                    "Point of Contact",
                ]
            ].copy()

        with _diagnostic_stage("feature_engineering_rent_conversion"):
            target = prepared["Rent"].astype(float)

        with _diagnostic_stage("feature_engineering_locality_conversion"):
            localities = prepared["Area Locality"].astype(str)
    return features, target, localities


@lru_cache(maxsize=1)
def _load_model() -> ModelBundle:
    """Fit the notebook's final estimator once, using its exact train split."""
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Rent dataset not found at {DATASET_PATH}")

    with _diagnostic_stage("csv_loading"):
        frame = pd.read_csv(DATASET_PATH)
        row_count = len(frame)
        column_count = len(frame.columns)
    _log_diagnostic("csv_loading", "summary", rows=row_count, columns=column_count)
    features, target, localities = _training_frame(frame)
    with _diagnostic_stage("train_test_split_and_locality_frequency"):
        X_train, _X_test, y_train, _y_test = train_test_split(
            features,
            target,
            test_size=0.2,
            random_state=42,
        )
        train_localities = localities.loc[X_train.index]
        locality_frequency = train_localities.value_counts()

        X_train = X_train.copy()
        X_train["Locality_Frequency"] = train_localities.map(locality_frequency)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    with _diagnostic_stage("training_preprocessing"):
        X_train_encoded = preprocessor.fit_transform(X_train[MODEL_FEATURES])

    estimator = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        random_state=42,
    )
    with _diagnostic_stage("model_fitting"):
        estimator.fit(X_train_encoded, np.log1p(y_train))

    return ModelBundle(
        preprocessor=preprocessor,
        estimator=estimator,
        locality_frequency=locality_frequency.to_dict(),
    )


def _request_features(request: PredictionRequest, bundle: ModelBundle) -> pd.DataFrame:
    """Map API fields to the notebook's expected inference columns."""
    with _diagnostic_stage("request_feature_engineering"):
        return pd.DataFrame(
            [
                {
                    "BHK": request.bhk,
                    "Size": request.size,
                    "Bathroom": request.bathrooms,
                    "Floor_Number": request.floor,
                    "Total_Floors": request.total_floors,
                    # The frontend does not submit Posted On; use this month's
                    # number, as the notebook does for posted listings.
                    "Posted_month": datetime.now().month,
                    "Locality_Frequency": bundle.locality_frequency.get(request.locality, 1),
                    "Area Type": request.area_type,
                    "City": request.city,
                    "Furnishing Status": request.furnishing_status,
                    "Tenant Preferred": request.tenant_preferred,
                    "Point of Contact": request.point_of_contact,
                }
            ],
            columns=MODEL_FEATURES,
        )


app = FastAPI(
    title="Rentwise House Rent Prediction API",
    description="Predict monthly rent using the model and feature pipeline from the Rentwise notebook.",
    version="1.0.0",
)

configured_origins = os.getenv(
    "RENTWISE_CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000",
)
allowed_origins = [origin.strip() for origin in configured_origins.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, float]:
    """Return the model's estimated monthly rent in INR."""
    bundle = _load_model()
    features = _request_features(request, bundle)
    with _diagnostic_stage("prediction_preprocessing"):
        encoded_features = bundle.preprocessor.transform(features[MODEL_FEATURES])
    with _diagnostic_stage("prediction"):
        log_rent = bundle.estimator.predict(encoded_features)[0]
        predicted_rent = float(np.expm1(log_rent))
        if not np.isfinite(predicted_rent):
            raise ValueError("The trained model produced a non-finite prediction.")
    return {"predicted_rent": predicted_rent}
