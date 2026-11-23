"""
Data Loader Module for Wildlife Corridor Monitoring Platform.
Handles safe CSV ingestion, column validation, caching, and dataset summary extraction.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import streamlit as st

import config


def find_default_dataset_path() -> Optional[Path]:
    """Find the first existing dataset path from configured candidate locations."""
    for path in config.DATASET_CANDIDATE_PATHS:
        if path.exists() and path.is_file():
            return path
    return None


@st.cache_data(show_spinner=True)
def load_and_validate_dataset(
    file_or_path: Any,
    nrows: Optional[int] = None
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Load dataset from a file path or Streamlit UploadedFile object.
    Performs column validation, data type coercion, and safe date parsing.

    Returns:
        (df, metadata_dict)
    """
    metadata: Dict[str, Any] = {
        "source": str(file_or_path),
        "total_rows": 0,
        "valid_dates": 0,
        "invalid_dates": 0,
        "missing_columns": [],
        "errors": []
    }

    try:
        if isinstance(file_or_path, (str, Path)):
            path = Path(file_or_path)
            if not path.exists():
                metadata["errors"].append(f"File not found: {path}")
                return None, metadata
            df = pd.read_csv(path, nrows=nrows, low_memory=False)
            metadata["source_name"] = path.name
        else:
            # Streamlit UploadedFile
            df = pd.read_csv(file_or_path, nrows=nrows, low_memory=False)
            metadata["source_name"] = getattr(file_or_path, "name", "Uploaded File")

        # Validate columns
        present_cols = set(df.columns)
        missing_cols = [c for c in config.EXPECTED_COLUMNS if c not in present_cols]
        metadata["missing_columns"] = missing_cols
        metadata["total_rows"] = len(df)

        # Standardize string fields if present
        if "category" in df.columns:
            df["category"] = df["category"].astype(str).str.strip().str.lower()
        if "labels" in df.columns:
            df["labels"] = df["labels"].astype(str).str.strip().str.lower()
        if "location" in df.columns:
            df["location"] = df["location"].astype(str).str.strip()

        # Date parsing with error coercion
        if "date_captured" in df.columns:
            # Use coerce so invalid values like '11 11' become NaT
            parsed_dates = pd.to_datetime(df["date_captured"], errors="coerce")
            invalid_mask = parsed_dates.isna()
            metadata["invalid_dates"] = int(invalid_mask.sum())
            metadata["valid_dates"] = int((~invalid_mask).sum())
            df["datetime"] = parsed_dates
            
            # Extract useful time features safely from valid dates
            df["year"] = df["datetime"].dt.year
            df["month"] = df["datetime"].dt.month
            df["day_name"] = df["datetime"].dt.day_name()
            df["hour"] = df["datetime"].dt.hour
            df["date"] = df["datetime"].dt.date
            
            # Day vs Night feature (approximate: 06:00 to 18:00 is Day)
            df["time_of_day"] = df["hour"].apply(
                lambda h: "Day" if pd.notna(h) and 6 <= h < 18 else ("Night" if pd.notna(h) else "Unknown")
            )
        else:
            metadata["invalid_dates"] = 0
            metadata["valid_dates"] = 0

        # Classify species vs non-animal
        if "category" in df.columns:
            df["is_animal"] = ~df["category"].isin(config.NON_ANIMAL_CATEGORIES)
            
            # Species group categorization
            def get_species_group(cat: str) -> str:
                if cat in config.NON_ANIMAL_CATEGORIES:
                    return "Non-Animal / Background"
                if cat in config.PREDATOR_SPECIES:
                    return "Carnivore / Predator"
                if cat in config.PREY_HERBIVORE_SPECIES:
                    return "Herbivore / Prey"
                if cat in config.HUMAN_DOMESTIC_CATEGORIES:
                    return "Domestic / Livestock"
                return "Other Wildlife"
                
            df["species_group"] = df["category"].apply(get_species_group)

        return df, metadata

    except Exception as e:
        metadata["errors"].append(str(e))
        return None, metadata


def get_dataset_summary(df: pd.DataFrame, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Calculate summary metrics strictly derived from the actual dataset."""
    if df is None or df.empty:
        return {}

    total_records = len(df)
    unique_categories = df["category"].nunique() if "category" in df.columns else 0
    unique_locations = df["location"].nunique() if "location" in df.columns else 0
    unique_sequences = df["seq_id"].nunique() if "seq_id" in df.columns else 0
    
    # Calculate animal-only statistics
    animal_records = 0
    top_animal_species = "N/A"
    top_animal_count = 0
    unique_animal_species = 0
    
    if "category" in df.columns and "is_animal" in df.columns:
        animal_df = df[df["is_animal"]]
        animal_records = len(animal_df)
        unique_animal_species = animal_df["category"].nunique()
        if not animal_df.empty:
            top_counts = animal_df["category"].value_counts()
            top_animal_species = top_counts.index[0]
            top_animal_count = int(top_counts.iloc[0])

    # Date ranges
    date_min = None
    date_max = None
    if "datetime" in df.columns:
        valid_dates = df["datetime"].dropna()
        if not valid_dates.empty:
            date_min = valid_dates.min().strftime("%Y-%m-%d")
            date_max = valid_dates.max().strftime("%Y-%m-%d")

    # Invalid / missing records
    invalid_dates_count = metadata.get("invalid_dates", 0) if metadata else 0

    return {
        "total_records": total_records,
        "unique_categories": unique_categories,
        "unique_animal_species": unique_animal_species,
        "animal_records": animal_records,
        "empty_records": int((df["category"] == "empty").sum()) if "category" in df.columns else 0,
        "vehicle_records": int((df["category"] == "car").sum()) if "category" in df.columns else 0,
        "unique_locations": unique_locations,
        "unique_sequences": unique_sequences,
        "invalid_dates_count": invalid_dates_count,
        "top_animal_species": top_animal_species,
        "top_animal_count": top_animal_count,
        "date_min": date_min,
        "date_max": date_max,
        "total_annotations": int(df["num_annotations"].sum()) if "num_annotations" in df.columns else total_records,
    }


def get_data_quality_report(df: pd.DataFrame, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Generate a data-quality audit report for dataset validation."""
    if df is None or df.empty:
        return {}

    missing_per_col = df.isnull().sum().to_dict()
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = sum(missing_per_col.values())
    completeness_score = ((total_cells - missing_cells) / total_cells * 100) if total_cells > 0 else 0

    # Duplicates check on image_id
    duplicate_ids = 0
    if "image_id" in df.columns:
        duplicate_ids = int(df["image_id"].duplicated().sum())

    # Sequence integrity check
    seq_consistency = "Good"
    if "seq_id" in df.columns and "frame_num" in df.columns and "seq_num_frames" in df.columns:
        # Check if frame_num exceeds seq_num_frames
        invalid_frames = (df["frame_num"] > df["seq_num_frames"]).sum()
        if invalid_frames > 0:
            seq_consistency = f"{invalid_frames} frames exceed sequence length"

    return {
        "completeness_score": round(completeness_score, 2),
        "total_cells": total_cells,
        "missing_cells": missing_cells,
        "missing_per_column": missing_per_col,
        "duplicate_image_ids": duplicate_ids,
        "invalid_timestamps": metadata.get("invalid_dates", 0) if metadata else 0,
        "sequence_integrity": seq_consistency,
        "dimensions_count": df[["width", "height"]].drop_duplicates().to_dict(orient="records") if "width" in df.columns else [],
    }
