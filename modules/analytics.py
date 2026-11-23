"""
Ecological and Wildlife Activity Analytics Module.
Calculates species distributions, Shannon diversity index, temporal patterns,
and station-level wildlife metrics from camera-trap observations.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

import config


def compute_species_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize observation volume, percentage, day vs night breakdown,
    and sequence frequency for each species/category.
    """
    if df is None or df.empty or "category" not in df.columns:
        return pd.DataFrame()

    total_records = len(df)
    grouped = df.groupby("category").agg(
        total_count=("category", "count"),
        unique_locations=("location", "nunique") if "location" in df.columns else ("category", lambda x: 1),
        unique_sequences=("seq_id", "nunique") if "seq_id" in df.columns else ("category", lambda x: 1),
    ).reset_index()

    grouped["percentage"] = (grouped["total_count"] / total_records * 100).round(2)
    grouped["is_animal"] = ~grouped["category"].isin(config.NON_ANIMAL_CATEGORIES)

    # Diurnal vs Nocturnal proportion if available
    if "time_of_day" in df.columns:
        day_counts = df[df["time_of_day"] == "Day"].groupby("category").size()
        night_counts = df[df["time_of_day"] == "Night"].groupby("category").size()
        grouped["day_count"] = grouped["category"].map(day_counts).fillna(0).astype(int)
        grouped["night_count"] = grouped["category"].map(night_counts).fillna(0).astype(int)
        total_time = grouped["day_count"] + grouped["night_count"]
        grouped["nocturnal_ratio"] = np.where(total_time > 0, (grouped["night_count"] / total_time).round(3), 0.0)

    grouped = grouped.sort_values("total_count", ascending=False).reset_index(drop=True)
    return grouped


def compute_shannon_diversity(species_counts: pd.Series) -> float:
    """
    Calculate Shannon Diversity Index (H) for an ecological community:
    H = - sum(p_i * ln(p_i))
    Where p_i is the proportion of total observations belonging to species i.
    """
    # Exclude empty and non-animal triggers for true ecological diversity
    counts = species_counts[~species_counts.index.isin(config.NON_ANIMAL_CATEGORIES)]
    if counts.empty or counts.sum() == 0:
        return 0.0
    proportions = counts / counts.sum()
    proportions = proportions[proportions > 0]
    return float(-np.sum(proportions * np.log(proportions)))


def compute_location_analytics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate station-level statistics:
    - Total recorded captures
    - Wildlife (animal) captures
    - Vehicle (human intrusion) count
    - Empty false trigger rate
    - Species richness (unique species observed)
    - Shannon Diversity Index
    - Average burst sequence length
    """
    if df is None or df.empty or "location" not in df.columns:
        return pd.DataFrame()

    loc_summary = []
    for loc_id, group in df.groupby("location"):
        total = len(group)
        empty_count = int((group["category"] == "empty").sum()) if "category" in group.columns else 0
        vehicle_count = int((group["category"] == "car").sum()) if "category" in group.columns else 0
        animal_records = total - empty_count - vehicle_count
        
        # Species richness
        animal_group = group[group["category"].isin(config.VALID_ANIMAL_SPECIES)] if "category" in group.columns else pd.DataFrame()
        species_richness = animal_group["category"].nunique() if not animal_group.empty else 0
        
        # Shannon index
        species_counts = group["category"].value_counts() if "category" in group.columns else pd.Series(dtype=int)
        shannon_index = compute_shannon_diversity(species_counts)
        
        # Mean sequence frames
        avg_seq_frames = group["seq_num_frames"].mean() if "seq_num_frames" in group.columns else 1.0

        loc_summary.append({
            "location_id": str(loc_id),
            "total_records": total,
            "wildlife_records": animal_records,
            "empty_records": empty_count,
            "vehicle_records": vehicle_count,
            "empty_rate_pct": round(empty_count / total * 100, 1) if total > 0 else 0.0,
            "species_richness": species_richness,
            "shannon_diversity": round(shannon_index, 3),
            "avg_sequence_frames": round(avg_seq_frames, 2),
        })

    loc_df = pd.DataFrame(loc_summary)
    
    # Identify high-activity hotspots relative to mean baseline
    mean_act = loc_df["total_records"].mean()
    std_act = loc_df["total_records"].std()
    threshold = mean_act + (1.5 * std_act)
    loc_df["is_hotspot"] = loc_df["total_records"] > threshold

    loc_df = loc_df.sort_values("total_records", ascending=False).reset_index(drop=True)
    return loc_df


def compute_sequence_analytics(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze camera burst sequences, frame timing, and multi-frame consistency."""
    if df is None or df.empty or "seq_id" not in df.columns:
        return {}

    total_seqs = df["seq_id"].nunique()
    frames_per_seq = df.groupby("seq_id")["frame_num"].count()
    
    # Sequences with animals vs empty
    seq_has_animal = {}
    if "is_animal" in df.columns:
        seq_has_animal = df.groupby("seq_id")["is_animal"].any().to_dict()
        animal_seqs = sum(seq_has_animal.values())
    else:
        animal_seqs = 0

    return {
        "total_sequences": total_seqs,
        "mean_frames_per_seq": round(float(frames_per_seq.mean()), 2),
        "median_frames_per_seq": round(float(frames_per_seq.median()), 2),
        "max_frames_per_seq": int(frames_per_seq.max()) if not frames_per_seq.empty else 0,
        "single_shot_ratio_pct": round(float((frames_per_seq == 1).mean() * 100), 1),
        "animal_sequences_count": animal_seqs,
        "animal_sequence_ratio_pct": round((animal_seqs / total_seqs * 100), 1) if total_seqs > 0 else 0.0
    }


def compute_historical_activity_trends(df: pd.DataFrame, top_locations_n: int = 10) -> pd.DataFrame:
    """
    Compute multi-year annual observation counts (2010 to 2015) and
    historical activity trajectory for top camera stations.
    """
    if df is None or df.empty or "year" not in df.columns or "location" not in df.columns:
        return pd.DataFrame()

    valid_df = df.dropna(subset=["year"]).copy()
    valid_df["year"] = valid_df["year"].astype(int)

    # Filter to main monitoring years
    valid_df = valid_df[valid_df["year"].between(2010, 2015)]
    top_locs = valid_df["location"].value_counts().head(top_locations_n).index

    yearly = valid_df[valid_df["location"].isin(top_locs)].groupby(
        ["location", "year"]
    ).size().unstack(fill_value=0)

    # Calculate trajectory: compare 2011-2012 average vs 2013-2014 average
    early_cols = [y for y in [2011, 2012] if y in yearly.columns]
    late_cols = [y for y in [2013, 2014] if y in yearly.columns]

    trajectory = []
    for loc_id in yearly.index:
        early_avg = yearly.loc[loc_id, early_cols].mean() if early_cols else 0.0
        late_avg = yearly.loc[loc_id, late_cols].mean() if late_cols else 0.0
        if early_avg > 0:
            pct_change = round(((late_avg - early_avg) / early_avg) * 100, 1)
        else:
            pct_change = 0.0

        if pct_change > 25.0:
            trend = "Surging Activity (Increasing)"
        elif pct_change < -25.0:
            trend = "Declining Activity (Decreasing)"
        else:
            trend = "Stable / Consistent"

        trajectory.append({
            "location_id": str(loc_id),
            "historical_change_pct": pct_change,
            "activity_trajectory": trend,
            "total_period_captures": int(yearly.loc[loc_id].sum())
        })

    traj_df = pd.DataFrame(trajectory)
    yearly_reset = yearly.reset_index()
    yearly_reset["location_id"] = yearly_reset["location"].astype(str)

    merged = pd.merge(traj_df, yearly_reset, on="location_id").drop(columns=["location"], errors="ignore")
    return merged.sort_values("total_period_captures", ascending=False).reset_index(drop=True)


def compute_predator_prey_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Examine camera locations for spatial co-occurrence of apex predators
    and prey herbivores, indicating active foraging corridors.
    """
    if df is None or df.empty or "location" not in df.columns or "category" not in df.columns:
        return pd.DataFrame()

    loc_summary = []
    for loc_id, group in df.groupby("location"):
        total = len(group)
        predator_count = int(group["category"].isin(config.PREDATOR_SPECIES).sum())
        prey_count = int(group["category"].isin(config.PREY_HERBIVORE_SPECIES).sum())
        vehicle_count = int((group["category"] == "car").sum())

        has_both = predator_count > 0 and prey_count > 0
        ratio = round(predator_count / prey_count, 2) if prey_count > 0 else 0.0

        loc_summary.append({
            "location_id": str(loc_id),
            "total_records": total,
            "predator_count": predator_count,
            "prey_count": prey_count,
            "predator_prey_ratio": ratio,
            "vehicle_count": vehicle_count,
            "predator_prey_cooccurrence": has_both
        })

    res_df = pd.DataFrame(loc_summary)
    return res_df.sort_values("predator_count", ascending=False).reset_index(drop=True)

