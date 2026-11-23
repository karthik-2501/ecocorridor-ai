"""
Image Record Viewer Module.
Locates local camera-trap image files by `file_name`, displays verified metadata,
and provides a clean camera-trap placeholder visualizer when raw image files are offline.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import streamlit as st

import config


def create_synthetic_camera_frame(
    record: Dict[str, Any],
    output_path: Optional[Path] = None
) -> Image.Image:
    """
    Generate an informational camera-trap telemetry card image
    with timestamp, camera station ID, frame number, and species tag overlay.
    """
    width = int(record.get("width", 1280)) if pd.notna(record.get("width")) else 1280
    height = int(record.get("height", 720)) if pd.notna(record.get("height")) else 720

    # Downscale for preview rendering
    preview_w, preview_h = 800, int(800 * (height / width))
    img = Image.new("RGB", (preview_w, preview_h), color="#0a140e")
    draw = ImageDraw.Draw(img)

    # Draw grid overlay
    for x in range(0, preview_w, 80):
        draw.line([(x, 0), (x, preview_h)], fill="#14261b", width=1)
    for y in range(0, preview_h, 80):
        draw.line([(0, y), (preview_w, y)], fill="#14261b", width=1)

    # Sensor Crosshair Center
    cx, cy = preview_w // 2, preview_h // 2
    draw.line([(cx - 40, cy), (cx + 40, cy)], fill="#22c55e", width=2)
    draw.line([(cx, cy - 40), (cx, cy + 40)], fill="#22c55e", width=2)
    draw.ellipse([(cx - 20, cy - 20), (cx + 20, cy + 20)], outline="#22c55e", width=1)

    # Telemetry Header Bar
    draw.rectangle([(0, 0), (preview_w, 42)], fill="#0f2115")
    header_text = f"CAM-TRAP SENSOR STATION #{record.get('location', 'N/A')}  |  SEQ: {str(record.get('seq_id', 'N/A'))[:8]}...  |  FRAME {record.get('frame_num', 1)}/{record.get('seq_num_frames', 1)}"
    draw.text((15, 14), header_text, fill="#86efac")

    # Species & Telemetry Footer Bar
    draw.rectangle([(0, preview_h - 48), (preview_w, preview_h)], fill="#0f2115")
    dt_str = str(record.get("date_captured", "N/A"))
    cat_str = str(record.get("category", "unknown")).upper()
    footer_text = f"CAPTURED: {dt_str}   |   CLASS: [{cat_str}]   |   RES: {width}x{height}px"
    draw.text((15, preview_h - 34), footer_text, fill="#f8fafc")

    if output_path:
        img.save(output_path, "JPEG", quality=85)

    return img


def locate_image_file(file_name: str, search_directory: Optional[Path] = None) -> Optional[Path]:
    """Search for matching image file in user-specified or common directories."""
    if not file_name:
        return None

    candidate_dirs = []
    if search_directory and Path(search_directory).exists():
        candidate_dirs.append(Path(search_directory))

    candidate_dirs.extend([
        config.DATA_DIR / "images",
        config.BASE_DIR / "images",
        Path(r"E:\wildlife_corridor_corrected\data\images"),
        Path(r"E:\jassem\images")
    ])

    for d in candidate_dirs:
        if d.exists() and d.is_dir():
            target = d / file_name
            if target.exists() and target.is_file():
                return target

    return None


def render_image_record(record: Dict[str, Any], image_dir: Optional[str] = None):
    """Render image record with dual columns: visualizer and verified metadata."""
    file_name = str(record.get("file_name", ""))
    img_path = locate_image_file(file_name, Path(image_dir) if image_dir else None)

    col1, col2 = st.columns([1.3, 1.0])

    with col1:
        if img_path and img_path.exists():
            try:
                img = Image.open(img_path)
                st.image(img, caption=f"Original Camera Capture: {file_name}", use_container_width=True)
            except Exception as e:
                st.error(f"Error opening image file: {e}")
                card_img = create_synthetic_camera_frame(record)
                st.image(card_img, caption=f"Sensor Metadata Card: {file_name}", use_container_width=True)
        else:
            # Render aesthetic sensor telemetry card
            card_img = create_synthetic_camera_frame(record)
            st.image(
                card_img,
                caption=f"Camera Sensor Card (Physical image '{file_name}' not in local image folder)",
                use_container_width=True
            )
            st.caption("ℹ️ To display raw photographs, configure your local image folder path in the sidebar.")

    with col2:
        st.markdown(f"### Record Details")
        st.markdown(f"**Image ID:** <span class='id-chip'>{record.get('image_id', 'N/A')}</span>", unsafe_allow_html=True)
        st.markdown(f"**Filename:** `{file_name}`")
        st.markdown(f"**Capture Timestamp:** `{record.get('date_captured', 'N/A')}`")
        st.markdown(f"**Camera Station ID:** `Location #{record.get('location', 'N/A')}`")
        st.markdown(f"**Sequence ID:** <span class='id-chip'>{record.get('seq_id', 'N/A')}</span>", unsafe_allow_html=True)
        st.markdown(f"**Sequence Burst:** Frame `{record.get('frame_num', 1)}` of `{record.get('seq_num_frames', 1)}`")
        
        category = str(record.get('category', 'unknown'))
        is_animal = category not in config.NON_ANIMAL_CATEGORIES
        badge_style = "badge-success" if is_animal else ("badge-warning" if category == "car" else "badge-neutral")
        
        st.markdown(
            f"**Classified Label:** <span class='badge {badge_style}'>{category.upper()}</span>",
            unsafe_allow_html=True
        )
        st.markdown(f"**Verified Annotations:** `{record.get('num_annotations', 1)}` object(s)")
        st.markdown(f"**Native Resolution:** `{record.get('width', 'N/A')} x {record.get('height', 'N/A')} px`")
        st.markdown(f"**Rights Holder:** `{record.get('rights_holder', 'N/A')}`")
