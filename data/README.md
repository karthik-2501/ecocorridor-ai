# Caltech Camera Traps Dataset (Image Level)

This directory hosts the primary dataset used by the **AI-Powered Wildlife Corridor Monitoring and Protection** platform.

## Primary File
- `caltech_camera_traps_image_level(2).csv` (also accessible as `caltech_camera_traps_image_level.csv`)

## Provenance
- **Source**: Caltech Camera Traps (CCT-20) dataset, curated by Sara Beery et al. in partnership with the USGS and local conservation partners in Southwestern United States (California).
- **Records**: 243,100 camera-trap image metadata records.
- **Monitoring Span**: 2010 to 2015 (~5 years of continuous ecological observation).
- **Sensor Deployment**: 140 distinct camera trap locations across ecological corridors.
- **Capture Sequences**: 181,008 motion-activated sequence bursts.

## Schema
| Column | Type | Description |
| :--- | :--- | :--- |
| `image_id` | String (UUID) | Unique identifier for each captured image record |
| `file_name` | String | Original image filename (e.g., `5968c0f9...jpg`) |
| `date_captured` | Timestamp | Date and time of capture recorded by camera trap |
| `location` | Integer / String | Camera deployment station identifier |
| `seq_id` | String (UUID) | Motion burst sequence identifier grouping consecutive shots |
| `frame_num` | Integer | Frame index within the motion burst sequence (e.g., 1, 2, 3) |
| `seq_num_frames` | Integer | Total frames recorded in this motion burst sequence |
| `width` | Integer | Image pixel width |
| `height` | Integer | Image pixel height |
| `category` | String | Primary classified label (23 distinct categories) |
| `labels` | String | Annotation label string |
| `num_annotations` | Integer | Count of annotated objects in frame |
| `rights_holder` | String | Research institution / conservation partner holding image rights |

## Categories Identified
- **Wild Fauna**: deer, coyote, bobcat, mountain_lion, fox, raccoon, opossum, rabbit, bird, squirrel, rodent, badger, skunk, lizard, bat, insect.
- **Domestic / Livestock**: cat, dog, cow, pig.
- **Anthropogenic / Other**: car (human/vehicle presence), empty (false trigger / background motion), multiple (multi-species/multi-animal frames).
