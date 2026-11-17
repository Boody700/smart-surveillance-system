# Smart Surveillance System

Computer vision experiments for person tracking, temporal stabilization, zone-based rules and event storage using Python, Ultralytics YOLO, OpenCV and SQLite.

## Repository contents

| Component | Location |
|---|---|
| Person tracking and tracking experiments | `agent1_tracking/` |
| Zone calibration and rules | `agent2_rules/` |
| Shared model, video and output paths | `config.py` |
| Database setup and zone configuration | `database/` |
| Recorded tracking outputs | `output_videos/` |

The `agent3_vlm/` and `agent4_dashboard/` directories currently contain placeholders. The published source covers the tracking and rules components. An independently runnable VLM or dashboard component is not included in this repository.

## Run the tracking experiment

Create a Python environment and install `requirements.txt`. Place an input video at the `VIDEO_PATH` configured in `config.py`, then run from the repository root.

```bash
python -m venv .venv
# PowerShell: .venv\Scripts\Activate.ps1
# Bash: source .venv/bin/activate
python -m pip install -r requirements.txt
python agent1_tracking/test_run.py
```

The configuration currently references `yolo26s.pt`. Ensure that the configured weights are available and compatible with the installed Ultralytics version. The runner reads `agent1_tracking/custom_tracker.yaml`, processes person detections and writes an annotated video under `output_videos/`.

## Inspect the decisions

The tracking runner uses a persistence threshold before assigning a stable visible ID. Zone configuration is stored in `database/zones.json`, with calibration and rule handling in `agent2_rules/`. Review those files before interpreting event outputs.

## Scope

This is a development repository. It currently lacks a published reproducible detection benchmark, an automated integration suite and complete source for every component described in the broader project concept. Recorded outputs demonstrate the experiments, while detector quality and deployment behavior require separate evaluation.

For fully documented portfolio studies with screenshots, case studies and automated browser checks, see the [GitHub profile](https://github.com/Boody700).
