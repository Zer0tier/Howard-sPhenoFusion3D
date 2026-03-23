# PhenoFusion3D

RGB-D Point Cloud Desktop Application for plant phenotyping. Sprint 1 delivery.

## Quick Start

```bash
# Python 3.12
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
python main.py
```

## Project Structure

```
PhenoFusion3D/
  main.py                 # Entry point
  requirements.txt
  phenofusion3d/
    app/                   # PyQt UI (Howard)
      main_window.py       # Main window
      controller.py        # Orchestrates processing
      panels/              # data_panel, controls_panel, metrics_panel, log_panel
      processing_worker.py # Background reconstruction thread
    processing/            # Standalone, testable (Tanisha, Tianyu)
      rgbd.py             # RGB-D to point cloud
      icp.py              # Coloured ICP registration
      utils.py            # clean_pcd
    io/                    # I/O (Adithya)
      loader.py           # Image pairs, intrinsics
      exporter.py         # PLY, metrics CSV
    visualiser/            # 3D viewer
      viewer.py
```

## Data Format

- **Combined folder**: `rgb_*.png` and `depth_*.png` in same folder (stakeholder format)
- **Intrinsics**: Optional `kdc_intrinsics.txt` (JSON with K, dist). Default used if omitted.
- **Public dataset**: ICL-NUIM or TUM RGB-D for development before APPN data arrives.

## Legacy Code

Original scripts are at repo root: `3D_recons.py`, `rospy_thread_fin_1.py`, `3D_reconstruction.ipynb`.
