# 360° Traffic Monitoring System

This project is a **360-degree traffic monitoring system** that processes multiple camera feeds to generate a unified and comprehensive view of traffic scenes. It utilizes depth estimation, point cloud reconstruction, alignment, and stitching of video frames to provide a panoramic and spatially accurate visualization.

## 🧠 Key Features

- Multi-camera video feed processing
- Depth estimation from video
- Point cloud estimation and reconstruction
- Point cloud alignment and merging
- Frame stitching for 360° scene reconstruction

## 🗂️ Project Structure

| File | Description |
|------|-------------|
| `01_run.py` | Main entry point to orchestrate the overall process. |
| `02_video_reconstruct.py` | Extracts frames and reconstructs video sequences. |
| `03_Point_Cloud_Estimation.py` | Performs depth estimation and point cloud generation. |
| `04_PC_Align_Merge.py` | Aligns and merges point clouds from different cameras. |
| `Aline_transform.py` | Applies geometric transformations for alignment. |
| `check.py` | Utility script for quick tests and validations. |
| `hubconf.py` | Defines configurations for model loading. |
| `point_cloud_coordinates.py` | Extracts coordinates from point cloud data. |
| `PreProcessing_DE_Vid.py` | Preprocesses video frames for depth estimation. |
| `RenderOption_*.json` | Rendering options for visualization tools (e.g., Open3D). |
| `stiching.py` | Handles image stitching post point cloud alignment. |

## 🔁 Workflow Pipeline

```mermaid
graph LR
A[Multiple Camera Feeds] --> B[02_video_reconstruct.py]
B --> C[PreProcessing_DE_Vid.py]
C --> D[03_Point_Cloud_Estimation.py]
D --> E[04_PC_Align_Merge.py]
E --> F[stiching.py]


pip install -r requirements.txt

python 01_run.py
python 02_video_reconstruct.py
python 03_Point_Cloud_Estimation.py
python 04_PC_Align_Merge.py
