# Stereo–Inertial Visual Odometry (VIO) from Scratch

This repository contains an end-to-end stereo–inertial visual odometry (VIO) system implemented from scratch in Python and evaluated on the EuRoC MAV dataset.

The goal of this project is to demonstrate a deep understanding of visual–inertial estimation, spanning:

- raw sensor data handling,

- stereo geometry,

- IMU propagation,

- nonlinear optimization,

- estimator validation against ground truth.

This is not a wrapper around an existing VIO library.

## System Overview

The pipeline estimates the 6-DoF pose of a moving platform using:

- Stereo camera images (20 Hz)

- IMU measurements (200 Hz)

### High-level architecture

```text
EuRoC Dataset
├── IMU (accelerometer + gyroscope)
├── Stereo cameras (cam0, cam1)
│
↓
[ IMU Propagation ]
│
↓
[ Stereo Frontend ]
- FAST feature detection
- LK optical flow (temporal tracking)
- Stereo matching (rectified)
- Triangulation
│
↓
[ Pose-only Gauss-Newton Update ]
```
## Features Implemented
### Stereo Frontend

- FAST corner detection

- Lucas–Kanade optical flow (temporal tracking)

- Stereo rectification using calibration

- Stereo matching with disparity-guided initialization

- Rectified triangulation

- Track management and outlier rejection

### IMU Frontend

- Midpoint integration

- Gyroscope and accelerometer bias handling

- Gravity-aligned initialization from IMU averaging

### Estimation Backend

- Nonlinear pose-only Gauss–Newton optimization

- Robust loss (Huber weighting)

- Numeric Jacobians for measurement model

- Landmark management and pruning

### Validation & Analysis

- Evaluation on EuRoC MAV – Vicon Room

- Ground-truth comparison using Vicon motion capture

- Yaw gauge freedom analysis and alignment

- Position error and drift analysis
