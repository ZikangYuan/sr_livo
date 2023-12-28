# SR_LIVO

**SR-LIVO** (LiDAR-Inertial-Visual Odometry and Mapping System with Sweep Reconstruction) is designed based on the framework of [**R3Live**](https://github.com/hku-mars/r3live). We employ the **sweep reconstruction** method to align reconstructed sweeps with image timestamps. This allows the LIO module to accurately determine states at all imaging moments, enhancing pose accuracy and processing efficiency. In **SR-LIVO**, an ESIKF is utilized to solve state in LIO module, and utilize an ESIKF to optimize camera parameters in vision module respectively for optimal state estimation and colored point cloud map reconstruction.

## Demo Video (2023-12-27 Update)

The **colored point cloud map (left)** and the **x8 Real-Time Performance** (right) on the sequence *hku_campus_seq_00* of self-collected dataset from [**R3Live**](https://github.com/hku-mars/r3live). On our currently hardware platform (**Intel Core i7-11700 and 32 GB RAM**) needs **30~34ms** to handle a sweep with image under this environment.

<div align="left">
<img src="doc/map.png" width=40.0% />
<img src="doc/runx8.gif" width=52.11% />
</div>
