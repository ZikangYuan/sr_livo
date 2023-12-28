# SR_LIVO

**SR-LIVO** (LiDAR-Inertial-Visual Odometry and Mapping System with Sweep Reconstruction) is designed based on the framework of [**R3Live**](https://github.com/hku-mars/r3live). We employ the **sweep reconstruction** method to align reconstructed sweeps with image timestamps. This allows the LIO module to accurately determine states at all imaging moments, enhancing pose accuracy and processing efficiency. In **SR-LIVO**, an ESIKF is utilized to solve state in LIO module, and utilize an ESIKF to optimize camera parameters in vision module respectively for optimal state estimation and colored point cloud map reconstruction.

## Related Work

SR-LIVO: LiDAR-Inertial-Visual Odometry and Mapping with Sweep Reconstruction

Authors: [*Zikang Yuan*](https://scholar.google.com/citations?hl=zh-CN&user=acxdM9gAAAAJ), *Jie Deng*, *Ruiye Ming*, [*Fengtian Lang*](https://scholar.google.com/citations?hl=zh-CN&user=zwgGSkEAAAAJ&view_op=list_works&gmla=ABEO0Yrl4-YPuowyntSYyCW760yxM5-IWkF8FGV4t9bs9qz1oWrqnlHmPdbt7LMcMDc04kl2puqRR4FaZvaCUONsX7MQhuAC6a--VS2pTsuwj-CyKgWp3iWDP2TS0I__Zui5da4) and [*Xin Yang*](https://scholar.google.com/citations?user=lsz8OOYAAAAJ&hl=zh-CN)

[SR-LIO: LiDAR-Inertial Odometry with Sweep Reconstruction](https://arxiv.org/abs/2210.10424)

Authors: [*Zikang Yuan*](https://scholar.google.com/citations?hl=zh-CN&user=acxdM9gAAAAJ), [*Fengtian Lang*](https://scholar.google.com/citations?hl=zh-CN&user=zwgGSkEAAAAJ&view_op=list_works&gmla=ABEO0Yrl4-YPuowyntSYyCW760yxM5-IWkF8FGV4t9bs9qz1oWrqnlHmPdbt7LMcMDc04kl2puqRR4FaZvaCUONsX7MQhuAC6a--VS2pTsuwj-CyKgWp3iWDP2TS0I__Zui5da4), *Tianle Xu* and [*Xin Yang*](https://scholar.google.com/citations?user=lsz8OOYAAAAJ&hl=zh-CN)

[SDV-LOAM: Semi-Direct Visual-LiDAR Odometry and Mapping](https://ieeexplore.ieee.org/abstract/document/10086694)

Authors: [*Zikang Yuan*](https://scholar.google.com/citations?hl=zh-CN&user=acxdM9gAAAAJ), *Qingjie Wang*, *Ken Cheng*, *Tianyu Hao* and [*Xin Yang*](https://scholar.google.com/citations?user=lsz8OOYAAAAJ&hl=zh-CN)

## Demo Video (2023-12-27 Update)

The **colored point cloud map (left)** and the **x8 Real-Time Performance** (right) on the sequence *hku_campus_seq_00* of self-collected dataset from [**R3Live**](https://github.com/hku-mars/r3live). On our currently hardware platform (**Intel Core i7-11700 and 32 GB RAM**) needs **30~34ms** to handle a sweep with image under this environment.

<div align="left">
<img src="doc/map.png" width=40.0% />
<img src="doc/runx8.gif" width=52.11% />
</div>
