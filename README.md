CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Caroline Fernandes
  * [LinkedIn](https://www.linkedin.com/in/caroline-fernandes-0-/), [personal website](https://0cfernandes00.wixsite.com/visualfx)
* Tested on: Windows 11, i9-14900HX @ 2.20GHz, Nvidia GeForce RTX 4070

### Project Intro

Features
- CPU Scan and Stream Compaction
- Naive GPU Scan
- Work Efficient Scan and Compaction
- Thrust Scan implementation
  
The overarching goals for this project were to
1) understand and implement Stream Compaction on the GPU
2) practice converting algorithms to be parallel 

This implementation of stream compaction is for removing zeroes from an array of ints but this algorithm will be useful for removing unhelpful rays for a path tracer.

In this process, I first implemented these algorithms on the CPU including Scan(Prefix Sum) and built up to a naive, work efficient, and thrust implementation of stream compaction.
I also optimized my work efficient scan bringing the time on a non-power-of-2 array from 0.2 ms to 0.02ms

Can you find the performance bottlenecks? Is it memory I/O? Computation? Is it different for each implementation?

### CPU
An Exclusive Prefix Sum (Scan)
Stream Compaction without scan
Stream Compaction with scan


### Naive
Pseudocode from [GPU Gems 3 Chapter 39 (Section 39.2.1)](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)
<img width="552" height="164" alt="image" src="https://github.com/user-attachments/assets/ff3d5de7-79a9-44ac-ab81-98b86c4155b3" />

Performance
The primary hit to performance for the naive implementation could be 



### Work Efficient
![](img/workeff_scan_compute.png)

### Thrust

Unlike the other implementations, thrust seems to be using the cudaStreamSynchronize which is slowing the runtime. On average, the other implementations seem to be allocating less memory than the thrust implementation. For thrust the average runtime for the cudaMalloc API call is 34.4 microseconds, and for the other implementations the average is 29.93 microseconds.
![](img/thrust_compute.png)

All implementations except for thrust
<img width="1062" height="211" alt="Screenshot 2025-09-15 225852" src="https://github.com/user-attachments/assets/86d45f3b-9953-4124-9617-3003d034e566" />
Thrust only
<img width="1073" height="155" alt="thrust_events" src="https://github.com/user-attachments/assets/3a834ca3-0f78-4976-9f9a-edb974cb2ea5" />

### Optimization Analysis

<img width="1592" height="270" alt="image" src="https://github.com/user-attachments/assets/b241883c-dc2b-4ae3-9b8f-004a867289f2" />

