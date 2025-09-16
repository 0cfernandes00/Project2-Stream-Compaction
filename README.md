CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Caroline Fernandes
  * [LinkedIn](https://www.linkedin.com/in/caroline-fernandes-0-/), [personal website](https://0cfernandes00.wixsite.com/visualfx)
* Tested on: Windows 11, i9-14900HX @ 2.20GHz, Nvidia GeForce RTX 4070

### Project Intro

Features & Sections
- CPU Scan and Stream Compaction
- Naive GPU Scan
- Work Efficient Scan and Compaction
- Thrust Scan
- Optimization
- Final Output
  
The overarching goals for this project were to
1) Understand and implement Stream Compaction on the GPU
2) Practice converting algorithms to be parallel 

This implementation of stream compaction is for removing zeroes from an array of ints but this algorithm will be useful for removing unhelpful rays for a path tracer.

In this process, I first implemented these algorithms on the CPU including Scan(Prefix Sum) and built up to a naive, work efficient, and thrust implementation of stream compaction.
I also optimized my work efficient scan bringing the time on a non-power-of-2 array from 0.2 ms to 0.02ms

Can you find the performance bottlenecks? Is it memory I/O? Computation? Is it different for each implementation?

### CPU
This section implemented an Exclusive Prefix Sum (Scan), a Stream Compaction without scan, and finally built up to a Stream Compaction with scan.

Stream Compaction with scan
    1) Map the input into a bools array of 0s and 1s
    2) Perform a prefix sum of the input array into the bool array
    3) The resulting scan (from step 2) will tell which index to scatter the input array to


### Naive
Pseudocode from [GPU Gems 3 Chapter 39 (Section 39.2.1)](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)

The primary hit to performance for the naive implementation could be the multiple accesses to global memory, additionally this implementation required to buffers to swap between in order to prevent race conditions since this algorithm was not intended to be done in-place.


### Work Efficient
The Work Efficient Compaction utilized an parallel reduction up-sweep kernel and a down-sweep kernel as part of the sum, provided again from the book.

I had originally expected this to be the fastest implementation.


The NSight Compute report seemed to suggest that the bottleneck was largely in computation usage for this algorithm. I had originally implemented this algorithm using the pseudocode from the book and slides. However combining the upsweep and downsweep kernels into a single kernel (as well as removing the memory copy I was using in my approach), proved to be a benefit in reducing the amount of time.
![](img/workeff_scan_compute.png)

### Thrust
This was simply a call to the Thrust library to compare against our other implementations. The compute report suggested that the kernel thrust uses has high register usage and low occupancy requiring more hardware resources for a single thread.
![](img/thrust_compute.png)

Unlike the other implementations, thrust seems to be using the cudaStreamSynchronize which is slowing the runtime. On average, the other implementations seem to be allocating less memory than the thrust implementation. For thrust the average runtime for the cudaMalloc API call is 34.4 microseconds, and for the other implementations the average is 29.93 microseconds.


All implementations except for thrust
<img width="1062" height="211" alt="Screenshot 2025-09-15 225852" src="https://github.com/user-attachments/assets/86d45f3b-9953-4124-9617-3003d034e566" />
Thrust only
<img width="1073" height="155" alt="thrust_events" src="https://github.com/user-attachments/assets/3a834ca3-0f78-4976-9f9a-edb974cb2ea5" />

### Optimization Analysis

<img width="1592" height="270" alt="image" src="https://github.com/user-attachments/assets/b241883c-dc2b-4ae3-9b8f-004a867289f2" />


### Final Output
\\\

****************
** SCAN TESTS **
****************
    [  35  47  37   8  44  39  23  37   0   6  11  25  35 ...  47   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 0.0009ms    (std::chrono Measured)
    [   0  35  82 119 127 171 210 233 270 270 276 287 312 ... 6365 6412 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 0.0006ms    (std::chrono Measured)
    [   0  35  82 119 127 171 210 233 270 270 276 287 312 ... 6304 6341 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 0.354304ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 0.027648ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 0.082944ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 0.040256ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 2.22134ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 0.746208ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   1   3   1   0   0   1   3   3   0   2   3   3   1 ...   3   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 0.001ms    (std::chrono Measured)
    [   1   3   1   1   3   3   2   3   3   1   3   1   1 ...   1   3 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 0.001ms    (std::chrono Measured)
    [   1   3   1   1   3   3   2   3   3   1   3   1   1 ...   3   1 ]
    passed
==== cpu compact with scan ====
   elapsed time: 0.0076ms    (std::chrono Measured)
    [   1   3   1   1   3   3   2   3   3   1   3   1   1 ...   1   3 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 0.011264ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 0.009216ms    (CUDA Measured)
    passed
\\\

