CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Caroline Fernandes
  * [LinkedIn](https://www.linkedin.com/in/caroline-fernandes-0-/), [personal website](https://0cfernandes00.wixsite.com/visualfx)
* Tested on: Windows 11, i9-14900HX @ 2.20GHz, Nvidia GeForce RTX 4070

### Features & Sections
- CPU Scan and Stream Compaction
- Naive GPU Scan
- Work Efficient Scan and Compaction
- Thrust Scan
- Optimization
- Final Output

![](img/diagram_two.png)

![](img/diagram_one.png)
  
The overarching goals for this project were to
1) Understand and implement Stream Compaction on the GPU
2) Practice converting algorithms to be parallel

This implementation of stream compaction is for removing zeroes from an array of ints but this algorithm will be useful for removing unhelpful rays for a path tracer.

In this process, I first implemented these algorithms on the CPU including Scan(Prefix Sum) and built up to a naive, work efficient, and the thrust implementation of stream compaction.
I also optimized my work efficient scan bringing the time on a power-of-2 array from 0.23 ms to 0.08ms for an array size of 2 to the power of 8.

### CPU
This section implemented an Exclusive Prefix Sum (Scan), a Stream Compaction without scan, and finally built up to a Stream Compaction with scan.

Stream Compaction with scan
1) Map the input into a bools array of 0s and 1s
2) Perform a prefix sum of the input array into the bool array
3) The resulting scan (from step 2) will tell which index to scatter the input array to

The primary performance hit for this implementation was Memory I/O, multiple reads and writes were most impactful as well as multiple passes over the data.
Smaller arrays size made this implementation look fast since there was little computation overhead compared to their parallel counterparts.

### Naive
Pseudocode from [GPU Gems 3 Chapter 39 (Section 39.2.1)](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)

The primary hit to performance for the naive implementation could be the multiple accesses to global memory, additionally this implementation required to buffers to swap between in order to prevent race conditions since this algorithm was not intended to be done in-place.


### Work Efficient
The Work Efficient Compaction utilized a parallel reduction up-sweep kernel and a down-sweep kernel as part of the sum, provided again from the book.

I tried a few different methods for optimization. I originally implemented it according to the book and lecture slides keeping the kernels seperate. I also tried combining the two operations into a single kernel to reduce the time going to the memory copy I currently do before performing the downsweep. The preformance change that ended up making a difference was to reduce the number of blocks that are launched based on the number of threads needed for the current iteration of the loop.

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


### Final Output (for Size = 2 ^ 8)
```
****************
** SCAN TESTS **
****************
    [  10  43  13  38   6   1  25  30   0  26   9  46  40 ...  19   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 0.0006ms    (std::chrono Measured)
    [   0  10  53  66 104 110 111 136 166 166 192 201 247 ... 6452 6471 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 0.0008ms    (std::chrono Measured)
    [   0  10  53  66 104 110 111 136 166 166 192 201 247 ... 6385 6400 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 0.446464ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 0.105472ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 0.19968ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 0.137216ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 2.11613ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 0.871744ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   2   3   1   0   0   1   3   2   0   0   3   2   0 ...   1   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 0.0009ms    (std::chrono Measured)
    [   2   3   1   1   3   2   3   2   3   1   3   1   1 ...   2   1 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 0.0006ms    (std::chrono Measured)
    [   2   3   1   1   3   2   3   2   3   1   3   1   1 ...   1   2 ]
    passed
==== cpu compact with scan ====
   elapsed time: 0.0036ms    (std::chrono Measured)
    [   2   3   1   1   3   2   3   2   3   1   3   1   1 ...   2   1 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 0.175104ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 0.182464ms    (CUDA Measured)
    passed
```
