CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Caroline Fernandes
  * [LinkedIn](https://www.linkedin.com/in/caroline-fernandes-0-/), [personal website](https://0cfernandes00.wixsite.com/visualfx)
* Tested on: Windows 11, i9-14900HX @ 2.20GHz, Nvidia GeForce RTX 4070

### Project Intro

This project was to understand and implement Stream Compaction on the GPU, it was also an introduction into converting algorithms to be parallel on the GPU.
In this process, I first implemented these algorithms on the CPU including Scan(Prefix Sum) and built up to a naive, work efficient, and thrust implementation of stream compaction.
I also optimized my work efficient scan bringing the time on a non-power-of-2 array from 0.2 ms to 0.02ms



### Thrust

Unlike the other implementations, thrust seems to be using the cudaStreamSynchronize which is slowing the runtime. On average, the other implementations seem to be allocating less memory than the thrust implementation. For thrust the average runtime for the cudaMalloc API call is 34.4 microseconds, and for the other implementations the average is 29.93 microseconds.

<img width="1062" height="211" alt="Screenshot 2025-09-15 225852" src="https://github.com/user-attachments/assets/86d45f3b-9953-4124-9617-3003d034e566" />
<img width="1073" height="155" alt="thrust_events" src="https://github.com/user-attachments/assets/3a834ca3-0f78-4976-9f9a-edb974cb2ea5" />
