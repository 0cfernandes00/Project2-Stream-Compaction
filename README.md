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

