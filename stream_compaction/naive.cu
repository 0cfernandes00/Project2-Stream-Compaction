#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#include <iostream>

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__


        __global__ void kernNaiveScan(int n, int d, int* odata, const int* idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);


            if (index >= n) return;
			int val = 1 << (d-1);
            if (index == 0) {
                odata[0] = idata[0];
                //return;
			}

            if (index >= val ) {
                odata[index] = idata[index - val] + idata[index];
            }
            else {
                odata[index] = idata[index];
            }

        }

        // This uses the "Naive" algorithm from GPU Gems 3, Section 39.2.1. 
        // Example 39-1 uses shared memory. This is not required in this project. 
        // You can simply use global memory. As a result of this, you will have to do ilog2ceil(n) separate kernel invocations.

        // Since your individual GPU threads are not guaranteed to run simultaneously, 
        // you can't generally operate on an array in-place on the GPU; 
        // it will cause race conditions. Instead, create two device arrays. 
        // Swap them at each iteration: read from A and write to B, read from B and write to A, and so on.


        __global__ void exclusiveShift(int n, int* odata, const int* idata) {
            // TODO
            int index = threadIdx.x + (blockIdx.x * blockDim.x);

            if (index >= n || index < 0) return;

            odata[index] = idata[index - 1];
              
		}

        int ilog2ceil(int n) {
            int log = 0;
            int pow2 = 1;
            while (pow2 < n) {
                log++;
                pow2 *= 2;
            }
            return log;
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

            int* d_A;
            int* d_B;

            cudaMalloc((void**)&d_A, n * sizeof(int));
            cudaMalloc((void**)&d_B, n * sizeof(int));
            cudaMemcpy(d_A, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(d_B, odata, n * sizeof(int), cudaMemcpyHostToDevice);


            timer().startGpuTimer();
            // TODO

            int blockSize = 32;
            int numBlocks = (n + blockSize - 1) / blockSize;

            int swapBuffer = 0;
            for (int d = 1; d <= ilog2ceil(n); ++d) {
                kernNaiveScan << <numBlocks, blockSize >> > (n, d, d_B, d_A);
                cudaDeviceSynchronize();

                int* temp;

				std::swap(d_A, d_B);

            }

            exclusiveShift << <numBlocks, blockSize >> > (n, d_B, d_A);

            timer().endGpuTimer();

			cudaMemcpy(odata, d_B, n * sizeof(int), cudaMemcpyDeviceToHost);



            cudaFree(d_A);
			cudaFree(d_B);
        }
    }
}
