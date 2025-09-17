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


        __global__ void kernNaiveScan(int n, int d, int* odata, const int* idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);

            if (index >= n) return;
			int val = 1 << (d-1);

            if (index >= val ) {
                odata[index] = idata[index] + idata[index - val];
            }
            else {
                odata[index] = idata[index];
            }

        }

        __global__ void exclusiveShift(int n, int* odata, const int* idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);

            if (index >= n || index < 0) return;

            if (index == 0) odata[0] = idata[0];

            odata[index] = idata[index - 1];
              
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

            
            int blockSize = 256;
            int numBlocks = (n + blockSize - 1) / blockSize;

            int* tmp_A;
            int* tmp_B;

			timer().startGpuTimer();
            for(int d = 1; d <= ilog2ceil(n); d++) {
                tmp_A = d % 2 == 1 ? d_A : d_B;
                tmp_B = d % 2 == 1 ? d_B : d_A;
                kernNaiveScan<<<numBlocks, blockSize>>>(n, d, tmp_B, tmp_A);
				cudaDeviceSynchronize();

			}
            timer().endGpuTimer();

            cudaMemcpy(odata, tmp_B, n * sizeof(int), cudaMemcpyDeviceToHost);

            for (int i = n - 1; i > 0; i--) {
                odata[i] = odata[i - 1];
            }
            odata[0] = 0;



            cudaFree(d_A);
			cudaFree(d_B);
        }

        __global__ void kernRadixSort(int n, int* odata, const int* idata, int bit) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) return;

			int lsb = (idata[index] >> bit) & 1;

        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void radixSort(int n, int* odata, const int* idata) {

			// get the least significant bit
            //int lsb = idata & 1;

            return;
        }
    }
}
