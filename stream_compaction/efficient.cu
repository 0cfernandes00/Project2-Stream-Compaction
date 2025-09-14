#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#include <iostream>

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }


        __global__ void kernDownSweepEfficientScan(int n, int d, int* odata) {
            int k = threadIdx.x + (blockIdx.x * blockDim.x);

            int stride = 1 << (d + 1);  // 2^(d+1)

            if (k * stride >= n) return;

            int leftIdx = stride * k + (1 << d) - 1;
            int rightIdx = stride * k + stride - 1;

            int t = odata[leftIdx];

            //if (rightIdx < n) {
                odata[leftIdx] = odata[rightIdx];
                odata[rightIdx] += t;
            //}
        }


        __global__ void kernUpSweepEfficientScan(int n, int d, int* odata) {
            int index = (blockIdx.x * blockDim.x + threadIdx.x);
            int stride = 1 << (d + 1);  // 2^(d+1)

            // Only use every 2^(d+1)th thread
            if (index * stride >= n) return;

            int leftIdx = stride * index + (1 << d) - 1;   // Left child
            int rightIdx = stride * index + stride - 1;    // Right child

            if (rightIdx < n) {
                odata[rightIdx] += odata[leftIdx];
            }         
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

            int* d_data;
            
            int paddedN = 1 << ilog2ceil(n);
            int logceilN = ilog2ceil(paddedN);

            cudaMalloc((void**)&d_data, paddedN * sizeof(int));
            cudaMemcpy(d_data, idata, paddedN * sizeof(int), cudaMemcpyHostToDevice);

            // TODO
			int blockSize = 64;
			int numBlocks = (paddedN + blockSize - 1) / blockSize;

            timer().startGpuTimer();

            for (int d = 0; d < logceilN; ++d) {
                //int threadsNeeded = n / (1 << (d + 1));
                //if (threadsNeeded == 0) break;  // No more work to do

                //int numBlocksLevel = (threadsNeeded + blockSize - 1) / blockSize;


                kernUpSweepEfficientScan << <numBlocks, blockSize >> > (n, d, d_data);
                cudaDeviceSynchronize();           
            }

            int setNMinusOne_ToZero = 0;
            cudaMemcpy(d_data + paddedN - 1, &setNMinusOne_ToZero, sizeof(int), cudaMemcpyHostToDevice);

            for (int d = logceilN - 1; d > -1; --d) {
                //int threadsNeeded = (paddedN)/ (1 << (d + 1));
                //if (threadsNeeded == 0) break;

                //int numBlocksLevel = (threadsNeeded + blockSize - 1) / blockSize;

                kernDownSweepEfficientScan << <numBlocks, blockSize >> > (paddedN, d, d_data);
                cudaDeviceSynchronize();
			}

            timer().endGpuTimer();

            cudaMemcpy(odata, d_data, paddedN * sizeof(int), cudaMemcpyDeviceToHost);


			cudaFree(d_data);

  
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            // TODO

            int *count;
            int *d_bools;
			int *d_indices;
			int *d_idata;
			int* d_odata;


            int paddedN = 1 << ilog2ceil(n);

            cudaMalloc((void**)&d_bools, paddedN * sizeof(int));
            cudaMalloc((void**)&d_indices, paddedN * sizeof(int));
            cudaMalloc((void**)&d_idata, paddedN * sizeof(int));
			cudaMalloc((void**)&d_odata, paddedN * sizeof(int));

            cudaMemcpy(d_idata, idata, paddedN * sizeof(int), cudaMemcpyHostToDevice);

            // map to bools
            int blockSize = 64;
            int numBlocks = (paddedN + blockSize - 1) / blockSize;

            //timer().startGpuTimer();

            StreamCompaction::Common::kernMapToBoolean << <numBlocks, blockSize >> > (paddedN, d_bools, d_idata);
            cudaDeviceSynchronize();

			int* tmp_bools = new int[paddedN];
			cudaMemcpy(tmp_bools, d_bools, paddedN * sizeof(int), cudaMemcpyDeviceToHost);


            int* tmp_indices = new int[paddedN];
            // scan bools into indices
            scan(paddedN, tmp_indices, tmp_bools);
			cudaDeviceSynchronize();


			// copy indices to device for scatter kernel
            cudaMemcpy(d_indices, tmp_indices, paddedN * sizeof(int), cudaMemcpyHostToDevice);


			// scatter
            StreamCompaction::Common::kernScatter << <numBlocks, blockSize >> > (paddedN, d_odata, d_idata, d_bools, d_indices);
			cudaDeviceSynchronize();


            int returnVal = 0;
            for (int i = 0; i < n; i++) {
				tmp_bools[i] == 1 ? returnVal++ : returnVal += 0;
            }

            //timer().endGpuTimer();

            cudaMemcpy(odata, d_odata, paddedN * sizeof(int), cudaMemcpyDeviceToHost);

			delete[] tmp_bools;
			delete[] tmp_indices;
			cudaFree(d_bools);
			cudaFree(d_indices);
			cudaFree(d_idata);
            cudaFree(d_odata);



            return returnVal;
        }
    }
}
