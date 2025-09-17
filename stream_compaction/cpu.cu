#include <cstdio>
#include "cpu.h"

#include "common.h"
#include <iostream>


namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /*
        // counting sort implementation
        void count_sort(int arr[], int n, int pos)
        {

            // we declare a count array and initialize the array by
            // 0
            int count[10] = { 0 };

            // we count the frequency of each distinct digit at
            // given place for every element in the original array
            for (int i = 0; i < n; i++) {
                count[(arr[i] / pos) % 10]++;
            }

            // we perform prefix sum and update the count array
            for (int i = 1; i < 10; i++) {
                count[i] = count[i] + count[i - 1];
            }

            // we store our answer in the ans array
            int ans[n];
            for (int i = n - 1; i >= 0; i--) {
                ans[--count[(arr[i] / pos) % 10]] = arr[i];
            }

            // here we copy the contents of ans array to our
            // original array
            for (int i = 0; i < n; i++) {
                arr[i] = ans[i];
            }
        }
        */

        //void cpuRadixSort(int n, int* odata, const int* idata) {
        /*
        void cpuRadixSort(int n, int arr[])
        {
            // max_element() is a c++ stl function to find the
            // maximum element from an array
            int k = *std::max_element(arr, arr + n);

            for (int pos = 1; (k / pos) > 0; pos *= 10) {
                //count_sort(arr, n, pos);
            }
        }*/

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
			odata[0] = 0;
            for(int i = 1; i <n; i++) {
                odata[i] += idata[i-1] + odata[i-1];
			}
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO

			int count = 0;

			for (int i = 0; i < n; i++) {
				if (idata[i] != 0) {
					odata[count] = idata[i];
					count++;
				}
			}

            timer().endCpuTimer();
            return count;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */


        
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO

			int count = 0;

            // stream compaction using the scan function. 
            // Map the input array to an array of 0s and 1s, 
            // scan it, and use scatter to produce the output. 
            // You will need a CPU scatter implementation for this 
            // (see slides or GPU Gems chapter for an explanation).

			int* bools = new int[n];
			int* indices = new int[n];

            for (int i = 0; i < n; ++i) {
                bools[i] = 0;
                indices[i] = 0;
            }

			// map input array into bools array
            for (int i = 0; i < n; i++) {
                if (idata[i] == 0) {
                    bools[i] = 0;
                }
                else {
                    bools[i] = 1;
					count++;
                }
            }

			// bools contains the 0s and 1s
			// indices contains the scanned results 

			// scan bools into indices
			// odata == indices, idata == bools
            indices[0] = 0;
            for (int i = 1; i < n; i++) {
                indices[i] += bools[i - 1] + indices[i - 1];
            }

            // the scan result will tell which index to scatter to
            // for the output array
            
            // scatter
            for (int i = 0; i < n; i++) {
                if (bools[i] == 1) {
                    int idx = indices[i];
					odata[idx] = idata[i];
                }
            }

            timer().endCpuTimer();


			delete[] bools;
            delete[] indices;

			return count;
        }
       
    }
}
