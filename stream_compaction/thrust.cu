#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
    namespace Thrust {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {


            timer().startGpuTimer();
            // TODO use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:
            // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());

            
            thrust::device_vector<int> dv(idata, idata + n);
            thrust::exclusive_scan(dv.begin(), dv.end(), dv.begin());
            thrust::copy(dv.begin(), dv.end(), odata);
            
            /*
            const size_t chunk_size = 1024 * 1024; // Adjust based on available memory
            thrust::device_vector<int> dv_chunk(chunk_size);

            for (size_t i = 0; i < n; i += chunk_size) {
                size_t current_chunk = std::min(chunk_size, n - i);
                dv_chunk.resize(current_chunk);
                thrust::copy(idata + i, idata + i + current_chunk, dv_chunk.begin());
                thrust::exclusive_scan(dv_chunk.begin(), dv_chunk.end(), dv_chunk.begin());
                thrust::copy(dv_chunk.begin(), dv_chunk.end(), odata + i);
            }
            */

            timer().endGpuTimer();

        }
    }
}
