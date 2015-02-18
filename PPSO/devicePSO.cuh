#ifndef _DEVICEPSO_CUH_
#define _DEVICEPSO_CUH_

__device__ __host__ inline long getIdx(long particle, long dim, long nDim){
	return  dim + particle * nDim;
}

__device__ inline gpu_fp_t restrictValue(gpu_fp_t value, gpu_fp_t min, gpu_fp_t max) {
	if (value < min)
		return min;
	else if (value > max)
		return max;
	else
		return value;
}

#endif