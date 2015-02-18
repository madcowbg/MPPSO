#pragma once

#include "PSO.cuh"
#include <vector>

using namespace std;

class FindIndTNuZeroDeriv :
	public PSO
{
	// nTasks
	thrust::device_vector<gpu_fp_t> otherdxdmu; 
	thrust::device_vector<gpu_fp_t> otherdxdsigma;
	thrust::device_vector<gpu_fp_t> otherdxdnu;

	gpu_fp_t nu; 
	gpu_fp_t mu; 
	gpu_fp_t sigma; 

	int sampleSize;

public:
	FindIndTNuZeroDeriv(vector<gpu_fp_t*> samples, int _sampleSize, gpu_fp_t _nu, gpu_fp_t _mu, gpu_fp_t _sigma, int nParticles, int nIter);
	~FindIndTNuZeroDeriv(void);

	gpu_fp_t getResult(int taskIdx, int coordIdx);
	virtual void evaluateCurrent();
};

