#ifndef _PSO_CUH_
#define _PSO_CUH_

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>

#include <curand.h>

typedef double gpu_fp_t;

struct psodim {
	int nParticles;
	int nDimension;
	int nTasks;
};

struct psoopts {
	gpu_fp_t inertia;
	gpu_fp_t cpbest;
	gpu_fp_t cgbest;
	gpu_fp_t vmax;

	psoopts(gpu_fp_t inertia, gpu_fp_t cpbest, gpu_fp_t cgbest, gpu_fp_t vmax) {
		this->inertia = inertia;
		this->cpbest = cpbest;
		this->cgbest = cgbest;
		this->vmax = vmax;
	}
};

class PSO {
	// nParticles * nDimension
	thrust::device_vector<gpu_fp_t> particleVs;

	// nParticles * nDimension
	thrust::device_vector<gpu_fp_t> particleBestXs;

	// nParticles
	thrust::device_vector<gpu_fp_t> particleBestEvalValues;

	// random number generator
	curandGenerator_t m_prng;

	// bounds
	thrust::device_vector<gpu_fp_t> xlDev;
	thrust::device_vector<gpu_fp_t> xuDev;
	thrust::device_vector<gpu_fp_t> scale;
	thrust::device_vector<gpu_fp_t> x0;

	// various optimization parameters
	psoopts opts;

	// randoms variable, no need to re-allocate it all the time
	thrust::device_vector<gpu_fp_t> randoms;

protected:
	psodim dims;
	int nIter;

	// nParticles * nDimension
	thrust::device_vector<gpu_fp_t> particleXs;

	// nParticles
	thrust::device_vector<gpu_fp_t> particleEvalValues;

	// nParticles * nDimension
	thrust::device_vector<gpu_fp_t> particleEvalGradient;

	// nDimension
	thrust::device_vector<gpu_fp_t> globalBestXs;

	// 1
	thrust::device_vector<int> gBestIdx;
	thrust::device_vector<gpu_fp_t> gBestValue;
	thrust::device_vector<gpu_fp_t> newGBestValue; 
	thrust::device_vector<int> newGBestIdx; 

	cudaStream_t stream;

	// GCPSO
	thrust::device_vector<gpu_fp_t> bestParticleRho;
	thrust::device_vector<int> bestParticleErrorSuccessIdx;

	int successThreshold;
	int failThreshold;
	
public:
	PSO(int nParticles, int nDimension, int nIter, int nTasks = 1);

	~PSO();

	void init(gpu_fp_t * xl, gpu_fp_t * xu, gpu_fp_t * _scale, gpu_fp_t * _x0);

	void optimize();

	virtual void evaluateCurrent();
	virtual void evaluateGradient();

	void updateBest();

	void updateVelocitiesAndMove();
	
	void moveParticles();

	void dump();

	gpu_fp_t getFit(int taskIdx){ return gBestValue[taskIdx];}

	void setParams(psoopts opts);

	int getNTasks() { return dims.nTasks;}
};

#endif