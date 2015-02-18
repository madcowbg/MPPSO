#ifndef _IND_STUDENT_T_PSO_CUH_
#define _IND_STUDENT_T_PSO_CUH_

#include "PSO.cuh"
#include <vector>

class IndStudentTPSO : public PSO {
	thrust::device_vector<gpu_fp_t> sampleVector;
	int nSample;
	gpu_fp_t nu;
	
public:
	virtual void evaluateCurrent();
	
	IndStudentTPSO(std::vector<gpu_fp_t*> sample, unsigned int nSample, gpu_fp_t _nu = 20, int nParticles = 128, int nIter = 600);
	
	IndStudentTPSO(gpu_fp_t* sample, unsigned int nSample, gpu_fp_t _nu = 20, int nParticles = 128, int nIter = 600);
	
	void evaluateGradient();
	void gradientDescent();

	gpu_fp_t getMu(int taskIdx);
	gpu_fp_t getSigma(int taskIdx);
};

#endif