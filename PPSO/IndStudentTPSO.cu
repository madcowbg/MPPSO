#include "IndStudentTPSO.cuh"
#include "devicePSO.cuh"

__global__ void evaluateIndStudentTFitness(
		gpu_fp_t* fitness, 
		gpu_fp_t* particleXs, 
		psodim dims, 
		gpu_fp_t* sample, 
		unsigned int nSample, 
		gpu_fp_t nu) {
	// offset with task index
	int taskIdx = blockIdx.y;
	fitness = fitness + taskIdx * dims.nParticles;
	particleXs = particleXs + taskIdx * dims.nParticles * dims.nDimension;
	sample = sample + taskIdx * nSample;
	// offset done

	int particleIdx = blockIdx.x;
	
	gpu_fp_t sigma = particleXs[getIdx(particleIdx, 0, dims.nDimension)];
	gpu_fp_t mu = particleXs[getIdx(particleIdx, 1, dims.nDimension)];
	
	__shared__ gpu_fp_t loglikelihood[1024];

	// calculate per-value log-likelihood
	int iSam = threadIdx.x;
	loglikelihood[iSam] = -log(sigma) -((nu + 1.0) / 2) * log1p(pow(((sample[iSam] - mu) / sigma), 2) / nu);

	// now reduce
	unsigned int neigh = 1;
	while (neigh < nSample) {
		__syncthreads();
		if (iSam % (2*neigh) != 0 || (iSam + neigh >= nSample)) {
			break;
		}
		loglikelihood[iSam] += loglikelihood[iSam + neigh];
		neigh <<= 1;
	}

	__syncthreads(); //sync after reduce

	if (iSam == 0) {
		fitness[particleIdx] = -loglikelihood[iSam];
	}
	//__syncthreads();

	//if (iSam == 0) {
	//	gpu_fp_t result = 0;
	//	for (int i = 0; i < nSample; i++) {
	//		result += loglikelihood[i];
	//	}
	//	fitness[particleIdx] = -result;
	//}
}

__global__ void evaluateIndStudentTGrad(gpu_fp_t* gradient, gpu_fp_t* particleXs, 
											psodim dims, gpu_fp_t* sample, unsigned int nSample, gpu_fp_t nu) {
	int particleIdx = blockIdx.x;
	
	gpu_fp_t sigma = particleXs[getIdx(particleIdx, 0, dims.nDimension)];
	gpu_fp_t mu = particleXs[getIdx(particleIdx, 1, dims.nDimension)];
	
	__shared__ gpu_fp_t gradientElem[1024];

	// calculate per-value log-likelihood
	int iSam = threadIdx.x;
	//−(ν+1)/2 (1+1/ν ((x−μ)/σ)^2 )^(−1) (2/νσ ((x−μ)/σ))
	gpu_fp_t centerX = (sample[iSam]-mu) / sigma;
	gradientElem[2*iSam] = -(nu+1)/2 / (1 + centerX * centerX / nu ) * (2 * centerX / (nu * sigma));
	// −σ^(−1)−(ν+1)/2 (1+1/ν ((x−μ)/σ)^2 )^(−1) (2/ν ((x−μ)/σ)(−(x−μ)/σ^2 ))
	gradientElem[2*iSam+1] = - 1/sigma - (nu+1)/2 / (1 + centerX * centerX / nu ) * (- 2*centerX*centerX / ( nu * sigma));

	// now reduce
	unsigned int neigh = 1;
	while (neigh < nSample) {
		__syncthreads();
		if (iSam % (2*neigh) != 0 || (iSam + neigh >= nSample)) {
			break;
		}
		gradientElem[2*iSam] += gradientElem[2*(iSam + neigh)];
		gradientElem[2*iSam + 1] += gradientElem[2*(iSam + neigh)+1];
		neigh <<= 1;
	}

	__syncthreads(); //sync after reduce

	if (iSam == 0) {
		gradient[2*particleIdx] = -gradientElem[2*iSam];
		gradient[2*particleIdx+1] = -gradientElem[2*iSam+1];
	}
	//__syncthreads();

	//if (iSam == 0) {
	//	gpu_fp_t result = 0;
	//	for (int i = 0; i < nSample; i++) {
	//		result += loglikelihood[i];
	//	}
	//	fitness[particleIdx] = -result;
	//}
}

__global__ void moveTo(gpu_fp_t* newParticleXs, gpu_fp_t* particleXs, gpu_fp_t* particleXGradient, gpu_fp_t delta, int nDim, gpu_fp_t* sample, unsigned int nSample, gpu_fp_t nu) {
	int newParticleIdx = threadIdx.x;

	newParticleXs[getIdx(newParticleIdx, 0, nDim)] = particleXs[0] + delta * (newParticleIdx+1) * particleXGradient[0];
	newParticleXs[getIdx(newParticleIdx, 1, nDim)] = particleXs[1] + delta * (newParticleIdx+1) * particleXGradient[1];
}

IndStudentTPSO::IndStudentTPSO(gpu_fp_t* sample, unsigned int nSample, gpu_fp_t _nu, int nParticles, int nIter) 
	: PSO(nParticles, 2, nIter, 1), nu(_nu), sampleVector(nSample) {
	this->nSample = nSample;
	cudaMemcpyAsync(thrust::raw_pointer_cast(sampleVector.data()), sample, sizeof(gpu_fp_t) * nSample, cudaMemcpyHostToDevice, stream);
	
	// init boundaries, for mu in [-1000, 1000] and sigma in [1e-10, 1000]
	double xl[] = {0.001, -1e50}, xu[] = {1e50, 1e50};
	double _scale[] = {0.5, 10}, _x0[] = {1, 0};
	init(xl, xu, _scale, _x0);
}

IndStudentTPSO::IndStudentTPSO(std::vector<gpu_fp_t*> sample, unsigned int nSample, gpu_fp_t _nu, int nParticles, int nIter)
	: PSO(nParticles, 2, nIter, sample.size()), nu(_nu), sampleVector(nSample * sample.size()) {
	this->nSample = nSample;

	gpu_fp_t* hostData = new gpu_fp_t[nSample * sample.size()];
	for (int iTask = 0; iTask < sample.size(); iTask++) {
		memcpy(hostData+iTask * nSample, sample[iTask], sizeof(gpu_fp_t) * nSample);
	}
	
	cudaMemcpyAsync(thrust::raw_pointer_cast(sampleVector.data()), hostData, sizeof(gpu_fp_t) * nSample * sample.size(), cudaMemcpyHostToDevice, stream);

	
	// init boundaries, for mu in [-1000, 1000] and sigma in [1e-10, 1000]
	double xl[] = {0.001, -1e50}, xu[] = {1e50, 1e50};
	double _scale[] = {0.5, 10}, _x0[] = {1, 0};
	init(xl, xu, _scale, _x0);
	
	// delete here, because cudaMemcpy could overlap, but init has kernel execution...
	cudaStreamSynchronize(stream);
	delete[] hostData;
}


void IndStudentTPSO::evaluateCurrent() {
	evaluateIndStudentTFitness<<<dim3(dims.nParticles, dims.nTasks), nSample, 0, stream>>>(thrust::raw_pointer_cast(particleEvalValues.data()), 
								   thrust::raw_pointer_cast(particleXs.data()), 
								   dims, thrust::raw_pointer_cast(sampleVector.data()), 
								   nSample, nu);  
}

void IndStudentTPSO::evaluateGradient() {
	//evaluateIndStudentTGrad<<<dims.nParticles, sampleVector.size(), 0, stream>>>(thrust::raw_pointer_cast(particleEvalGradient.data()), 
	//							   thrust::raw_pointer_cast(particleXs.data()), 
	//							   dims, thrust::raw_pointer_cast(sampleVector.data()), 
	//							   sampleVector.size(), nu);  
}

void IndStudentTPSO::gradientDescent() {
	//we've found a decent point
	//thrust::device_vector<gpu_fp_t> bestX = thrust::device_vector<gpu_fp_t>(globalBestXs);
	//thrust::device_vector<gpu_fp_t> bestEvalGradient(nDimension); //==2
	//thrust::device_vector<bool> optimized(1);
	//		
	//int nParallelSteps = 128;
	//thrust::device_vector<gpu_fp_t> newParticleXs = thrust::device_vector<gpu_fp_t>(globalBestXs);
	//

	//gpu_fp_t bestVal = globalBestVal;
	//for (int i = 0; i < 100; i++) {
	//	//find gradient
	//	evaluateIndStudentTGrad<<<1, sampleVector.size(), 0, stream>>>(thrust::raw_pointer_cast(bestEvalGradient.data()), 
	//							   thrust::raw_pointer_cast(bestX.data()), 
	//							   nDimension, thrust::raw_pointer_cast(sampleVector.data()), 
	//							   sampleVector.size(), nu);  
	//	if (abs(bestEvalGradient[1]) + abs(bestEvalGradient[2]) < 1e-10);
	//		break;

	//	gpu_fp_t step = 1 / nParallelSteps;
	//	bool moved = false;

	//	while (!moved) {
	//		moveTo<<<1, nParallelSteps, 0, stream>>>(gpu_fp_t* newParticleXs, gpu_fp_t* particleXs, gpu_fp_t* particleXGradient, gpu_fp_t delta, int nDim, gpu_fp_t* sample, unsigned int nSample, gpu_fp_t nu)
	//}
}

gpu_fp_t IndStudentTPSO::getMu(int taskIdx) {
	return globalBestXs[dims.nDimension*taskIdx + 1];
}

gpu_fp_t IndStudentTPSO::getSigma(int taskIdx) {
	return globalBestXs[dims.nDimension*taskIdx + 0];
}
