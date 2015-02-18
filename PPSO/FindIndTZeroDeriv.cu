#include "FindIndTZeroDeriv.cuh"
#include <vector>
#include "devicePSO.cuh"

using namespace std;

__device__ __host__ inline gpu_fp_t calcDlogXDmu(gpu_fp_t x, gpu_fp_t nu, gpu_fp_t mu, gpu_fp_t sigma) {
	//−(ν+1)/2 (1+1/ν ((x−μ)/σ)^2 )^(−1) (-2/νσ ((x−μ)/σ))
	gpu_fp_t centerX = (x - mu) / sigma;
	return (nu+1) / (2 * (1 + centerX * centerX / nu)) * ( 2 * centerX / (nu * sigma));
}

__device__ __host__ inline gpu_fp_t calcDlogXDsigma(gpu_fp_t x, gpu_fp_t nu, gpu_fp_t mu, gpu_fp_t sigma) {
	//−σ^(−1)−(ν+1)/2 (1+1/ν ((x−μ)/σ)^2 )^(−1) (2/ν ((x−μ)/σ)(−(x−μ)/σ^2 ))
	gpu_fp_t centerX = (x - mu) / sigma;
	return -1/sigma -(nu+1)/ (2 * (1 + centerX * centerX / nu)) * (-2 * centerX * centerX /(nu*sigma));
}

__global__ void evaluateHowFarWereFromZeros(
	gpu_fp_t* fitness,
	gpu_fp_t* particleXs,
	gpu_fp_t* otherdxdmu, 
	gpu_fp_t* otherdxdsigma, 
	gpu_fp_t nu, gpu_fp_t mu, gpu_fp_t sigma, psodim dims) {
	// offset with task index
	int taskIdx = threadIdx.y;
	fitness = fitness + taskIdx * dims.nParticles;
	particleXs = particleXs + taskIdx * dims.nParticles * dims.nDimension;
	otherdxdmu = otherdxdmu + taskIdx;
	otherdxdsigma = otherdxdsigma + taskIdx;
	// offset done

	int particleIdx = blockIdx.x;
	int firstParticleXIdx = getIdx(particleIdx, 0, dims.nDimension),
		secondParticleXIdx = getIdx(particleIdx, 1, dims.nDimension);

	fitness[particleIdx] = 
		abs(calcDlogXDmu(particleXs[firstParticleXIdx], nu, mu, sigma)
			+ calcDlogXDmu(particleXs[secondParticleXIdx], nu, mu, sigma)
			+ otherdxdmu[0])
		+ abs(calcDlogXDsigma(particleXs[firstParticleXIdx], nu, mu, sigma)
			+ calcDlogXDsigma(particleXs[secondParticleXIdx], nu, mu, sigma)
			+ otherdxdsigma[0]);
}


FindIndTZeroDeriv::FindIndTZeroDeriv(vector<gpu_fp_t*> samples, int _sampleSize, gpu_fp_t _nu, gpu_fp_t _mu, gpu_fp_t _sigma, int nParticles, int nIter) 
	: PSO(nParticles, 2, nIter, samples.size()), 
	nu(_nu), mu(_mu), sigma(_sigma), sampleSize(_sampleSize),
	otherdxdmu(dims.nTasks), otherdxdsigma(dims.nTasks) {

//	cout << _sampleSize << ", " << _nu << endl;

	gpu_fp_t* otherdxdmuHost = new gpu_fp_t[samples.size()];
	gpu_fp_t* otherdxdsigmaHost = new gpu_fp_t[samples.size()];

	for(int iTask = 0; iTask < samples.size(); iTask++) {
		otherdxdmuHost[iTask] = 0;
		otherdxdsigmaHost[iTask] = 0;
		// calculate sample's derivatives TODO
		for (int j = 0; j < sampleSize; j++) {
//			cout << calcDlogXDmu(samples[iTask][j], nu, mu, sigma) << endl;
			otherdxdmuHost[iTask] += calcDlogXDmu(samples[iTask][j], nu, mu, sigma);
			otherdxdsigmaHost[iTask] += calcDlogXDsigma(samples[iTask][j], nu, mu, sigma);
		}
	}
//	FILE* file = fopen("d:/log.txt", "w");
//	fprintf(file, "ff is %g, %g, %d, %d, %d, %d\n", otherdxdmuHost[0] , otherdxdsigmaHost[0], _sampleSize, nParticles, nIter);
//	fclose(file);
	cudaMemcpyAsync(thrust::raw_pointer_cast(otherdxdmu.data()), otherdxdmuHost, sizeof(gpu_fp_t) * dims.nTasks, cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(thrust::raw_pointer_cast(otherdxdsigma.data()), otherdxdsigmaHost, sizeof(gpu_fp_t) * dims.nTasks, cudaMemcpyHostToDevice, stream);
//	cudaMemcpy(thrust::raw_pointer_cast(otherdxdmu.data()), otherdxdmuHost, sizeof(gpu_fp_t) * dims.nTasks, cudaMemcpyHostToDevice);
//	cudaMemcpy(thrust::raw_pointer_cast(otherdxdsigma.data()), otherdxdsigmaHost, sizeof(gpu_fp_t) * dims.nTasks, cudaMemcpyHostToDevice);

	// init boundaries, for mu in [-1000, 1000] and sigma in [1e-10, 1000]
	double _scale[] = {100, 100}, _x0[] = {0, 0};
	double xl[] = {-1e50, -1e50}, xu[] = {1e50, 1e50};
	init(xl, xu, _scale, _x0);

	// TODO: make it really asyncronious...
	cudaStreamSynchronize(stream);
	delete[] otherdxdmuHost;
	delete[] otherdxdsigmaHost;
}


void FindIndTZeroDeriv::evaluateCurrent() {
	evaluateHowFarWereFromZeros<<<dims.nParticles, dim3(1, dims.nTasks), 0, stream>>>(
								   thrust::raw_pointer_cast(particleEvalValues.data()), 
								   thrust::raw_pointer_cast(particleXs.data()), 
								   thrust::raw_pointer_cast(otherdxdmu.data()), 
								   thrust::raw_pointer_cast(otherdxdsigma.data()), 
								   nu, mu, sigma, dims);
}




FindIndTZeroDeriv::~FindIndTZeroDeriv(void)
{
}

gpu_fp_t FindIndTZeroDeriv::getResult(int taskIdx, int coordIdx) {
	return globalBestXs[dims.nDimension*taskIdx + coordIdx];
}