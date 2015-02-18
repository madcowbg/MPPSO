#include "PSO.cuh"
#include "devicePSO.cuh"
#include <thrust/random.h>
#include <thrust/copy.h>
#include <thrust/random/uniform_real_distribution.h>
#include <iostream>
#include <curand.h>
#include <time.h>

using namespace thrust::random;

__global__ void randPos(gpu_fp_t* particleXs/*[N x nDim]*/, gpu_fp_t* xl/*[nDim]*/, gpu_fp_t* xu/*[nDim]*/, psodim dims, gpu_fp_t* randoms, gpu_fp_t* scale, gpu_fp_t* x0)
{
	// offset with task index
	int taskIdx = threadIdx.y;
	particleXs = particleXs + taskIdx * dims.nParticles * dims.nDimension;
	randoms = randoms + taskIdx * dims.nParticles * dims.nDimension;
	// offset done

	int particleIdx = blockIdx.x;
	int dimIdx = threadIdx.x;
	
	int pxIdx = getIdx(particleIdx, dimIdx, dims.nDimension);
	//particleXs[pxIdx] = randoms[pxIdx] * xl[dimIdx] + (1 - randoms[pxIdx]) * xu[dimIdx];
	particleXs[pxIdx] = restrictValue(randoms[pxIdx] *  scale[dimIdx] + x0[dimIdx],  xl[dimIdx],  xu[dimIdx]);
}

__global__ void evaluateFun(gpu_fp_t* fitness, gpu_fp_t* particleXs, psodim dims) {
	// offset with task index
	int taskIdx = threadIdx.y;
	fitness = fitness + taskIdx * dims.nParticles;
	particleXs = particleXs + taskIdx * dims.nParticles * dims.nDimension;
	// offset done

	int particleIdx = blockIdx.x;
	
	gpu_fp_t result = 0;
	for (int iDim = 0; iDim < dims.nDimension; iDim++) {
		result += pow(particleXs[getIdx(particleIdx, iDim, dims.nDimension)], 2);
	}
	fitness[particleIdx] = result;
}

__global__ void updatePBest(
		gpu_fp_t* particleEvalValues, 
		gpu_fp_t* particleXs, 
		gpu_fp_t* particleBestEvalValues, 
		gpu_fp_t* particleBestXs, 
		psodim dims,
		/* GCPSO */
		int* bestParticleErrorSuccessIdx /*3*/,
		gpu_fp_t* bestParticleRho /*1*/,
		int successThreshold,
		int failThreshold) {
	// offset with task index
	int taskIdx = threadIdx.y;
	particleBestEvalValues = particleBestEvalValues + taskIdx * dims.nParticles;
	particleXs = particleXs + taskIdx * dims.nParticles * dims.nDimension;
	particleBestXs = particleBestXs + taskIdx * dims.nParticles * dims.nDimension;
	bestParticleErrorSuccessIdx = bestParticleErrorSuccessIdx + taskIdx * 3;
	bestParticleRho = bestParticleRho + taskIdx;
	// offset done

	int particleIdx = blockIdx.x;
	int dimIdx = threadIdx.x;
	
	bool update = particleEvalValues[particleIdx] < particleBestEvalValues[particleIdx];
	unsigned int pxIdx = getIdx(particleIdx, dimIdx, dims.nDimension);
	if (update) {
		particleBestXs [pxIdx] = particleXs[pxIdx];
	}

	if (dimIdx == 0 && update) { //update using first thread
		particleBestEvalValues[particleIdx] = particleEvalValues[particleIdx];
	}

	// GCPSO
	if (dimIdx == 0 && particleIdx == bestParticleErrorSuccessIdx[2]) { //this is the best particle
		// update success counts
		if (update) {
			bestParticleErrorSuccessIdx[0] = 0;
			bestParticleErrorSuccessIdx[1]++;
		} else {
			bestParticleErrorSuccessIdx[0]++;
			bestParticleErrorSuccessIdx[1] = 0;
		}

		// update rho if necessary 
		if (bestParticleErrorSuccessIdx[1] > successThreshold) {
			bestParticleRho[0] *= 2;
		}
		if (bestParticleErrorSuccessIdx[0] > failThreshold) {
			bestParticleRho[0] /= 2;
		}
	}

}

__global__ void findGBest(gpu_fp_t * particleEvalValues, int * gBestIdx, gpu_fp_t * gBestValue, psodim dims) {
	// offset with task index
	int taskIdx = blockIdx.y;
	particleEvalValues = particleEvalValues + taskIdx * dims.nParticles;
	gBestIdx = gBestIdx + taskIdx;
	gBestValue = gBestValue + taskIdx;
	// offset done

	extern __shared__ int sharedMem[];
	int* bestIdx = sharedMem; // store the index achieving the minimal value so far
	gpu_fp_t* bestValues = (gpu_fp_t*)(sharedMem + dims.nParticles); // store the minimal value

	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	bestIdx[tid] = tid;
	bestValues[tid] = particleEvalValues[tid];

	__syncthreads();

	// do reduction in shared mem
	for(unsigned int s = 1; tid + s < dims.nParticles; s <<= 1) {
		if (tid % (2*s) == 0) {
			if (bestValues[tid] > bestValues[tid+s]) {
				bestValues[tid] = bestValues[tid+s];
				bestIdx[tid] = bestIdx[tid+s];
			}
		}
		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0) {
		gBestIdx[0] = bestIdx[tid];
		gBestValue[0] = bestValues[tid];
	}
}

#define CudaAssert( X ) if ( !(X) ) { printf( "Thread %d:%d failed assert at %s:%d!", blockIdx.x, threadIdx.x, __FILE__, __LINE__ ); return; }


__global__ void updatePosAndVel(gpu_fp_t* particleXs/*[N x nDim]*/, 
	gpu_fp_t* particleBestXs /*[N x nDim]*/, 
	gpu_fp_t* particleVs /*[N x nDim]*/,
	gpu_fp_t* particleEvalGradient /*[N x nDim]*/, 
	gpu_fp_t* globalBestXs /*[nDim]*/,
	gpu_fp_t* xl /*[nDim]*/, 
	gpu_fp_t* xu /*[nDim]*/, 
	psodim dims, gpu_fp_t inertia, gpu_fp_t cpbest, gpu_fp_t cgbest, gpu_fp_t vmax,
	gpu_fp_t* randoms /*[3 x N x nDim + nDim]*/, 
	/* GCPSO */
	int* bestParticleErrorSuccessIdx /*3*/,
	gpu_fp_t* bestParticleRho /*1*/)
{
	// offset with task index
	int taskIdx = threadIdx.y;
	particleXs = particleXs + taskIdx * dims.nParticles * dims.nDimension;
	particleBestXs = particleBestXs + taskIdx * dims.nParticles * dims.nDimension;
	particleVs = particleVs + taskIdx * dims.nParticles * dims.nDimension;
	particleEvalGradient = particleEvalGradient + taskIdx * dims.nParticles * dims.nDimension;
	globalBestXs = globalBestXs + taskIdx * dims.nDimension;
	randoms = randoms + taskIdx * (3 * dims.nParticles * dims.nDimension + dims.nDimension);
	bestParticleErrorSuccessIdx = bestParticleErrorSuccessIdx + taskIdx * 3;
	bestParticleRho = bestParticleRho + taskIdx;
	// offset done


	// 1 block per individual
	// nDimension threads per block
	unsigned int particleIdx = blockIdx.x;
	unsigned int dimIdx = threadIdx.x;

	unsigned int pxIdx = getIdx(particleIdx, dimIdx, dims.nDimension);

	gpu_fp_t move = restrictValue(particleVs[pxIdx], -vmax, vmax);

//	printf("move %d dim %d (%d) from inertia %f from pbest %f from gbest %f\n", 
//		particleIdx, dimIdx, pxIdx, 
//		inertia * move, 
//		cpbest * randoms[3 * pxIdx] * (particleBestXs[pxIdx] - particleXs[pxIdx]), 
//		cgbest * randoms[3 * pxIdx + 1] * (globalBestXs[dimIdx] - particleXs[pxIdx]));

	// move
	particleXs[pxIdx] = restrictValue(particleXs[pxIdx] + move, xl[dimIdx], xu[dimIdx]);

	// calculate next velocity on each coordinate
	particleVs[pxIdx] = //-randoms[3 * pxIdx+2] * particleEvalGradient[pxIdx] // minus because we're searching for a minimum
		+ inertia * move 
		+ cpbest * randoms[3 * pxIdx] * (particleBestXs[pxIdx] - particleXs[pxIdx])
		+ cgbest * randoms[3 * pxIdx + 1] * (globalBestXs[dimIdx] - particleXs[pxIdx]);

//	CudaAssert( ( randoms[3 * pxIdx] < 1 && randoms[3 * pxIdx] > 0 ) && "Input data not valid!" )
//	CudaAssert( ( randoms[3 * pxIdx+1] < 1 && randoms[3 * pxIdx+1] > 0 ) && "Input data not valid!" )
//	CudaAssert( ( randoms[3 * pxIdx+2] < 1 && randoms[3 * pxIdx+2] > 0 ) && "Input data not valid!" )

	// GCPSO
	//if (particleIdx == bestParticleErrorSuccessIdx[2]) { //this is the best particle
	//	particleVs[pxIdx] = 
	//		inertia * move + 
	//		particleBestXs[pxIdx] - particleXs[pxIdx] + 
	//		bestParticleRho[0] * (2 * randoms[3 * dims.nParticles * dims.nDimension + dimIdx] - 1); // move randomly about
	//}

	// calculate new particle coordinates
//	printf("move %d dim %d (%d) by %f to %f\n", particleIdx, dimIdx, pxIdx, move, particleXs[pxIdx]);
}

PSO::PSO(int nParticles, int nDimension, int nIter, int nTasks) : 
		particleXs(nParticles * nDimension * nTasks), particleEvalValues(nParticles * nTasks),
		particleVs(nParticles * nDimension * nTasks), 
		particleEvalGradient(nParticles * nDimension * nTasks),
		particleBestXs(nParticles * nDimension * nTasks), particleBestEvalValues(nParticles * nTasks), 
		globalBestXs(nDimension * nTasks), opts(0.7, 1.4, 1.4, 1e5),
		/* GCPSO */ bestParticleRho(nTasks), bestParticleErrorSuccessIdx(3 * nTasks), gBestIdx(1 * nTasks),
		randoms((3 * nParticles * nDimension + nDimension) * nTasks), gBestValue(1 * nTasks),
		newGBestIdx (nTasks), newGBestValue(nTasks){
	dims.nDimension = nDimension;
	dims.nParticles = nParticles;
	dims.nTasks = nTasks;
	
	this->nIter = nIter;

	// init best values
	thrust::fill(gBestValue.begin(), gBestValue.end(), 1e50);
	thrust::fill(particleBestEvalValues.begin(), particleBestEvalValues.end(), 1e50);

	// create stream for this swarm
	cudaStreamCreate ( &stream) ;

	// init random number generator
	curandCreateGenerator(&m_prng, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(m_prng, ((int)&stream) + time(NULL));
	curandSetStream(m_prng, stream);

	// GCPSO
	successThreshold = 2;
	failThreshold = 5;
	thrust::fill(bestParticleRho.begin(), bestParticleRho.end(), 1);
}

void PSO::init(gpu_fp_t * xl, gpu_fp_t * xu, gpu_fp_t * _scale, gpu_fp_t * _x0) {
	xlDev = thrust::device_vector<gpu_fp_t>(xl, xl + dims.nDimension);
	xuDev = thrust::device_vector<gpu_fp_t>(xu, xu + dims.nDimension);
	scale = thrust::device_vector<gpu_fp_t>(_scale, _scale + dims.nDimension);
	x0 = thrust::device_vector<gpu_fp_t>(_x0, _x0 + dims.nDimension);
	
	// generate random positions for the swarm, only nParticles * nDimensions are needed
	curandGenerateNormalDouble(m_prng, thrust::raw_pointer_cast(randoms.data()), dims.nParticles * dims.nDimension * dims.nTasks, 0, 1);

	// initialize positions
	randPos<<<dims.nParticles, dim3(dims.nDimension, dims.nTasks), 0, stream>>>(thrust::raw_pointer_cast(particleXs.data()), 
		thrust::raw_pointer_cast(xlDev.data()), thrust::raw_pointer_cast(xuDev.data()), dims, 
		thrust::raw_pointer_cast(randoms.data()), thrust::raw_pointer_cast(scale.data()), thrust::raw_pointer_cast(x0.data()));
}

void PSO::optimize() {
	for (int iIter = 0; iIter < nIter; iIter++) {
		evaluateCurrent();
		//evaluateGradient();
		//dump();

		updateBest();
		
		updateVelocitiesAndMove();
	}
}

void PSO::evaluateCurrent() {
	evaluateFun<<<dims.nParticles, dim3(1, dims.nTasks), 0, stream>>>(thrust::raw_pointer_cast(particleEvalValues.data()), 
								   thrust::raw_pointer_cast(particleXs.data()), 
								   dims);  
}

__global__ void updateGBest(
		gpu_fp_t* globalBestXs, 
		gpu_fp_t* particleXs, 
		gpu_fp_t* gBestValue, 
		gpu_fp_t* newGBestValue, 
		int* gBestIdx, 
		int* newGBestIdx, 
		int* bestParticleErrorSuccessIdx, 
		psodim dims){
	// offset with task index
	int taskIdx = blockIdx.y;
	globalBestXs = globalBestXs + taskIdx * dims.nDimension;
	particleXs = particleXs + taskIdx * dims.nParticles * dims.nDimension;
	gBestValue = gBestValue + taskIdx;
	newGBestValue = newGBestValue + taskIdx;
	gBestIdx = gBestIdx + taskIdx;
	newGBestIdx = newGBestIdx + taskIdx;
	bestParticleErrorSuccessIdx = bestParticleErrorSuccessIdx + taskIdx * 3;
	// offset done


	if (gBestValue[0] > newGBestValue[0]) {
		gBestValue[0] = newGBestValue[0];

		int _newGBestIdx = newGBestIdx[0];
		int dimIdx = threadIdx.x;

		globalBestXs[dimIdx] = particleXs[getIdx(_newGBestIdx, dimIdx, dims.nDimension)];

		if (dimIdx == 0 && gBestIdx[0] != _newGBestIdx) { //update positions
			gBestIdx[0] = _newGBestIdx;

			bestParticleErrorSuccessIdx[0] = 0;
			bestParticleErrorSuccessIdx[1] = 0;
			bestParticleErrorSuccessIdx[2] = _newGBestIdx;
		}
	}

}

//#define CudaAssert( X ) if ( !(X) ) { printf( "Thread %d:%d failed assert at %s:%d!", blockIdx.x, threadIdx.x, __FILE__, __LINE__ ); return; }

void PSO::updateBest() {	
	findGBest<<<dim3(1, dims.nTasks), dims.nParticles, dims.nParticles * (sizeof(int) + sizeof(gpu_fp_t)), stream>>>(
		thrust::raw_pointer_cast(particleEvalValues.data()), 
		thrust::raw_pointer_cast(newGBestIdx.data()), 
		thrust::raw_pointer_cast(newGBestValue.data()), 
		dims);
	
	updateGBest<<<dim3(1, dims.nTasks), dims.nDimension, 0, stream>>>(
		thrust::raw_pointer_cast(globalBestXs.data()), 
		thrust::raw_pointer_cast(particleXs.data()),
		thrust::raw_pointer_cast(gBestValue.data()),
		thrust::raw_pointer_cast(newGBestValue.data()),
		thrust::raw_pointer_cast(gBestIdx.data()),
		thrust::raw_pointer_cast(newGBestIdx.data()), 
		thrust::raw_pointer_cast(bestParticleErrorSuccessIdx.data()), 
		dims);
	
//	using namespace std;
//	cout << bestParticleErrorSuccessIdx[0] << ' ' << bestParticleErrorSuccessIdx[1] << ' ' << bestParticleErrorSuccessIdx[2] << endl;
//	cout << bestParticleRho[0] << "  " << globalBestVal << endl;
//	cout << "xs = "<< globalBestXs[0] << ", " << globalBestXs[1] << endl;
//	cout << "vel = " << particleVs[getIdx(gBestIdx[0], 0, nDimension)] << ", " << particleVs[getIdx(gBestIdx[0], 1, nDimension)] << endl;

	// update personal best
	updatePBest<<<dims.nParticles, dim3(dims.nDimension, dims.nTasks), 0, stream>>>(
		thrust::raw_pointer_cast(particleEvalValues.data()),
		thrust::raw_pointer_cast(particleXs.data()), 
		thrust::raw_pointer_cast(particleBestEvalValues.data()), 
		thrust::raw_pointer_cast(particleBestXs.data()), 
		dims,
		thrust::raw_pointer_cast(bestParticleErrorSuccessIdx.data()) /*3*/,
		thrust::raw_pointer_cast(bestParticleRho.data()) /*1*/,
		successThreshold,
		failThreshold);

//S	cout << "best xs = " << particleBestXs[getIdx(gBestIdx[0], 0, nDimension)] << ", " << particleBestXs[getIdx(gBestIdx[0], 1, nDimension)] << endl;
}

void PSO::updateVelocitiesAndMove(){
	curandGenerateUniformDouble(m_prng, thrust::raw_pointer_cast(randoms.data()), randoms.size());

	updatePosAndVel<<<dims.nParticles, dim3(dims.nDimension, dims.nTasks), 0, stream>>>(thrust::raw_pointer_cast(particleXs.data())/*[N x nDim]*/, 
		thrust::raw_pointer_cast(particleBestXs.data()) /*[N x nDim]*/, 
		thrust::raw_pointer_cast(particleVs.data()) /*[N x nDim]*/, 
		thrust::raw_pointer_cast(particleEvalGradient.data()) /*[N x nDim]*/, 
		thrust::raw_pointer_cast(globalBestXs.data()) /*[nDim]*/, 
		thrust::raw_pointer_cast(xlDev.data()) /*[nDim]*/,
		thrust::raw_pointer_cast(xuDev.data()) /*[nDim]*/,
		dims, opts.inertia, opts.cpbest, opts.cgbest, opts.vmax,
		thrust::raw_pointer_cast(randoms.data()) /*[2 x N x nDim]*/,
		thrust::raw_pointer_cast(bestParticleErrorSuccessIdx.data()) /*3*/,
		thrust::raw_pointer_cast(bestParticleRho.data()) /*1*/);
}

void PSO::dump() {
	int nDimension = dims.nDimension;
	int nParticles = dims.nParticles;
	std::cout<<"nDimension = "<< nDimension << std::endl;
	std::cout<<"nParticles = "<< nParticles << std::endl;
	for (int i = 0; i < nParticles; i++) {
		std::cout<<"particle "<< i << std::endl;
		std::cout<<"  pos = [";
		for (int j = 0; j < nDimension; j++) {
			std::cout << particleXs[getIdx(i, j, nDimension)] << (j == nDimension - 1 ? "]" : ", ");
		}
		std::cout << " -> fitness = " << particleEvalValues[i] <<std::endl;
		std::cout<<"  vel = [";
		for (int j = 0; j < nDimension; j++) {
			std::cout << particleVs[getIdx(i, j, nDimension)] << (j == nDimension - 1 ? "]" : ", ");
		}
		std::cout << std::endl;
	}
}

PSO::~PSO() {
	curandDestroyGenerator(m_prng);
	cudaStreamDestroy ( stream) ;
}

void PSO::setParams(psoopts opts) {
	this->opts = opts;
}

void PSO::evaluateGradient() {
	// do nothing, zero gradient is ignored anyway
}