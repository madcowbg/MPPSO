#ifndef _CUDA_PSO_KERNELS_CU_
#define _CUDA_PSO_KERNELS_CU_

#include <cuda_runtime.h>
//#include <cutil_inline.h>
#include "utilities.h"
#include "reductions.cu"
#include "cudaPSO.cuh"
#include "cutil_compat.h"



///Number of fitness functions defined inside this file
#define NUMBER_OF_FUNCTIONS 3


///Texture interface for fast acces to the positions array
texture<float,  1, cudaReadModeElementType> t_texPos;


///Function pointer which permits to dinamically change the fitness function to be evaluated
typedef void (*h_FitnessFunctionPointer) (float* g_fitnesses, int actualParticleSize, int problemDimension, dim3 calculateFitnessesGrid, dim3 calculateFitnessesBlock);


///array of pointers containing the references to all the implemented fitness functions
h_FitnessFunctionPointer h_calculateFitnessesPointer[NUMBER_OF_FUNCTIONS];


///@brief parallel inline evaluation of the Sphere function
///  All the threads load one dimension and calculate one addend.
///  All computed addends are to be later reduced to a single sum
///@param s_addends shared memory pointer to single addend results (to be later reduced)
///@param tid thread index
///@param posID index of the value to be loaded from the global array containing the particles' positions (current coordinate)
///@param problemDimension dimension of the search space
__device__ void sphere(float* s_addends, u_int32_t tid, u_int32_t posID, int problemDimension){
	
	if (tid < problemDimension){
		float x = tex1Dfetch(t_texPos, posID);
		s_addends[tid] =  x*x;
	}
}


///@brief parallel inline evaluation of the Rastrigin function
///  All the threads load one dimension and calculate one addend.
///  All computed addends are to be later reduced to a single sum
///@param s_addends shared memory pointer to single addend results (to be later reduced)
///@param tid thread index
///@param posID index of the value to be loaded from the global array containing the particles' positions (current coordinate)
///@param problemDimension dimension of the search space
__device__ void rastrigin(float* s_addends, u_int32_t tid, u_int32_t posID, int problemDimension){
	if (tid < problemDimension){
		#define rastriginA 10.0f
		#define rastriginW  2.0f
		float x = tex1Dfetch(t_texPos, posID);
		s_addends[tid] =  x*x - rastriginA * __cosf( rastriginW * x ) + rastriginA;
	}
}

///@brief parallel inline evaluation of the Rosenbrock function
///  All the threads firstly cooperate to load the current particle's position to shared memory and secondly calculate one addend.
///  All computed addends are to be later reduced to a single sum
///@tparam threadNum number of threads composing each thread block
///@param s_addends shared memory pointer to single addend results (to be later reduced)
///@param tid thread index
///@param posID index of the value to be loaded from the global array containing the particles' positions (current coordinate)
///@param problemDimension dimension of the search space
template <u_int32_t threadNum>
__device__ void rosenbrock(float* s_addends, u_int32_t tid, u_int32_t posID, int problemDimension){
	__shared__ float s_x[threadNum];
	if (tid < problemDimension )
		s_x[tid] = tex1Dfetch(t_texPos, posID);

	if (tid < (problemDimension - 1) ){
		float a = s_x[tid];
		float b = 1.0 - a;
		a = s_x[tid+1] - a * a;
		s_addends[tid] = 100.0 * a*a + b*b;
	}
}




//****************************************
//Kernel code for fitnesses evaluation
//****************************************
///@brief cuda kernel to compute all the fitness values in parallel (at once)
///@tparam threadNum number of threads composing each thread block
///@tparam functionID index of the fitness function to be optimized
///@param g_fitnesses pointer to a global memory array where to store computed fitness values
///@param actualParticlesSize actual dimension of one particle's position inside the positions array (usually greater than the dimension of the search space)
///@param problemDimension dimension of the search space
template <u_int32_t threadNum, int functionID>
static __global__ void g_fitnessesEvaluation(float* g_fitnesses, int actualParticleSize, int problemDimension) {

	//blockIdx.x represents the index of the particle among all swarms
	//threadIdx.x represents the index of the dimension of the problem (in some way...)

	//gridDim.x represents the NUMBER_OF_SWARMS * NUMBER_OF_PARTICLES
	//blockDim.x represents the number of threads which is equal do 2^(floor( log2 PROBLEM_DIMENSIONS) )

	u_int32_t tid = threadIdx.x;
	//fitnessID is used to access d_positions and d_fitnesses arrays
	u_int32_t particleID = blockIdx.x;
	//posID is used to access d_positions array
	u_int32_t posID = IMUL( particleID, actualParticleSize) + tid;


	//Shared memory seen by every thread in the same thread block
	__shared__ float s_addends[threadNum];
	s_addends[tid] =  0.0;

	switch (functionID){
		case 0: sphere   (s_addends, tid, posID, problemDimension);
			break;
		case 1: rastrigin(s_addends, tid, posID, problemDimension);
			break;
		case 2: rosenbrock<threadNum>(s_addends, tid, posID, problemDimension);
			break;
	}

	//Parallel reduction (addition) of partial sums (single addends)
	reduceToSum<float,threadNum>(s_addends, tid);

	//Only the first thread updates the fitness value
	if (tid == 0)
		g_fitnesses[particleID] = s_addends[0];
}



///@brief host function to call the g_fitnessesEvaluation kernel
/// based on the current swarm parameter, the kernel is called with the appropriate parameters
///@tparam functionID index of the fitness function to be optimized
///@param g_fitnesses pointer to a global memory array where to store computed fitness values
///@param actualParticlesSize actual dimension of one particle's position inside the positions array (usually greater than the dimension of the search space)
///@param problemDimension dimension of the search space
///@param calculateFitnessesGrid definition of the blocks grid for this kernel (containing the number of thread-blocks for each dimension of the grid)
///@param calculateFitnessesBlock definition of the thread blocks for this kernel (containing the number of threads for each dimension of the block)
template <int functionID>
__host__ void h_evaluateFitnesses(float* g_fitnesses, int actualParticleSize, int problemDimension, dim3 calculateFitnessesGrid, dim3 calculateFitnessesBlock){

	switch (calculateFitnessesBlock.x){
		case   8:	g_fitnessesEvaluation<  8, functionID><<<calculateFitnessesGrid, calculateFitnessesBlock>>>(g_fitnesses, actualParticleSize, problemDimension);
				break;
		case  16:	g_fitnessesEvaluation< 16, functionID><<<calculateFitnessesGrid, calculateFitnessesBlock>>>(g_fitnesses, actualParticleSize, problemDimension);
				break;
		case  32:	g_fitnessesEvaluation< 32, functionID><<<calculateFitnessesGrid, calculateFitnessesBlock>>>(g_fitnesses, actualParticleSize, problemDimension);
				break;
		case  64:	g_fitnessesEvaluation< 64, functionID><<<calculateFitnessesGrid, calculateFitnessesBlock>>>(g_fitnesses, actualParticleSize, problemDimension);
				break;
		case 128:	g_fitnessesEvaluation<128, functionID><<<calculateFitnessesGrid, calculateFitnessesBlock>>>(g_fitnesses, actualParticleSize, problemDimension);
				break;
		case 256:	g_fitnessesEvaluation<256, functionID><<<calculateFitnessesGrid, calculateFitnessesBlock>>>(g_fitnesses, actualParticleSize, problemDimension);
				break;
		case 512:	g_fitnessesEvaluation<512, functionID><<<calculateFitnessesGrid, calculateFitnessesBlock>>>(g_fitnesses, actualParticleSize, problemDimension);
				break;
	}
}

///@brief wrapper to call the correct h_evaluateFitnesses() method based on the functionID parameter
///@param functionID index of the fitness function to be optimized
///@param g_fitnesses pointer to a global memory array where to store computed fitness values
///@param actualParticlesSize actual dimension of one particle's position inside the positions array (usually greater than the dimension of the search space)
///@param problemDimension dimension of the search space
///@param calculateFitnessesGrid definition of the blocks grid for this kernel (containing the number of thread-blocks for each dimension of the grid)
///@param calculateFitnessesBlock definition of the thread blocks for this kernel (containing the number of threads for each dimension of the block)
__host__ void h_calculateFitnessesValues(int functionID, float* g_fitnesses, int actualParticleSize, int problemDimension, dim3 calculateFitnessesGrid, dim3 calculateFitnessesBlock){
	(*h_calculateFitnessesPointer[functionID]) (g_fitnesses, actualParticleSize, problemDimension, calculateFitnessesGrid, calculateFitnessesBlock);
}



///@brief initializes this code module
///   sets the functions pointers appropriately and binds the texture interface onto the positios array
///@param g_positions pointer to a global memory array where all the current positions/coordinates of all the particles are stored
///@param particlesNumber dimension of the swarm
///@param actualParticlesSize actual dimension of one particle's position inside the positions array (usually greater than the dimension of the search space)
__host__ void h_initFitnessFunctions(float* g_positions, int particlesNumber, int actualParticleSize){
	int dim = particlesNumber * actualParticleSize * sizeof(float);
	cudaBindTexture(NULL, t_texPos, g_positions, dim);
	cutilCheckMsg("h_initFitnessFunctions: cudaBindTexture() execution failed\n");

	h_calculateFitnessesPointer[0] = h_evaluateFitnesses<0>;
	h_calculateFitnessesPointer[1] = h_evaluateFitnesses<1>;
	h_calculateFitnessesPointer[2] = h_evaluateFitnesses<2>;
}

///@brief unbinds the local texture interface to global memory positions
__host__ void clearFitnessFunctions(){
	cudaUnbindTexture(t_texPos);
	cutilCheckMsg("h_cudaPSO_Free: cudaUnbindTexture() execution failed\n");
}



#endif


