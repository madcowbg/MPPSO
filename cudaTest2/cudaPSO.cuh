#ifndef _CUDA_PSO_CUH_
#define _CUDA_PSO_CUH_

#include <cuda_runtime.h>
//#include <cutil.h>
//YS:
#include <curand.h>

#include <stdint.h>
typedef uint32_t u_int32_t;

extern "C" {

/// Initialization of the GPU...
/// Here global variables pointing to device memory are initialized...
/// @param particlesNumber number of particles in the swarm
/// @param problemDimension dimensionality of the problem
/// @param numberOfGenerations number of generation to be performed during the optimization
__host__ void h_cudaPSO_Init(int particlesNumber, int problemDimension, int numberOfGenerations);


/// Frees GPU's resources
__host__ void h_cudaPSO_Free();

///Realizes the actual optimization calling appropriately all the cuda kernels involved in this process
///@param functionID Index of the function to be optimized
///@param numberOfGenerations Number of generations/iterations in one optimization step
///@param particlesNumber Number of particles belonging to this swarm
///@param problemDimension Number of parameters to be optimized
///@param W Inertia weight (PSO algorithm)
///@param C1 Cognitive attraction factor (PSO algorithm)
///@param C2 Social attraction factor (PSO algorithm)
///@param minValue Lower limit for each dimension of the search space
///@param maxValue Upper limit for each dimension of the search space
///@param deltaValue Width of each dimension of the search space
///@param h_globalBestFitness pointer to an host variable where to store the final global best fitness value
///@param h_globalBestPosition Pointer to an host array where to store the final result (final global best position)
__host__ void h_cudaPSO_Optimize(
	int functionID,
	int numberOfGenerations,
	int particlesNumber,
	int problemDimension,
	float W,
	float C1,
	float C2,
	float minValue,
	float maxValue,
	float deltaValue,
	float* h_globalBestFitness,
	float* h_globalBestPosition
	);
}

#endif


