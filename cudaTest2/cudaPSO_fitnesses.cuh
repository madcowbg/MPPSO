#ifndef _CUDA_PSO_FITNESSES_CUH_
#define _CUDA_PSO_FITNESSES_CUH_

#include <cuda_runtime.h>
//#include <cutil_inline.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math_constants.h>


///@brief initializes this code module
///   sets the functions pointers appropriately and binds the texture interface onto the positios array
///@param g_positions pointer to a global memory array where all the current positions/coordinates of all the particles are stored
///@param particlesNumber dimension of the swarm
///@param actualParticlesSize actual dimension of one particle's position inside the positions array (usually greater than the dimension of the search space)
__host__ void h_initFitnessFunctions(float* g_positions, int particlesNumber, int actualParticleSize);



///@brief wrapper to call the correct h_evaluateFitnesses() method based on the functionID parameter
///@param functionID index of the fitness function to be optimized
///@param g_fitnesses pointer to a global memory array where to store computed fitness values
///@param actualParticlesSize actual dimension of one particle's position inside the positions array (usually greater than the dimension of the search space)
///@param problemDimension dimension of the search space
///@param calculateFitnessesGrid definition of the blocks grid for this kernel (containing the number of thread-blocks for each dimension of the grid)
///@param calculateFitnessesBlock definition of the thread blocks for this kernel (containing the number of threads for each dimension of the block)
__host__ void h_calculateFitnessesValues(int functionID, float* g_fitnesses, int actualParticleSize, int problemDimension, dim3 calculateFitnessesGrid, dim3 calculateFitnessesBlock);


///@brief unbinds the local texture interface to global memory positions
__host__ void clearFitnessFunctions();

#endif


