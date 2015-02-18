#ifndef _CUDA_PSO_CU_
#define _CUDA_PSO_CU_

#include <cuda_runtime.h>
//#include <cutil_inline.h>
#include <curand.h>
#include <curand_kernel.h>

#include <stdint.h>
typedef uint32_t u_int32_t;

//#include <helper_cuda.h>
#include <cstdio>

#include "cudaPSO.cuh"
#include "reductions.cu"
#include "utilities.h"
#include "cutil_compat.h"

//#include "MersenneTwister.cuh"



//NOTE To print debug messages uncomment the following line
//#define PRINT_DEBUG



///Number of array elements used for each particle, usually greater than the problem dimensionality
unsigned int actualParticleSize;

///CUDA Random states global array
curandState* devStates;


//available GPU memory tracking
	///Amount of used gobal memory
	unsigned long int usedGlobalMem;

	///Amount of free gobal memory
	unsigned long int freeGlobalMem;

	///Amount of available gobal memory
	unsigned long int totGlobalMem;


//********
//Constant Memory Data

	///starting coordinate of the hypercubical search space (Resides in GPU's constant memory)
	__constant__ float  c_minValue;

	///ending coordinate of the hypercubical search space (Resides in GPU's constant memory)
	__constant__ float  c_maxValue;

	///width of the hypercubical search space (Resides in GPU's constant memory)
	__constant__ float  c_deltaValue;

//********


//********
//Global Memory Data Arrays

	///pointer to the GPU's global-memory array containing the current position of all particles (from all swarms, in case of multi-swarm simulation)
	float        *g_positions;

	///pointer to the GPU's global-memory array containing the current personal best position of all particles
	float        *g_bestPositions;

	///pointer to the GPU's global-memory array containing the current velocity of all particles
	float        *g_velocities;

	///pointer to the GPU's global-memory array containing the current fitness of all particles
	float        *g_fitnesses;

	///pointer to the GPU's global-memory array containing the current personal best fitness of all particles
	float        *g_bestFitnesses;

	///pointer to the GPU's global-memory array containing the final global best fitness value
	float        *g_globalBestFitness;

	///pointer to the GPU's global-memory array containing the coordinates of the global best position of all swarms
	float        *g_globalBestPositions;


	///pointer to the GPU's global-memory array containing the indexes (for all particles) of the best neighbour (for the ring topology in this case)
	u_int32_t    *g_localBestIDs;

	///pointer to the GPU's global-memory array containing the index of the best particle
	u_int32_t    *g_globalBestID;

	///pointer to the GPU's global-memory array containing the flags saying to each particle whether to update their personal best
	u_int32_t    *g_update;

//********




//********
//Textures

	///GPU's texture interface used for fast acces to the update flags in global memory
	texture<unsigned int,   1, cudaReadModeElementType> t_texUpdatePositionFlags;

	///GPU's texture interface used for fast acces to the local best indices in global memory
	texture<unsigned int,   1, cudaReadModeElementType> t_texLocalBestIDs;

	///GPU's texture interface used for fast acces to the current particles's velocities in global memory
	texture<float,  1, cudaReadModeElementType> t_texVelocities;

	///GPU's texture interface used for fast acces to the current particles's positions in global memory
	texture<float,  1, cudaReadModeElementType> t_texPositions;

	///GPU's texture interface used for fast acces to the current particles's best positions in global memory
	texture<float,  1, cudaReadModeElementType> t_texBestPositions;

	///GPU's texture interface used for fast acces to the current particles's fitnesses in global memory
	texture<float,  1, cudaReadModeElementType> t_texFitnesses;

	///GPU's texture interface used for fast acces to the current particles's best fitnesses in global memory
	texture<float,  1, cudaReadModeElementType> t_texBestFitnesses;

//********




//*******************************************************************************************
//  DEVICE KERNELS
//*******************************************************************************************

//includes all kernels code...
#include "cudaPSO_kernels.cuh"
//includes fitnesses computation stuff...
#include "cudaPSO_fitnesses.cuh"



__global__ void setup_kernel(curandState *state, unsigned long seed)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	/* Each thread gets same seed, a different sequence number,
	no offset */
	curand_init(seed, id, 0, &state[id]);
}


//*******************************************************************************************
//  HOST ROUTINES
//*******************************************************************************************



//***********************************************************
//  DATA ALLOCATION ON GPU
//***********************************************************


/// Initialization of the GPU...
/// Here global variables pointing to device memory are initialized...
/// @param particlesNumber number of particles in the swarm
/// @param problemDimension dimensionality of the problem
/// @param numberOfGenerations number of generation to be performed during the optimization
__host__ void h_cudaPSO_Init(int particlesNumber, int problemDimension, int numberOfGenerations){
	#ifdef PRINT_DEBUG
		printf("Allocating data structures on GPU...\n");
	#endif
	int dim;

	//Determination of the total amount of global memory
	int devID = cutGetMaxGflopsDeviceId();
	cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, devID);
	totGlobalMem = deviceProp.totalGlobalMem;

	usedGlobalMem = 0;

	//To accomplish CUDA byte alignment requirements, we need data arrays with a number of elements
	//  for each particle which is a multiple of 16
	//The actual number of simulated problem dimension might be greater than the required one:
	//  during cost funziont evaluation, obviously, only the needed coordinates will be considered
	actualParticleSize = iAlignUp(problemDimension, 16);


	dim = particlesNumber * actualParticleSize * sizeof(float);
	#ifdef PRINT_DEBUG
		printf("\t - actualParticleSize = %d\n", actualParticleSize);
	#endif

	//Allocation of the positions array
	cudasafe(cudaMalloc( (void**) &g_positions, dim), "h_init_cudaPSO: cudaMalloc() execution failed\n");
	cudasafe(cudaMemset( g_positions, 0, dim), "h_init_cudaPSO: cudaMemset() execution failed\n");
	cudasafe(cudaBindTexture(NULL, t_texPositions, g_positions, dim), "h_init_cudaPSO: cudaBindTexture() execution failed\n");
	usedGlobalMem += dim;

	//Allocation of the best positions array
	cudasafe(cudaMalloc( (void**) &g_bestPositions, dim), "cudaPSO: cudaMalloc() execution failed\n");
	cudasafe(cudaMemset( g_bestPositions, 0, dim), "cudaPSO: cudaMemset() execution failed\n");
	cudasafe(cudaBindTexture(NULL, t_texBestPositions, g_bestPositions, dim), "h_cudaPSOBindBestPositionsTextures: cudaBindTexture() execution failed\n");
	usedGlobalMem += dim;

	//Allocation of the velocities array
	cudasafe(cudaMalloc( (void**) &g_velocities, dim), "cudaPSO: cudaMalloc() execution failed\n");
	cudasafe(cudaMemset( g_velocities, 0, dim), "cudaPSO: cudaMemset() execution failed\n");
	cudasafe(cudaBindTexture(NULL, t_texVelocities, g_velocities, dim), "h_cudaPSOBindPositionsTextures: cudaBindTexture() execution failed\n");
	usedGlobalMem += dim;


	dim = particlesNumber * sizeof(float);

	//Allocation of the fitnesses array
	cudaMalloc( (void**) &g_fitnesses, dim);
	cutilCheckMsg("cudaPSO: cudaMalloc() execution failed\n");
	cudaMemset( g_fitnesses, 0, dim);
	cutilCheckMsg("cudaPSO: cudaMemset() execution failed\n");
	cudaBindTexture(NULL, t_texFitnesses, g_fitnesses, dim);
	cutilCheckMsg("h_cudaPSOBindFitnessesTextures: cudaBindTexture() execution failed\n");
	usedGlobalMem += dim;

	//Allocation of the best fitnesses array
	cudaMalloc( (void**) &g_bestFitnesses, dim);
	cutilCheckMsg("cudaPSO: cudaMalloc() execution failed\n");
	cudaMemset( g_bestFitnesses, 0, dim);
	cutilCheckMsg("cudaPSO: cudaMemset() execution failed\n");
	cudaBindTexture(NULL, t_texBestFitnesses, g_bestFitnesses, dim);
	cutilCheckMsg("h_cudaPSOBindBestFitnessesTextures: cudaBindTexture() execution failed\n");
	usedGlobalMem += dim;


	dim = particlesNumber * sizeof(unsigned int);
	
	//Allocation of the local best ids array
	cudaMalloc( (void**) &g_localBestIDs, dim);
	cutilCheckMsg("cudaPSO: cudaMalloc() execution failed\n");
	cudaMemset( g_localBestIDs, 0, dim);
	cutilCheckMsg("cudaPSO: cudaMemset() execution failed\n");
	cudaBindTexture(NULL, t_texLocalBestIDs, g_localBestIDs, dim);
	cutilCheckMsg("h_cudaPSO_BindUpdatePositionFlagsTexture: cudaBindTexture() execution failed\n");
	usedGlobalMem += dim;

	//Allocation of the update flag array
	cudaMalloc( (void**) &g_update, dim);
	cutilCheckMsg("cudaPSO: cudaMalloc() execution failed\n");
	cudaMemset( g_update, 0, dim);
	cutilCheckMsg("cudaPSO: cudaMemset() execution failed\n");
	cudaBindTexture(NULL, t_texUpdatePositionFlags, g_update, dim);
	cutilCheckMsg("h_cudaPSO_BindUpdatePositionFlagsTexture: cudaBindTexture() execution failed\n");
	usedGlobalMem += dim;


	dim = actualParticleSize * sizeof(float);

	//Allocation of the global best positions array
	cudaMalloc( (void**) &g_globalBestPositions, dim);
	cutilCheckMsg("cudaPSO: cudaMalloc() execution failed\n");
	cudaMemset( g_globalBestPositions, 0, dim);
	cutilCheckMsg("cudaPSO: cudaMemset() execution failed\n");
	usedGlobalMem += dim;


	dim = sizeof(float);

	//Allocation of the global best fitnesse value
	cudaMalloc( (void**) &g_globalBestFitness, dim);
	cutilCheckMsg("cudaPSO: cudaMalloc() execution failed\n");
	cudaMemset( g_globalBestFitness, 0, dim);
	cutilCheckMsg("cudaPSO: cudaMemset() execution failed\n");
	usedGlobalMem += dim;


	dim = sizeof(u_int32_t);

	//Allocation of the global best id
	cudaMalloc( (void**) &g_globalBestID, dim);
	cutilCheckMsg("cudaPSO: cudaMalloc() execution failed\n");
	cudaMemset( g_globalBestID, 0, dim);
	cutilCheckMsg("cudaPSO: cudaMemset() execution failed\n");
	usedGlobalMem += dim;

	dim = particlesNumber * actualParticleSize * sizeof(curandState);

	//Allocation of the CUDA random states
	cudaMalloc((void **)&devStates, dim);
	cutilCheckMsg("cudaPSO: cudaMalloc() execution failed\n");
	usedGlobalMem += dim;

	freeGlobalMem = totGlobalMem - usedGlobalMem;

	/* Setup prng states */
	setup_kernel<<<particlesNumber, actualParticleSize>>>(devStates, time(NULL));


	h_initFitnessFunctions(g_positions, particlesNumber, actualParticleSize);


	#ifdef PRINT_DEBUG
		printf("\t - totalGlobalMem = %ld\n", totGlobalMem);
		printf("\t - usedGlobalMem = %ld\n", usedGlobalMem);
		printf("\t - usedGlobalMemForRandNums = %ld\n", dim);
		printf("\t - freeGlobalMem = %ld\n", freeGlobalMem);
		printf("Done!\n");
	#endif
}

/// Frees GPU's resources
__host__ void h_cudaPSO_Free(){

	//**********************
	//  TEXTURES UN-BINDINGS
	//**********************

	clearFitnessFunctions();

	cudaUnbindTexture(t_texVelocities);
	cutilCheckMsg("h_cudaPSO_Free: cudaUnbindTexture() execution failed\n");
	cudaUnbindTexture(t_texPositions);
	cutilCheckMsg("h_cudaPSO_Free: cudaUnbindTexture() execution failed\n");
	cudaUnbindTexture(t_texBestPositions);
	cutilCheckMsg("h_cudaPSO_Free: cudaUnbindTexture() execution failed\n");
	cudaUnbindTexture(t_texFitnesses);
	cutilCheckMsg("h_cudaPSO_Free: cudaUnbindTexture() execution failed\n");
	cudaUnbindTexture(t_texBestFitnesses);
	cutilCheckMsg("h_cudaPSO_Free: cudaUnbindTexture() execution failed\n");
	cudaUnbindTexture(t_texUpdatePositionFlags);
	cutilCheckMsg("h_cudaPSO_Free: cudaUnbindTexture() execution failed\n");
	cudaUnbindTexture(t_texLocalBestIDs);
	cutilCheckMsg("h_cudaPSO_Free: cudaUnbindTexture() execution failed\n");

	//**********************
	//  ARRAYS DE-ALLOCATION
	//**********************

	cudaFree(g_positions);
	cutilCheckMsg("h_cudaPSO_Free: cudaFree() execution failed\n");
	cudaFree(g_bestPositions);
	cutilCheckMsg("h_cudaPSO_Free: cudaFree() execution failed\n");
	cudaFree(g_velocities);
	cutilCheckMsg("h_cudaPSO_Free: cudaFree() execution failed\n");
	cudaFree(g_fitnesses);
	cutilCheckMsg("h_cudaPSO_Free: cudaFree() execution failed\n");
	cudaFree(g_bestFitnesses);
	cutilCheckMsg("h_cudaPSO_Free: cudaFree() execution failed\n");
	cudaFree(g_globalBestPositions);
	cutilCheckMsg("h_cudaPSO_Free: cudaFree() execution failed\n");
	cudaFree(g_globalBestFitness);
	cutilCheckMsg("h_cudaPSO_Free: cudaFree() execution failed\n");
	cudaFree(g_localBestIDs);
	cutilCheckMsg("h_cudaPSO_Free: cudaFree() execution failed\n");
	cudaFree(g_globalBestID);
	cutilCheckMsg("h_cudaPSO_Free: cudaFree() execution failed\n");
	cudaFree(g_update);
	cutilCheckMsg("h_cudaPSO_Free: cudaFree() execution failed\n");
	cudaFree(devStates);
	cutilCheckMsg("h_cudaPSO_Free: cudaFree() execution failed\n");
}





///@brief wrapper to appropriately call the g_findGlobalBest() kernel
///@param g_globalBestFitness pointer to the GPU's global-memory array containing the final global best fitness value
///@param g_globalBestID pointer to the GPU's global-memory array containing the index of the best particle
///@param numberOfParticles swarm's size
///@param finalBestsUpdateGrid definition of the blocks grid for this kernel (containing the number of thread-blocks for each dimension of the grid)
///@param finalBestsUpdateBlock definition of the thread blocks for this kernel (containing the number of threads for each dimension of the block)
__host__ void h_findGlobalBest(float* g_globalBestFitness, u_int32_t* g_globalBestID, int numberOfParticles, dim3 finalBestsUpdateGrid, dim3 finalBestsUpdateBlock){

	switch (finalBestsUpdateBlock.x){
		case   8:	g_findGlobalBest<  8><<<finalBestsUpdateGrid, finalBestsUpdateBlock>>>(g_globalBestFitness, g_globalBestID, numberOfParticles);
				break;
		case  16:	g_findGlobalBest< 16><<<finalBestsUpdateGrid, finalBestsUpdateBlock>>>(g_globalBestFitness, g_globalBestID, numberOfParticles);
				break;
		case  32:	g_findGlobalBest< 32><<<finalBestsUpdateGrid, finalBestsUpdateBlock>>>(g_globalBestFitness, g_globalBestID, numberOfParticles);
				break;
		case  64:	g_findGlobalBest< 64><<<finalBestsUpdateGrid, finalBestsUpdateBlock>>>(g_globalBestFitness, g_globalBestID, numberOfParticles);
				break;
		case 128:	g_findGlobalBest<128><<<finalBestsUpdateGrid, finalBestsUpdateBlock>>>(g_globalBestFitness, g_globalBestID, numberOfParticles);
				break;
		case 256:	g_findGlobalBest<256><<<finalBestsUpdateGrid, finalBestsUpdateBlock>>>(g_globalBestFitness, g_globalBestID, numberOfParticles);
				break;
		case 512:	g_findGlobalBest<512><<<finalBestsUpdateGrid, finalBestsUpdateBlock>>>(g_globalBestFitness, g_globalBestID, numberOfParticles);
				break;
	}
}




/*
__host__ void h_cudaPSOPrintBestFitnesses(u_int32_t* g_globalBestID, float* g_bestFitnesses, float* g_bestPositions, float* g_globalBestFitness, float* g_globalBestPositions, float* g_positions, int particlesNumber, int actualParticleSize, int problemDimension){
	
	//Load fintnesses values from GPU and print them to screen

	
	//float* fitnesses = (float*) malloc(particlesNumber * sizeof(float));
	//cudaMemcpy(fitnesses, g_bestFitnesses, particlesNumber * sizeof(float), cudaMemcpyDeviceToHost);
	//cutilCheckMsg("h_cudaPSOPrintBestFitnesses: cudaMemcpy() execution failed\n");
	//printf(" - Fitnesses:\n    ");
	//for(int i = 0; i < particlesNumber; ++i)
	//	printf("%e ", fitnesses[i]);
	//printf("\n");
	//free(fitnesses);

	u_int32_t h_globalBestID;
	cudaMemcpy(&h_globalBestID, g_globalBestID, sizeof(u_int32_t), cudaMemcpyDeviceToHost);
	printf(" - Best particle ID: %u\n", h_globalBestID);

	float h_globalBestFitness;
	cudaMemcpy(&h_globalBestFitness, g_globalBestFitness, sizeof(float), cudaMemcpyDeviceToHost);
	printf(" - Global best fitness: %e\n", h_globalBestFitness);

	int dim = actualParticleSize * sizeof(float);
	float* h_globalBestPositions = (float*) malloc(dim);
	cudaMemcpy(h_globalBestPositions, g_globalBestPositions, dim, cudaMemcpyDeviceToHost);
	printf("      - Global best positions:\n");
	for(int m = 0; m < problemDimension; ++m)
		printf("          %2d: % 12.10e\n", m+1, h_globalBestPositions[m]);
	printf("\n");
	delete h_globalBestPositions;

	printf("particlesNumber = %d\n", particlesNumber);
	printf("problemDimension = %d\n", problemDimension);
	printf("actualParticleSize = %d\n", actualParticleSize);

	dim = particlesNumber * actualParticleSize;

	float* h_positions = (float*) malloc(dim * sizeof(float));
	cudaMemcpy(h_positions, g_positions, dim * sizeof(float), cudaMemcpyDeviceToHost);
	cutilCheckMsg("h_cudaPSOPrintBestFitnesses: cudaMemcpy() execution failed\n");

	printf("      - Particle's positions:\n");
	for(int j = 0; j < particlesNumber; ++j){
		printf("         ");
		for(int m = 0; m < problemDimension; ++m)
			printf("% 3.1f", h_positions[j * actualParticleSize + m]);
		printf("\n");
	}
	printf("\n");
	free(h_positions);


	dim = particlesNumber * actualParticleSize;
	float* h_bestPositions = (float*) malloc(dim * sizeof(float));
	cudaMemcpy(h_bestPositions, g_bestPositions, dim * sizeof(float), cudaMemcpyDeviceToHost);
	cutilCheckMsg("h_cudaPSOPrintBestFitnesses: cudaMemcpy() execution failed\n");

	printf("      - Personal best positions:\n");
	for(int j = 0; j < particlesNumber; ++j){
		printf("         ");
		for(int m = 0; m < problemDimension; ++m)
			printf("% 3.1f", h_bestPositions[j * actualParticleSize + m]);
		printf("\n");
	}
	printf("\n");
	free(h_bestPositions);
}
*/




//************************
//  OPTIMIZATION CALL-BACK
//************************

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
extern "C" __host__ void h_cudaPSO_Optimize(
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
	)
{

	//kernel parameters for particles initialization:
	//	- one thread block for each particle
	//	- one thread for each problem dimension
	dim3 initializationGrid(particlesNumber, 1);
	dim3 initializationBlock(actualParticleSize,1,1);


	//kernel parameters for positions update:
	//	- one thread block for each particle
	//	- one thread for each problem dimension
	dim3 updateGrid(particlesNumber, 1, 1);
	dim3 updateBlock(actualParticleSize,1,1);


	//kernel parameters for local bests update:
	//	- one thread block
	//	- one thread for each particle
	dim3 bestsUpdateGrid(1, 1, 1);
	dim3 bestsUpdateBlock(particlesNumber,1,1);
	unsigned int bestsUpdateSharedAmount = (particlesNumber + 2) * sizeof(float);


	//kernel parameters for the global bests update:
	//	- one thread block
	//	- the number of threads is chosen to have enough thread to perform
	//	  a parallel reduction of the fitness values
	dim3 globalBestUpdateGrid(1, 1, 1);
	int thNum = (int) rint( pow(2.0f, ceil( log2( (float) particlesNumber) ) ) );
	dim3 globalBestUpdateBlock(thNum,1,1);


	//kernel parameters for the computation of fitnesses values:
	//	- one thread block for each individual
	//	- one thread for dimension of the problem
	dim3 calculateFitnessesGrid(particlesNumber, 1, 1);
	thNum = (int) rint( pow(2.0f, ceil( log2( (float) problemDimension) ) ) );
	thNum = max(8, thNum);
	dim3 calculateFitnessesBlock(thNum,1,1);


	//kernel parameters for the global bests update:
	//	- one thread block
	//	- one thread for each dimension of the problem
	dim3 globalBestCopyGrid(1, 1, 1);
	dim3 globalBestCopyBlock(actualParticleSize,1,1);


	//CUDA routines to time events
	cudaEvent_t start, stop;
	cudasafe( cudaEventCreate(&start), "h_cudaPSO_Optimize: cudaEventCreate() execution failed\n");
	cudasafe( cudaEventCreate(&stop), "h_cudaPSO_Optimize: cudaEventCreate() execution failed\n");

	//printf("Starting Optimization...");

	//Start timing...
	cudaEventRecord(start,0);


	//Set up search space limits
	cudaMemcpyToSymbol(c_minValue,   &minValue,   sizeof(float), 0, cudaMemcpyHostToDevice);
	cutilCheckMsg("h_cudaPSO_Optimize: cudaMemcpyToSymbol() execution failed\n");
	cudaMemcpyToSymbol(c_maxValue,   &maxValue,   sizeof(float), 0, cudaMemcpyHostToDevice);
	cutilCheckMsg("h_cudaPSO_Optimize: cudaMemcpyToSymbol() execution failed\n");
	cudaMemcpyToSymbol(c_deltaValue, &deltaValue, sizeof(float), 0, cudaMemcpyHostToDevice);
	cutilCheckMsg("h_cudaPSO_Optimize: cudaMemcpyToSymbol() execution failed\n");



	//Particles initialization
	g_initParticles<<<initializationGrid, initializationBlock>>>(g_positions, g_bestPositions, g_velocities, devStates);
	cutilCheckMsg("h_cudaPSO_Optimize: g_initParticles() execution failed\n");


	//Set to zero the update flags
	cudaMemset(g_update, 0, particlesNumber * sizeof(u_int32_t));


	//First fitnesses evaluation
	h_calculateFitnessesValues(functionID, g_fitnesses, actualParticleSize, problemDimension, calculateFitnessesGrid, calculateFitnessesBlock);
	cutilCheckMsg("h_cudaPSO_Optimize: h_calculateFitnessesValues() execution failed\n");


	//First Local bests update
	g_firstBestsUpdate<<<bestsUpdateGrid, bestsUpdateBlock, bestsUpdateSharedAmount>>>(g_bestFitnesses, g_localBestIDs);
	cutilCheckMsg("h_cudaPSO_Optimize: g_firstBestsUpdate() execution failed\n");


	//Generations main cycle
	for(unsigned int generationNumber = 1; generationNumber < numberOfGenerations; ++generationNumber){

		//Position Update
		g_positionsUpdate<<<updateGrid, updateBlock>>>(W, C1, C2, g_positions, g_bestPositions, g_velocities, devStates);
		cutilCheckMsg("h_cudaPSO_Optimize: g_positionsUpdate() execution failed\n");


		//Fitnesses evaluation
		h_calculateFitnessesValues(functionID, g_fitnesses, actualParticleSize, problemDimension, calculateFitnessesGrid, calculateFitnessesBlock);
		cutilCheckMsg("h_cudaPSO_Optimize: h_calculateFitnessesValues() execution failed\n");


		//Local bests update
		g_bestsUpdate<<<bestsUpdateGrid, bestsUpdateBlock, bestsUpdateSharedAmount>>>(g_bestFitnesses, g_localBestIDs, g_update);
		cutilCheckMsg("h_cudaPSO_Optimize: g_bestsUpdate() execution failed\n");

	}


	//Global best determination
	h_findGlobalBest(g_globalBestFitness, g_globalBestID, particlesNumber, globalBestUpdateGrid, globalBestUpdateBlock);
	cutilCheckMsg("h_cudaPSO_Optimize: h_findGlobalBest() execution failed\n");


	//Copy global best positions
	g_globalBestCopy<<<globalBestCopyGrid, globalBestCopyBlock>>>(g_globalBestPositions, g_globalBestID);
	cutilCheckMsg("h_cudaPSO_Optimize: g_copyBests() execution failed\n");

	cudaThreadSynchronize();
	//Stop timing...
	cudaEventRecord(stop,0);
	//waits for the stop event to be recorded...
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);


	//Print the current best fitnesses
	//h_cudaPSOPrintBestFitnesses(g_globalBestID, g_bestFitnesses, g_bestPositions, g_globalBestFitness, g_globalBestPositions, g_positions, particlesNumber, actualParticleSize, problemDimension);
	//cutilCheckMsg("h_cudaPSO_Optimize: h_cudaPSOPrintBestFitnesses() execution failed\n");


	//Retrieves the global best fitness value
	cudaMemcpy(h_globalBestFitness, g_globalBestFitness, sizeof(float), cudaMemcpyDeviceToHost);

	//Retrieves the global best position
	cudaMemcpy(h_globalBestPosition, g_globalBestPositions, problemDimension * sizeof(float), cudaMemcpyDeviceToHost);

	//Prints the amount of time elapsed for optimization
	//printf("Elapsed time = %f ms\n", elapsedTime);
	printf("%d %d %d %d %f %e %d\n", 1 /*swarmsNumber*/, particlesNumber, problemDimension, numberOfGenerations, elapsedTime, *h_globalBestFitness, functionID);
}


__host__ int cutGetMaxGflopsDeviceId()
{
	int device_count = 0;
	cudaGetDeviceCount( &device_count );

	cudaDeviceProp device_properties;
	int max_gflops_device = 0;
	int max_gflops = 0;
	
	int current_device = 0;
	cudaGetDeviceProperties( &device_properties, current_device );
	max_gflops = device_properties.multiProcessorCount * device_properties.clockRate;
	++current_device;

	while( current_device < device_count )
	{
		cudaGetDeviceProperties( &device_properties, current_device );
		int gflops = device_properties.multiProcessorCount * device_properties.clockRate;
		if( gflops > max_gflops )
		{
			max_gflops        = gflops;
			max_gflops_device = current_device;
		}
		++current_device;
	}

	return max_gflops_device;
}

__host__ void cudasafe( cudaError_t error, char* message)
{
   if(error!=cudaSuccess) { fprintf(stderr,"ERROR: %s : %i\n",message,error); exit(-1); }
}
 
 
__host__ void cutilCheckMsg( const char *errorMessage)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) {
        fprintf(stderr, "cutilCheckMsg() CUTIL CUDA error : %s : %s.\n",
                errorMessage, cudaGetErrorString( err) );
        exit(-1);
    }
#ifdef _DEBUG
    err = cudaThreadSynchronize();
    if( cudaSuccess != err) {
        fprintf(stderr, "cutilCheckMsg cudaThreadSynchronize error: %s : %s.\n",
                errorMessage, cudaGetErrorString( err) );
        exit(-1);
    }
#endif
}

#endif


