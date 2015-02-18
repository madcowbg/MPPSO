#ifndef _CUDA_PSO_KERNELS_CU_
#define _CUDA_PSO_KERNELS_CU_

#include <stdint.h>

#include <cuda_runtime.h>
//#include <cutil_inline.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#include "utilities.h"
#include "reductions.cu"
#include "cudaPSO_fitnesses.cuh"
#include "cudaPSO.cuh"

//Chose whether to minimize or maximize
//Comment the following line to perform a minimization
//#define MAXIMIZE

///Change this define appropriately to select between maximisation and minimisation
#ifdef MAXIMIZE
	#define BETTER_THAN >
#else
	#define BETTER_THAN <
#endif




//*******************************************************************************************
//  DEVICE KERNELS
//*******************************************************************************************



//****************************************
//Kernel code for particles initialization
//****************************************
///@brief cuda kernel to initialize the position and the velocity of all particles in parallel (at once)
/// Each thread initializes only one coordinate value of a particle's pose/velocity.
/// Hence the parallelization is not at the particles level but even at a lower level
///@param g_positions pointer to the GPU's global-memory array containing the current position of all particles (from all swarms, in case of multi-swarm simulation)
///@param g_bestPositions pointer to the GPU's global-memory array containing the current personal best position of all particles
///@param g_velocities pointer to the GPU's global-memory array containing the current velocity of all particles
///@param g_rand pointer to the GPU's global-memory array containing the pseudo-random numbers to be consumed by the simulation
static __global__ void g_initParticles(float*  g_positions, float*  g_bestPositions, float*  g_velocities, curandState_t* devStates) {

	//blockIdx.x represents the index of the particle inside the swarm
	//threadIdx.x represents the index of the dimension

	//gridDim.x represents the NUMBER_OF_PARTICLES
	//blockDim.x represents the actualParticleSize (PROBLEM_DIMENSIONS aligned up to a multiple of 16)

	uint32_t tid = threadIdx.x;
	uint32_t particleID = blockIdx.x;
	uint32_t posID = IMUL( particleID, blockDim.x) + tid;

	//Load the 2 pseudo random numbers needed to s_update position and velocity
	float2 R;
	curandState localState = devStates[posID];
	R.x = curand_uniform(&localState);
	R.y = curand_uniform(&localState);
	devStates[posID] = localState;

	float pos = c_minValue + R.x * c_deltaValue;
	g_positions[posID]     = pos;
	g_bestPositions[posID] = pos;

	//Non uniform Initialization as in standardPSO2006
	float vel = c_minValue + R.y * c_deltaValue;
	g_velocities[posID] = (vel - pos) / 2.0;
}




//********************************
//Kernel code for positions update
//********************************
///@brief cuda kernel to update the position and the velocity of all particles in parallel (at once)
/// Each thread initializes only one coordinate value of a particle's pose/velocity.
/// Hence the parallelization is not at the particles level but even at a lower level
///@param W Inertia weight (PSO algorithm)
///@param C1 Cognitive attraction factor (PSO algorithm)
///@param C2 Social attraction factor (PSO algorithm)
///@param g_positions pointer to the GPU's global-memory array containing the current position of all particles (from all swarms, in case of multi-swarm simulation)
///@param g_bestPositions pointer to the GPU's global-memory array containing the current personal best position of all particles
///@param g_velocities pointer to the GPU's global-memory array containing the current velocity of all particles
///@param g_rand pointer to the GPU's global-memory array containing the pseudo-random numbers to be consumed by the simulation
static __global__ void g_positionsUpdate(float W, float C1, float C2, float*  g_positions, float*  g_bestPositions, float*  g_velocities, curandState* devStates){

	//blockIdx.x represents the index of the particle inside the swarm
	//threadIdx.x represents the index of the dimension

	//gridDim.x represents the NUMBER_OF_PARTICLES
	//blockDim.x represents the actualParticleSize (PROBLEM_DIMENSIONS aligned up to a multiple of 16)

	uint32_t tid = threadIdx.x;
	//particleID is used to access local best indexes and and update vectors
	uint32_t particleID = blockIdx.x;
	//posID is used to access g_positions, g_bestPositions and g_velocities arrays
	uint32_t posID = IMUL( particleID, blockDim.x) + tid;

	__shared__ uint32_t s_update;
	__shared__ uint32_t s_bestID;

	//The first thread load the best position s_update flag and the index of the local best individual
	if(tid == 0){
		//Load the s_update flag
		s_update = tex1Dfetch(t_texUpdatePositionFlags, particleID);
		//Load the local best ID
		s_bestID = tex1Dfetch(t_texLocalBestIDs, particleID);
		//Calculate the beginning of the local best individual
		s_bestID = IMUL( s_bestID, blockDim.x);
	}
	__syncthreads();

	//Now all the threads in parallel...

	//Load current position
	float pos     = tex1Dfetch(t_texPositions, posID);
	//Load current best position
	float bestPos = tex1Dfetch(t_texBestPositions, posID);
	//Load current velocity
	float vel = tex1Dfetch(t_texVelocities, posID);
	//Load the 2 pseudo random numbers needed to s_update position and velocity
	float2 R;
	curandState localState = devStates[posID];
	R.x = curand_uniform(&localState);
	R.y = curand_uniform(&localState);
	devStates[posID] = localState;

	//Possibly s_update the best position
	//NOTE this is always a non-divergent branch!
	if (s_update){
		bestPos = pos;
		//Save back to global memory the s_updated best position
		g_bestPositions[posID] = bestPos;
	}

	//Apply inertia factor
	vel *= W;

	//Add to the velocity the Cognitive Contribution
	vel += C1 * R.x * (bestPos - pos);

	//Add to the velocity the Social Contribution
	vel += C2 * R.y * (tex1Dfetch(t_texBestPositions, s_bestID+tid) - pos);

	//Save back to global memory the new velocity
	g_velocities[posID] = vel;


	//Save back to global memory the s_updated and cropped new position

	//Position is kept inside the limits of the search space...
	pos += vel;

	//Clamps the new pose to the actual search space
	pos = min(pos, c_maxValue);
	pos = max(pos, c_minValue);

	//Save back to global memory the new position
	g_positions[posID] = pos;
}



//********************************************
//Kernel code for the first local bests update
//********************************************

///@brief cuda kernel to update the index of the local best particle and the "update best position" flag for all particles in parallel (at once)
///This kernel uses only one thread-block to load the current fitness value of all particles and determine all the local best indices. This is called only the first time to force the update of the personal best fitness values.
///@param g_bestFitnesses pointer to the GPU's global-memory array containing the current personal best fitness of all particles
///@param g_localBestIDs pointer to the GPU's global-memory array containing the indexes (for all particles) of the best neighbour (for the ring topology in this case)
static __global__ void g_firstBestsUpdate(float*  g_bestFitnesses, uint32_t* g_localBestIDs){

	//threadIdx.x represents the index of the particle inside the swarm
	//blockDim.x represents the NUMBER_OF_PARTICLES
	uint32_t particleID = threadIdx.x;

	extern __shared__ float s_fitnesses[];

	//Load the fitness value from global memory
	float fitness = tex1Dfetch(t_texFitnesses, particleID);
	g_bestFitnesses[particleID] = fitness;

	int bestID = particleID + 1;
	s_fitnesses[bestID] = fitness;

	//Toroidal ring topology...
	if (particleID == blockDim.x - 1)
		s_fitnesses[0 ] = fitness;
	if (particleID == 0)
		s_fitnesses[blockDim.x + 1] = fitness;
	//__syncthreads();

	//Find the local best (among the two neighbours of the ring)

	//Controls the left-neighbour
	if (s_fitnesses[particleID] BETTER_THAN s_fitnesses[bestID])
		bestID = particleID;

	//Controls the right-neighbour
	if (s_fitnesses[particleID+2] BETTER_THAN s_fitnesses[bestID])
		bestID = particleID+2;


	if (bestID == 0)
		bestID = blockDim.x;
	if (bestID == (blockDim.x + 1))
		bestID = 1;

	bestID--;

	//Writes the best-ID to global memory
	g_localBestIDs[particleID] = bestID;
}

//**************************************
//Kernel code for the local bests update
//**************************************

///@brief cuda kernel to update the index of the local best particle and the "update best position" flag for all particles in parallel (at once)
///This kernel uses only one thread-block to load the current fitness value of all particles and determine all the local best indices. This is called only the first time to force the update of the personal best fitness values.
///@param g_bestFitnesses pointer to the GPU's global-memory array containing the current personal best fitness of all particles
///@param g_localBestIDs pointer to the GPU's global-memory array containing the indexes (for all particles) of the best neighbour (for the ring topology in this case)
///@param g_update pointer to the GPU's global-memory array containing the "update" flags needed by the g_positionUpdate() kernel
static __global__ void g_bestsUpdate(float*  g_bestFitnesses, uint32_t* g_localBestIDs, uint32_t* g_update){

	//threadIdx.x represents the index of the particle inside the swarm
	//blockDim.x represents the NUMBER_OF_PARTICLES
	uint32_t particleID = threadIdx.x;

	extern __shared__ float s_fitnesses[];

	//Load the current fitness and best fitness value
	float newFitness  = tex1Dfetch(t_texFitnesses,     particleID);
	float bestFitness = tex1Dfetch(t_texBestFitnesses, particleID);

	//Possibly update of both the best fitness value and the best position update flag
	uint32_t update = newFitness BETTER_THAN bestFitness;
	g_update[particleID] = update;
	if (update){
		bestFitness = newFitness;
		g_bestFitnesses[particleID] = bestFitness;
	}
	//__syncthreads();

	int newID = particleID + 1;
	int bestID = newID;

	s_fitnesses[newID] = bestFitness;

	//Toroidal ring topology...
	if (particleID == 0)
		s_fitnesses[blockDim.x + 1] = bestFitness;
	if (particleID == blockDim.x - 1)
		s_fitnesses[0] = bestFitness;


	//Find the local best (among the two neighbours of the ring)

	//Controls the left-neighbour
	if (s_fitnesses[newID - 1] BETTER_THAN s_fitnesses[bestID])
		bestID = newID - 1;

	//Controls the right-neighbour
	if (s_fitnesses[newID + 1] BETTER_THAN s_fitnesses[bestID])
		bestID = newID + 1;

	if (particleID == 0 && bestID == 0)
		bestID = blockDim.x;
	if (particleID == (blockDim.x - 1) && bestID == (blockDim.x + 1))
		bestID = 1;

	bestID--;

	//Writes the best-ID to global memory
	g_localBestIDs[particleID] = bestID;
}


//**********************************************************************************************************
//Kernel code for the global best update (needed only at the end of the optimization with the ring topology)
//**********************************************************************************************************
///@brief cuda kernel to update the index of the global best particle
///This kernel uses only one thread-block to load the current best fitness value of all particles and determine the current global best particle/fitness. Each thread load one fitness value and then the global best value is determined by means of a parallel reduction. One second parallel reduction permits to also determine the index of the global best particle.
///@param g_globalBestFitness pointer to the GPU's global-memory variable to contain the global best fitness value
///@param g_globalBestID pointer to the GPU's global-memory variable to contain the index of the global best particle
///@param numberOfParticles the number of patricles in the swarm
template <uint32_t threadNum>
static __global__ void g_findGlobalBest(float *g_globalBestFitness, uint32_t* g_globalBestID, int numberOfParticles){

	//threadIdx.x represents the index of the particle inside the swarm
	//blockDim.x (equal to threadNum) represents the minimum between 2^(floor( log2 NUMBER_OF_PARTICLES) ) and the max #threads (512)

	__shared__ float s_fitnesses[threadNum];

	uint32_t tid = threadIdx.x;

	float bestFitness;
	#ifdef MAXIMIZE
		s_fitnesses[tid] = 0;		//Start with the lowest possible value
		bestFitness = 0;
	#else
		s_fitnesses[tid] = 1e20;	//Start with a very high value
		bestFitness = 1e20;
	#endif

	if (tid < numberOfParticles){
		bestFitness = tex1Dfetch(t_texBestFitnesses, tid);
		s_fitnesses[tid] = bestFitness;
	}

	//Reduction of the s_fitnesses vector to find the minimum value
	#ifdef MAXIMIZE
		reduceToMax<threadNum>(s_fitnesses, tid);
	#else
		reduceToMin<threadNum>(s_fitnesses, tid);
	#endif

	//The first thread updates the global best data
	if (tid == 0){
		*g_globalBestFitness = s_fitnesses[0];
	}
	__syncthreads();

	if (tid < numberOfParticles && bestFitness == s_fitnesses[0])
		bestFitness = tid;
	else
		bestFitness = blockDim.x;

	s_fitnesses[tid] = bestFitness;

	//Reduction of the s_fitnesses vector to find the global best ID
	//(the index of the best particle)
	reduceToMin<threadNum>(s_fitnesses, tid);

	if (tid == 0)
		*g_globalBestID = s_fitnesses[0];
}




//*****************************
//code to copy global best data
//*****************************
///@brief cuda kernel to copy the final pose of the best individual to a separate global memory location
///This kernel uses only one thread-block. Each thread copies the value of a single coordinate.
///@param g_globalBestPositions pointer to the GPU's global-memory array to contain the final best position
///@param g_globalBestID pointer to the GPU's global-memory variable containing the index of the final best individual
static __global__ void g_globalBestCopy(float* g_globalBestPositions, uint32_t* g_globalBestID){

	//threadIdx.x represents the index of the dimension of the problem
	//blockDim.x represents the actual particle size

	uint32_t tid = threadIdx.x;
	__shared__ uint32_t s_bestID;

	if (tid == 0){
		s_bestID = *g_globalBestID;
		s_bestID = IMUL( s_bestID, blockDim.x);
	}
	__syncthreads();

	g_globalBestPositions[tid] = tex1Dfetch(t_texBestPositions, s_bestID + tid);
}



#endif


