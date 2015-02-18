#ifndef _CUDA_PSO_H__
#define _CUDA_PSO_H__

#include <string>


/**
	@brief Main CUDA-PSO class.

	This class represents a swarm of particles which can be asked to optimize a fitness function completely on the GPU.
*/

class cudaPSO{
	public:
		///Initializes the class object loading parameters from file
		///@param parametersFile path of the file containing all the parameters to run the optimization
		cudaPSO(std::string parametersFile);
		~cudaPSO();

		///Sets the number of particles belonging to this swarm ( to be possibly called before init() )
		///@param pNumber number of particles
		void setNumberOfParticles(int pNumber);
		
		///Sets the dimension of the search space ( to be possibly called before init() )
		///@param pDim problem dimension
		void setProblemDimension(int pDim);
		
		///Sets the number of evolutionary generations to be performed by this swarm ( to be possibly called before init() )
		///@param gNum number of generations
		void setNumberOfGenerations(int gNum);
		
		///Sets the ID of the fitness function to be optimized by this swarm ( to be possibly called before init() )
		///@param fID function ID:\n
		///             0 - Sphere \n
		///		1 - Rastrigin \n
		///		2 - Rosenbrok
		void setFunctionID(int fID);

		///Finalizes the initialization of the class and prepares the GPU to run the swarm
		void init();

		///Actually starts the optimization process
		void optimize();

		///Returns the final fitness value obtained at the end of the optimization
		float getFinalFitnessValue();

		///Returns a poiter to the array containing the final global best position obtained at the end of the optimization
		float* getFinalGlobalBestPosition();

		///Prints final results to console output
		void printResults();


	private:

		///Number of particles belonging to this swarm
		unsigned int particlesNumber;

		///Number of parameters to be optimized
		unsigned int problemDimension;

		///Number of generations/iterations in one optimization step
		unsigned int numberOfGenerations;

		//PSO parameters
		///Inertia weight (PSO algorithm)
		float W;
		///Cognitive attraction factor (PSO algorithm)
		float C1;
		///Social attraction factor (PSO algorithm)
		float C2;

		
		///Final global best fitness value
		float h_globalBestFitness;

		///Pointer to an host array containing the final result (final global best position)
		float *h_globalBestPosition;

		
		///Lower limit for each dimension of the search space
		float minValue;
		///Upper limit for each dimension of the search space
		float maxValue;
		///Width of each dimension of the search space
		float deltaValue;

		
		///Index of the function to be optimized
		int functionID;

		///Flag to check wether the class has already been properly initialized calling the init() method
		bool initialized;
};



#endif



