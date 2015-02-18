#ifndef _CUDA_PSO_CPP__
#define _CUDA_PSO_CPP__


#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>

//#include <cutil_math.h>

#include "cudaPSO.h"
#include "cudaPSO.cuh"
#include "parametersParser.h"
#include "utilities.h"

//NOTE To print debug messages uncomment the following line
//#define PRINT_DEBUG



cudaPSO::cudaPSO(std::string parametersFile){
	int intValue;
	double doubleValue;
	std::string parameterName;
	parametersParser pp;

	#ifdef PRINT_DEBUG
		std::cout << "Initializing cudaPOS..." << std::endl;
	#endif


	//***********************************
	//Load parameters from file...
	//***********************************

	//parameterName = "OUTPUT_DIR";
	//if(pp.parse(parametersFile.c_str(), parameterName.c_str(), fileName))
	//	this->outputDirectory = fileName;
	//else
	//	exit(0);
	//#ifdef PRINT_DEBUG
	//	std::cout << "\t- outputDirectory = " << this->outputDirectory << std::endl;
	//#endif

	parameterName = "NUM_INDIVIDUALS";
	if(pp.parse(parametersFile.c_str(), parameterName.c_str(), &intValue))
		this->particlesNumber = intValue;
	else
		exit(0);
	#ifdef PRINT_DEBUG
		std::cout << "\t- particlesNumber = " << this->particlesNumber << std::endl;
	#endif


	parameterName = "PROBLEM_DIMENSION";
	if(pp.parse(parametersFile.c_str(), parameterName.c_str(), &intValue))
		this->problemDimension = intValue;
	else
		exit(0);
	#ifdef PRINT_DEBUG
		std::cout << "\t- problemDimension = "   << this->problemDimension << std::endl;
	#endif


	parameterName = "ITERATIONS_NUM";
	if(pp.parse(parametersFile.c_str(), parameterName.c_str(), &intValue))
		this->numberOfGenerations = intValue;
	else
		exit(0);
	#ifdef PRINT_DEBUG
		std::cout << "\t- numberOfGenerations = "   << this->numberOfGenerations << std::endl;
	#endif


	parameterName = "INERTIA_WEIGHT";
	if(pp.parse(parametersFile.c_str(), parameterName.c_str(), &doubleValue))
		this->W = doubleValue;
	else
		exit(0);
	#ifdef PRINT_DEBUG
		std::cout << "\t- Inertia Weight = "   << this->W << std::endl;
	#endif


	parameterName = "C1";
	if(pp.parse(parametersFile.c_str(), parameterName.c_str(), &doubleValue))
		this->C1 = doubleValue;
	else
		exit(0);
	#ifdef PRINT_DEBUG
		std::cout << "\t- C1 = "   << this->C1 << std::endl;
	#endif


	parameterName = "C2";
	if(pp.parse(parametersFile.c_str(), parameterName.c_str(), &doubleValue))
		this->C2 = doubleValue;
	else
		exit(0);
	#ifdef PRINT_DEBUG
		std::cout << "\t- C2 = "   << this->C2 << std::endl;
	#endif

	parameterName = "MIN_X";
	if(pp.parse(parametersFile.c_str(), parameterName.c_str(), &doubleValue))
		this->minValue = doubleValue;
	else
		exit(0);
	#ifdef PRINT_DEBUG
		std::cout << "\t- MIN_X = "   << this->minValue << std::endl;
	#endif

	parameterName = "MAX_X";
	if(pp.parse(parametersFile.c_str(), parameterName.c_str(), &doubleValue))
		this->maxValue = doubleValue;
	else
		exit(0);
	#ifdef PRINT_DEBUG
		std::cout << "\t- MAX_X = "   << this->maxValue << std::endl;
	#endif

	this->deltaValue = this->maxValue - this->minValue;


	this->h_globalBestPosition = new float[this->problemDimension];

	
	parameterName = "FUNCTION_ID";
	if(pp.parse(parametersFile.c_str(), parameterName.c_str(), &intValue))
		this->functionID = intValue;
	else
		this->functionID = 0;
	#ifdef PRINT_DEBUG
		std::cout << "\t- functionID = "   << this->functionID << std::endl;
	#endif
	

	#ifdef PRINT_DEBUG
		std::cout << "Done!" << std::endl;
	#endif

	this->initialized = false;
}


cudaPSO::~cudaPSO(){
	h_cudaPSO_Free();
	delete [] this->h_globalBestPosition;
}

void cudaPSO::setNumberOfParticles(int pNumber){
	this->particlesNumber = pNumber;
}

void cudaPSO::setProblemDimension(int pDim){
	this->problemDimension = pDim;

	delete [] this->h_globalBestPosition;
	this->h_globalBestPosition = new float[this->problemDimension];
}

void cudaPSO::setNumberOfGenerations(int gNum){
	this->numberOfGenerations = gNum;
}

void cudaPSO::setFunctionID(int fID){
	this->functionID = fID;
}


void cudaPSO::init(){
	if (!this->initialized){
		h_cudaPSO_Init(this->particlesNumber, this->problemDimension, this->numberOfGenerations);
		this->initialized = true;
	}else
		std::cout << "cudaPSO class already initialized!" << std::endl;
}


void cudaPSO::optimize(){
	if (this->initialized)
		h_cudaPSO_Optimize(
			this->functionID,
			this->numberOfGenerations,
			this->particlesNumber,
			this->problemDimension,
			this->W,
			this->C1,
			this->C2,
			this->minValue,
			this->maxValue,
			this->deltaValue,
			&this->h_globalBestFitness,
			this->h_globalBestPosition
		);
}


float cudaPSO::getFinalFitnessValue(){
	return this->h_globalBestFitness;
}

float* cudaPSO::getFinalGlobalBestPosition(){
	return this->h_globalBestPosition;
}

void cudaPSO::printResults(){
	std::cout << "Function ID = " << this->functionID << std::endl
		<< "Problem Dimension = " << this->problemDimension << std::endl
		<< "Number of Particles = " << this->particlesNumber << std::endl
		<< "Number of Generations = " << this-> numberOfGenerations << std::endl
		<< std::scientific << "Final Fitness Value = " << this->h_globalBestFitness << std::endl
		<< "Final Global Best Position = ";

	for (unsigned int i = 0; i < this->problemDimension; ++i)
		std::cout << h_globalBestPosition[i] << " ";
	std::cout << std::endl;
}



#endif


