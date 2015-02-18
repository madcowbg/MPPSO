#include <iostream>
#include <string>
#include "PSO.cuh"
#include "IndStudentTPSO.cuh"
#include "ParallelPSO.h"

#include <ctime>

//int main(void)
//{
// time_t tstart, tend; 
// tstart = time(0);
//
//	//int ndim = 2;
//	//double* xl = new double[ndim];
//	//double* xu = new double[ndim];
//	//for (int i = 0; i < ndim; i++) {
//	//	xl[i] = -10;
//	//	xu[i] = 10;
//	//}
//
//	//PSO pso(1<<10, ndim, 300);
//	//pso.init(xl, xu);
//	//pso.optimize();
//
//	//pso.dump();
//	//
//	//delete xl;
//	//delete xu;
//
//	std::cout << "Starting Student-T fit!";
//	gpu_fp_t sample[] = {1.0347, 0.7269, -0.3034, 0.2939, -0.7873};
//
///*	IndStudentTPSO fitter(sample, 5);
//	fitter.optimize();
//	fitter.dump();
//	std::cout << " mu = " << fitter.getMu() << std::endl << " sigma = "<< fitter.getSigma() << " logl = " << fitter.getFit() << std::endl;
//*/
//	std::cout << "Starting Student-T fits!";
//	//gpu_fp_t sample[] = {1.0347, 0.7269, -0.3034, 0.2939, -0.7873};
//
//	ParallelPSO ppso;
//	for (int i = 0; i < 16; i++) {
//		ppso.add(new IndStudentTPSO(sample, 5));
//	}
//	ppso.optimize();
//	
//
//	std::cout << "Ready!\n";
//	tend = time(0); 
//	std::cout << "It took "<< difftime(tend, tstart) <<" second(s)."<< std::endl;
//
//	std::string str;
//	std::cin >> str;
//    return 0;
//}
