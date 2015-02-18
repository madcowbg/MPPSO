#include "ParallelPSO.h"
#include "PSO.cuh"
#include <boost\thread.hpp>

void runOptim(PSO* pso) {
	pso->optimize();
}

ParallelPSO::ParallelPSO(void)
{
}


ParallelPSO::~ParallelPSO(void)
{
	for(unsigned int i = 0; i < tasks.size(); i++) {
		delete tasks[i];
	}
}

void ParallelPSO::add(PSO* task) {
	tasks.push_back(task);
}

PSO* ParallelPSO::get(unsigned int idx) {
	return tasks[idx];
}

unsigned ParallelPSO::size() {
	return tasks.size();
}

void ParallelPSO::optimize() {
	boost::thread* threads = new boost::thread[size()];
	for(unsigned int i = 0; i< size(); i++) {
		threads[i] = boost::thread(runOptim, tasks[i]);
	}

	for (unsigned int i = 0; i < size(); i++) {
		threads[i].join();
	}

	delete[] threads;
}