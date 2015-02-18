#pragma once
#include <vector>
#include "PSO.cuh"

class ParallelPSO
{
	std::vector<PSO*> tasks;

public:
	void add(PSO* task);

	void optimize();

	PSO* get(unsigned int idx);
	unsigned int size();

	ParallelPSO(void);
	~ParallelPSO(void);
};

