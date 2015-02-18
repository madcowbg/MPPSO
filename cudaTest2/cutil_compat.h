#ifndef CUTIL_COMPAT_H
#define CUTIL_COMPAT_H
//#include <helper_cuda.h>
#include <cuda_runtime.h>

int cutGetMaxGflopsDeviceId();
void cudasafe( cudaError_t error, char* message);
void cutilCheckMsg( const char *errorMessage);

//static __global__ void g_findGlobalBest(float *g_globalBestFitness, uint32_t* g_globalBestID, int numberOfParticles);
#endif