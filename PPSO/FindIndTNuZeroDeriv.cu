#include "FindIndTNuZeroDeriv.cuh"
#include <vector>
#include "devicePSO.cuh"
#include "math_constants.h"

using namespace std;

/* The digamma function is the derivative of gammaln.

   Reference:
    J Bernardo,
    Psi ( Digamma ) Function,
    Algorithm AS 103,
    Applied Statistics,
    Volume 25, Number 3, pages 315-317, 1976.

    From http://www.psc.edu/~burkardt/src/dirichlet/dirichlet.f
    (with modifications for negative numbers and extra precision)
*/
__device__ __host__ double digamma(double x)
{
  const double c = 12,
    digamma1 = -0.57721566490153286,
    trigamma1 = 1.6449340668482264365, /* pi^2/6 */
    s = 1e-6,
    s3 = 1./12,
    s4 = 1./120,
    s5 = 1./252,
    s6 = 1./240,
    s7 = 1./132,
    s8 = 691./32760,
    s9 = 1./12,
    s10 = 3617./8160;
  double result;
  
//  /* Illegal arguments */
//  if((x == neginf) || isnan(x)) {
//    return NAN;
//  }
//  /* Singularities */
//  if((x <= 0) && (floor(x) == x)) {
//    return neginf;
//  }

  /* Negative values */
  /* Use the reflection formula (Jeffrey 11.1.6):
   * digamma(-x) = digamma(x+1) + pi*cot(pi*x)
   *
   * This is related to the identity
   * digamma(-x) = digamma(x+1) - digamma(z) + digamma(1-z)
   * where z is the fractional part of x
   * For example:
   * digamma(-3.1) = 1/3.1 + 1/2.1 + 1/1.1 + 1/0.1 + digamma(1-0.1)
   *               = digamma(4.1) - digamma(0.1) + digamma(1-0.1)
   * Then we use
   * digamma(1-z) - digamma(z) = pi*cot(pi*z)
   */
//  if(x < 0) {
//    return digamma(1-x) + CUDART_PI/tan(-CUDART_PI * x);
//  }
  /* Use Taylor series if argument <= S */
  if(x <= s) return digamma1 - 1/x + trigamma1*x;
  /* Reduce to digamma(X + N) where (X + N) >= C */
  result = 0;
  while(x < c) {
    result -= 1/x;
    x++;
  }
  /* Use de Moivre's expansion if argument >= C */
  /* This expansion can be computed in Maple via asympt(Psi(x),x) */
  if(x >= c) {
    double r = 1/x, t;
    result += log(x) - 0.5*r;
    r *= r;
#if 1
    result -= r * (s3 - r * (s4 - r * (s5 - r * (s6 - r * s7))));
#else
    /* this version for lame compilers */
    t = (s5 - r * (s6 - r * s7));
    result -= r * (s3 - r * (s4 - r * t));
#endif
  }
  return result;
}

__device__ __host__ inline gpu_fp_t calcDlogXDmu(gpu_fp_t x, gpu_fp_t nu, gpu_fp_t mu, gpu_fp_t sigma) {
	//−(ν+1)/2 (1+1/ν ((x−μ)/σ)^2 )^(−1) (-2/νσ ((x−μ)/σ))
	gpu_fp_t centerX = (x - mu) / sigma;
	return (nu+1) / (2 * (1 + centerX * centerX / nu)) * ( 2 * centerX / (nu * sigma));
}

__device__ __host__ inline gpu_fp_t calcDlogXDsigma(gpu_fp_t x, gpu_fp_t nu, gpu_fp_t mu, gpu_fp_t sigma) {
	//−σ^(−1)−(ν+1)/2 (1+1/ν ((x−μ)/σ)^2 )^(−1) (2/ν ((x−μ)/σ)(−(x−μ)/σ^2 ))
	gpu_fp_t centerX = (x - mu) / sigma;
	return -1/sigma -(nu+1)/ (2 * (1 + centerX * centerX / nu)) * (-2 * centerX * centerX /(nu*sigma));
}

__device__ __host__ inline gpu_fp_t calcDlogXDnu(gpu_fp_t x, gpu_fp_t nu, gpu_fp_t mu, gpu_fp_t sigma) {
	//-((mu - x).*(- mu^2 + 2*mu*x + sigma^2 - x.^2))./(mu^2 - 2*mu*x + nu*sigma^2 + x.^2).^2
	//return -((mu - x)*(- mu*mu + 2*mu*x + sigma*sigma - x*x))/((mu*mu - 2*mu*x + nu*sigma*sigma + x*x) * (mu*mu - 2*mu*x + nu*sigma*sigma + x*x));
	// psi(nu/2 + 1/2)/2 - psi(nu/2)/2 - 1/(2*nu) - ((mu - x)^2/(nu*sigma^2) + 1)^(nu/2 + 1/2)*(log((mu - x)^2/(nu*sigma^2) + 1)/(2*((mu - x)^2/(nu*sigma^2) + 1)^(nu/2 + 1/2)) - ((nu/2 + 1/2)*(mu - x)^2)/(nu^2*sigma^2*((mu - x)^2/(nu*sigma^2) + 1)^(nu/2 + 3/2)))
	gpu_fp_t centerX = (x - mu) / sigma;
	//return (digamma(nu/2 + 1/2)/2 - digamma(nu/2)/2 - 1/(2*nu) - pow((centerX*centerX/(nu) + 1), (nu/2 + 1/2))*(log(centerX*centerX/(nu) + 1)/(pow(2*(centerX*centerX/(nu) + 1), (nu/2 + 1/2))) - ((nu/2 + 1/2)*(mu - x)*(mu - x))/(nu*nu*sigma*sigma*pow((centerX*centerX/(nu) + 1), (nu/2 + 3/2)))));
	return (digamma(nu/2 + 0.5)/2 - digamma(nu/2)/2 - 1/(2*nu) - pow(centerX*centerX/(nu) + 1, nu/2 + 0.5) * (log(centerX*centerX/(nu) + 1) / (2*pow(centerX*centerX/(nu) + 1, nu/2 + 0.5)) - ((nu/2 + 0.5)*centerX*centerX)/(nu*nu*pow(centerX*centerX/(nu) + 1, nu/2 + 1.5))));
}

__global__ void evaluateHowFarWereFromZeros(
	gpu_fp_t* fitness,
	gpu_fp_t* particleXs,
	gpu_fp_t* otherdxdmu, 
	gpu_fp_t* otherdxdsigma, 
	gpu_fp_t* otherdxdnu, 
	gpu_fp_t nu, gpu_fp_t mu, gpu_fp_t sigma, psodim dims) {
	// offset with task index
	int taskIdx = threadIdx.y;
	fitness = fitness + taskIdx * dims.nParticles;
	particleXs = particleXs + taskIdx * dims.nParticles * dims.nDimension;
	otherdxdmu = otherdxdmu + taskIdx;
	otherdxdsigma = otherdxdsigma + taskIdx;
	otherdxdnu = otherdxdnu + taskIdx;
	// offset done

	int particleIdx = blockIdx.x;
	int firstParticleXIdx = getIdx(particleIdx, 0, dims.nDimension),
		secondParticleXIdx = getIdx(particleIdx, 1, dims.nDimension),
		thirdParticleXIdx = getIdx(particleIdx, 2, dims.nDimension);

	fitness[particleIdx] = 
		pow(calcDlogXDmu(particleXs[firstParticleXIdx], nu, mu, sigma)
			+ calcDlogXDmu(particleXs[secondParticleXIdx], nu, mu, sigma)
			+ calcDlogXDmu(particleXs[thirdParticleXIdx], nu, mu, sigma)
			+ otherdxdmu[0], 2)
		+ pow(calcDlogXDsigma(particleXs[firstParticleXIdx], nu, mu, sigma)
			+ calcDlogXDsigma(particleXs[secondParticleXIdx], nu, mu, sigma)
			+ calcDlogXDsigma(particleXs[thirdParticleXIdx], nu, mu, sigma)
			+ otherdxdsigma[0], 2)
		+ pow(calcDlogXDnu(particleXs[firstParticleXIdx], nu, mu, sigma)
			+ calcDlogXDnu(particleXs[secondParticleXIdx], nu, mu, sigma)
			+ calcDlogXDnu(particleXs[thirdParticleXIdx], nu, mu, sigma)
			+ otherdxdnu[0], 2);
}


FindIndTNuZeroDeriv::FindIndTNuZeroDeriv(vector<gpu_fp_t*> samples, int _sampleSize, gpu_fp_t _nu, gpu_fp_t _mu, gpu_fp_t _sigma, int nParticles, int nIter) 
	: PSO(nParticles, 3, nIter, samples.size()), 
	nu(_nu), mu(_mu), sigma(_sigma), sampleSize(_sampleSize),
	otherdxdmu(dims.nTasks), otherdxdsigma(dims.nTasks), otherdxdnu(dims.nTasks) {

//	cout << _sampleSize << ", " << _nu << endl;

	gpu_fp_t* otherdxdmuHost = new gpu_fp_t[samples.size()];
	gpu_fp_t* otherdxdsigmaHost = new gpu_fp_t[samples.size()];
	gpu_fp_t* otherdxdnuHost = new gpu_fp_t[samples.size()];

	for(int iTask = 0; iTask < samples.size(); iTask++) {
		otherdxdmuHost[iTask] = 0;
		otherdxdsigmaHost[iTask] = 0;
		otherdxdnuHost[iTask] = 0;
		// calculate sample's derivatives TODO
		for (int j = 0; j < sampleSize; j++) {
//			cout << calcDlogXDmu(samples[iTask][j], nu, mu, sigma) << endl;
			otherdxdmuHost[iTask]    += calcDlogXDmu(samples[iTask][j], nu, mu, sigma);
			otherdxdsigmaHost[iTask] += calcDlogXDsigma(samples[iTask][j], nu, mu, sigma);
			otherdxdnuHost[iTask]    += calcDlogXDnu(samples[iTask][j], nu, mu, sigma);
		}
	}
//	FILE* file = fopen("d:/log.txt", "w");
//	fprintf(file, "ff is %g, %g, %d, %d, %d, %d\n", otherdxdmuHost[0] , otherdxdsigmaHost[0], _sampleSize, nParticles, nIter);
//	fclose(file);
	cudaMemcpyAsync(thrust::raw_pointer_cast(otherdxdmu.data()), otherdxdmuHost, sizeof(gpu_fp_t) * dims.nTasks, cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(thrust::raw_pointer_cast(otherdxdsigma.data()), otherdxdsigmaHost, sizeof(gpu_fp_t) * dims.nTasks, cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(thrust::raw_pointer_cast(otherdxdnu.data()), otherdxdnuHost, sizeof(gpu_fp_t) * dims.nTasks, cudaMemcpyHostToDevice, stream);
//	cudaMemcpy(thrust::raw_pointer_cast(otherdxdmu.data()), otherdxdmuHost, sizeof(gpu_fp_t) * dims.nTasks, cudaMemcpyHostToDevice);
//	cudaMemcpy(thrust::raw_pointer_cast(otherdxdsigma.data()), otherdxdsigmaHost, sizeof(gpu_fp_t) * dims.nTasks, cudaMemcpyHostToDevice);

	// init boundaries, for x in [-1e50, 1e50]
	double _scale[] = {100, 100, 100}, _x0[] = {0, 0, 0};
	double xl[] = {-1e50, -1e50, -1e50}, xu[] = {1e50, 1e50, 1e50};
	init(xl, xu, _scale, _x0);

	// TODO: make it really asyncronious...
	cudaStreamSynchronize(stream);
	delete[] otherdxdmuHost;
	delete[] otherdxdsigmaHost;
	delete[] otherdxdnuHost;
}


void FindIndTNuZeroDeriv::evaluateCurrent() {
	evaluateHowFarWereFromZeros<<<dims.nParticles, dim3(1, dims.nTasks), 0, stream>>>(
								   thrust::raw_pointer_cast(particleEvalValues.data()), 
								   thrust::raw_pointer_cast(particleXs.data()), 
								   thrust::raw_pointer_cast(otherdxdmu.data()), 
								   thrust::raw_pointer_cast(otherdxdsigma.data()), 
								   thrust::raw_pointer_cast(otherdxdnu.data()), 
								   nu, mu, sigma, dims);
}




FindIndTNuZeroDeriv::~FindIndTNuZeroDeriv(void)
{
}

gpu_fp_t FindIndTNuZeroDeriv::getResult(int taskIdx, int coordIdx) {
	return globalBestXs[dims.nDimension*taskIdx + coordIdx];
}
