
/*
 * File dists.cuh contains distributions that builds upon the hardware specific distributions. 
 * These implementations will be the same regardless of the hardware. Most of these distributions
 * are inspired by the WebPPL language (i.e. names and format).
 * 
 * The distribution functions uses a macro for the first parameter which specifies that a 
 * curandState pointer should be passed to the function if it is compiled for GPU. 

 * Sampling from these distributions (calling these functions), should be done
 * by using the macro SAMPLE(distName, args...). This will pass a curandState as argument.
 * This curandState will be present in the BBLOCK and BBLOCK_HELPER functions. 
 */


#include <random>
#include <time.h>
#include <cassert>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __NVCC__
#include <curand_kernel.h>
#endif

#include "inference/smc/smc.cuh"
#include "utils/misc.cuh"
#include "utils/math.cuh"
#include "scores.cuh"
#include "macros/macros.cuh"

#include "dists.cuh"

// Define the CPU generator and primitives, and initializes the generator state. 
#ifdef _OPENMP
// Unable to simply create an array of default_random_engine, so wrapped in a struct
struct generator_wrapper {
    std::default_random_engine gen;
};
generator_wrapper* genWrappers;
#else
std::default_random_engine gen;
#endif

std::uniform_real_distribution<floating_t> uniformDist(0.0, 1.0);
std::normal_distribution<floating_t> normalDist(0, 1);
void initGen() {
    #ifdef _OPENMP
    int maxThreads = omp_get_max_threads();
    genWrappers = new generator_wrapper[maxThreads];
    for(int i = 0; i < maxThreads; i++) {
        genWrappers[i].gen.seed(time(NULL) + i);
    }
    #else
    gen.seed(time(NULL));
    #endif
}

void freeGen() {
    #ifdef _OPENMP
    delete[] genWrappers;
    #endif
}

#ifdef __NVCC__
#include "dists_gpu.cuh"
#else
#include "dists_cpu.cuh"

#endif


DEV int bernoulli(RAND_STATE_DECLARE floating_t p) {
    // The probability needs to be between 0 and 1.
    assert(0.0 <= p <= 1.0);
    return SAMPLE(uniform, 0, 1) < p ? 1 : 0;
}

DEV floating_t exponential(RAND_STATE_DECLARE floating_t lambda) {
    
    return -log(1 - SAMPLE(uniform, 0, 1)) / lambda;
}

DEV floating_t gamma(RAND_STATE_DECLARE floating_t k, floating_t theta) {
    // This gamma sampling is based on the implementation used by GSL
    if (k < 1) {
        double u = SAMPLE(uniform, 0, 1);
        return SAMPLE(gamma, 1.0 + k, theta) * pow(u, 1.0 / k);
    }
    
    {
        double x, v, u;
        double d = k - 1.0 / 3.0;
        double c = (1.0 / 3.0) / sqrt(d);
    
        while (true) {
            do {
                x = SAMPLE(normal, 0, 1);
                v = 1.0 + c * x;
            } while (v <= 0);
    
            v = v * v * v;
            u = SAMPLE(uniform, 0, 1);
    
            if (u < 1 - 0.0331 * x * x * x * x) 
                break;
    
            if (log(u) < 0.5 * x * x + d * (1 - v + log(v)))
                break;
        }
    
        return theta * d * v;
    }
    
}

DEV void dirichlet(RAND_STATE_DECLARE const floating_t* alpha, const int n, floating_t* ret) {
    
    floating_t sum = 0;
    for (int k = 0; k < n; k++) {
        ret[k] = SAMPLE(gamma, alpha[k], 1);
        sum += ret[k];
    }

    for (int k = 0; k < n; k++) 
        ret[k] /= sum;
}

DEV int discrete(RAND_STATE_DECLARE const floating_t* ps, const int n) {
    // Could be optimized, if many samples should be done
    floating_t u = SAMPLE(uniform, 0, 1);
    floating_t sum = 0;
    int idx = 0;
    for(idx = 0; idx < n-1; idx++) {
        sum += ps[idx];
        if(u <= sum)
            break;
    }
    return idx;
}

DEV int randomInteger(RAND_STATE_DECLARE const int n) {
    floating_t u = SAMPLE(uniform, 0, n) - 0.000000000000001;
    return static_cast<int>(u);
}

DEV floating_t beta(RAND_STATE_DECLARE floating_t alpha, floating_t beta) {
    floating_t x = SAMPLE(gamma, alpha, 1);
    floating_t y = SAMPLE(gamma, beta, 1);
    return x / (x + y);
}

DEV int binomial(RAND_STATE_DECLARE floating_t p, int n) {
    // Could be done more efficiently, see alias methods or webppl source code
    int numSuccesses = 0;
    for(int t = 0; t < n; t++)
        numSuccesses += SAMPLE(bernoulli, p);

    return numSuccesses;
}

DEV floating_t cauchy(RAND_STATE_DECLARE floating_t loc, floating_t scale) {
    return loc + scale * tan(PI * (SAMPLE(uniform, 0, 1) - 0.5));
}

DEV floating_t laplace(RAND_STATE_DECLARE floating_t loc, floating_t scale) {
    floating_t u = SAMPLE(uniform, -0.5, 0.5);
    return loc - scale * sgn<floating_t>(u) * log(1 - 2 * abs(u));
}

DEV void multinomial(RAND_STATE_DECLARE floating_t* ps, int n, int numCategories, int* returnArr) {

    // Can be much more efficient for large n, using host API on GPU (alias methods)
    // Just sorting probabilities in descending order should give performance increase for many samples
    
    for(int k = 0; k < numCategories; k++)
        returnArr[k] = 0;

    for(int t = 0; t < n; t++) {
        int idx = SAMPLE(discrete, ps, numCategories);
        returnArr[idx]++;
    }
}

DEV void diagCovNormal(RAND_STATE_DECLARE floating_t* mu, floating_t* sigma, int n, floating_t* ret) {

    // Could be optimized with curand2 or curand4 that samples more than one value at a time.
    for(int k = 0; k < n; k++) {
        ret[k] = SAMPLE(normal, mu[k], sigma[k]);
    }
}

DEV void multivariateStandardNormal(RAND_STATE_DECLARE int n, floating_t* ret) {
    for(int k = 0; k < n; k++) {
        ret[k] = SAMPLE(normal, 0, 1);
    }
}

DEV floating_t lomax(RAND_STATE_DECLARE floating_t lambda, floating_t alpha) {
    floating_t u = SAMPLE(uniform, 0, 1);
    return lambda*(pow(u, -1.0/alpha) - 1.0);
}

DEV int negativeBinomial(RAND_STATE_DECLARE floating_t p, int n) {
    return n - SAMPLE(binomial, p, n);
}

/* Mathematically the Chi Square 
   distribution with n degrees of freedom is equivalent to a Gamma        
   distribution with shape parameter n/2 and scale parameter 2. */
DEV floating_t chi_squared(RAND_STATE_DECLARE floating_t k) {
  floating_t result = SAMPLE(gamma, k/2.0, 2.0);
  return result;
}

DEV floating_t student_t(RAND_STATE_DECLARE floating_t k, floating_t mu, floating_t v) {
  assert(0.0 < k);
  assert(0.0 < v);
  floating_t y = SAMPLE(normal, 0.0, sqrt(v/k));
  floating_t z = SAMPLE(chi_squared, k);
  return mu + y/sqrt(z/k);
}
