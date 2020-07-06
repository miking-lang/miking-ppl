#ifndef RESAMPLE_COMMON_INCLUDED
#define RESAMPLE_COMMON_INCLUDED

/*
 * File common.cuh contains definitions used by both sequential and parallel systematic resampling. 
 */

#include <stdlib.h>
#include <random>
#include <time.h>
#include "inference/smc/smc.cuh"
#include "inference/smc/particles_memory_handler.cuh"
#include "utils/misc.cuh"

// Generator for CPU RNG
default_random_engine generatorRes;
uniform_real_distribution<floating_t> uniformCPU(0.0, 1.0);

/*
 * Resampler structure for Systematic resampling. Contains pointers to structures necessary for the resampling. 
 *
 * ancestor: Array used to store indices of ancestors in resampling.
 * cumulativeOffspring: Array used to store the cumulative number of offspring for each particle. 
 * prefixSum: Array used to store the inclusive prefix sum. 
 * auxParticles: Particles structure used to copy particles in resample propagation. Required as the propagation is not in-place. 
 */
template <typename T>
struct resampler_t {

    int* ancestor; 
    int* cumulativeOffspring;
    floating_t* prefixSum;
    particles_t<T> auxParticles;
};

/**
 * Allocates resampler and its arrays and set the seed for the CPU RNG.
 * This should be used for top-level inference.  
 *
 * @param numParticles the number of particles used in SMC.
 * @return the allocated resampler struct. 
 */
 template <typename T>
resampler_t<T> initResampler(int numParticles) {

    generatorRes.seed(time(NULL) * 3); // Multiply by 3 to avoid same seed as distributions. 
    resampler_t<T> resampler;

    allocateMemory<int>(&resampler.ancestor, numParticles);
    allocateMemory<int>(&resampler.cumulativeOffspring, numParticles);
    allocateMemory<floating_t>(&resampler.prefixSum, numParticles);
    
    resampler.auxParticles = allocateParticles<T>(numParticles);

    return resampler;
}

/**
 * Allocates resampler and its arrays. 
 * This should be used for nseted inference. 
 *
 * @param numParticles the number of particles used in nested SMC.
 * @return the allocated resampler struct. 
 */
template <typename T>
HOST DEV resampler_t<T> initResamplerNested(int numParticles) {

    resampler_t<T> resampler;

    resampler.ancestor = new int[numParticles];
    resampler.cumulativeOffspring = new int[numParticles];
    resampler.prefixSum = new floating_t[numParticles];
    resampler.auxParticles = allocateParticlesNested<T>(numParticles);

    return resampler;
}

/**
 * Frees the allocated arrays used by the resampler.
 * This should be used for top-level inference.  
 *
 * @param resampler the resampler which should be freed.
 */
template <typename T>
void destResampler(resampler_t<T> resampler) {

    freeMemory<int>(resampler.ancestor);
    freeMemory<int>(resampler.cumulativeOffspring);
    freeMemory<floating_t>(resampler.prefixSum);
    freeParticles<T>(resampler.auxParticles);
}

/**
 * Frees the allocated arrays used by the resampler.
 * This should be used for nested inference.  
 *
 * @param resampler the resampler which should be freed.
 */
template <typename T>
HOST DEV void destResamplerNested(resampler_t<T> resampler) {
    delete[] resampler.ancestor;
    delete[] resampler.cumulativeOffspring;
    delete[] resampler.prefixSum;
    freeParticlesNested<T>(resampler.auxParticles);
}

/**
 * Copies data from one particle in the source array to a particle in the destination array and resets the weight. 
 * Used in resampling. Does NOT handle references that may exist in the progStates.
 * Such references could be handled by overloading the progState struct's assignment operator. 
 *
 * @param particlesDst the destination array to copy to.
 * @param particlesSrc the source array to copy from.
 * @param dstIdx the index in particlesDst to write to.
 * @param srcIdx the index in particlesSrc to read from. 
 */
template <typename T>
DEV void copyParticle(particles_t<T> particlesDst, const particles_t<T> particlesSrc, int dstIdx, int srcIdx) {
    particlesDst.progStates[dstIdx] = particlesSrc.progStates[srcIdx];
    particlesDst.pcs[dstIdx] = particlesSrc.pcs[srcIdx];
    particlesDst.weights[dstIdx] = 0;
}

#endif