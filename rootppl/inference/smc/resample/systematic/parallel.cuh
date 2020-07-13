#ifndef RESAMPLE_GPU_INCLUDED
#define RESAMPLE_GPU_INCLUDED

/*
 * File parallel.cuh contains the parallel implementation of the systematic resampling. 
 */

#ifdef GPU

#include "common.cuh"
#include "kernels.cuh"
#include "utils/cuda_error_utils.cuh"

#include <curand_kernel.h>
// #include <thrust/reduce.h>
#include <thrust/scan.h>
// #include <thrust/execution_policy.h>
// #include <thrust/extrema.h>


/**
 * Calculates the log weight prefix sums safely according to the identity in the appendix of 
 * the paper "Parallel resampling in the particle filter", L. M. Murray et. al. 
 * 
 * @param w the weight array.
 * @param resampler the resampler struct.
 * @param numParticles the number of particles used in SMC.
 * @param numBlocks kernel launch setting.
 * @param numThreadsPerBlock kernel launch setting.
 * @return the logarithm of the total weight sum. 
 */
template <typename T>
HOST DEV floating_t calcLogWeightSumPar(floating_t* w, resampler_t<T> resampler, int numParticles, int numBlocks, int numThreadsPerBlock) {

    floating_t maxLogWeight = *(thrust::max_element(thrust::device, w, w + numParticles));
    // floating_t maxLogWeight = maxNaive(w, numParticles);
    
    expWeightsKernel<<<numBlocks, numThreadsPerBlock>>>(w, numParticles, maxLogWeight);

    thrust::inclusive_scan(thrust::device, w, w + numParticles, resampler.prefixSum); // prefix sum

    renormaliseSumsKernel<<<numBlocks, numThreadsPerBlock>>>(resampler.prefixSum, numParticles, maxLogWeight);
    
    cudaDeviceSynchronize();
    return resampler.prefixSum[numParticles - 1];
}

/**
 * Calculates the cumulative offsprings and then uses it to calculate the ancestor indices. 
 * 
 * @param resampler the resampler struct reference.
 * @param u a sample from the standard uniform distribution.
 * @param numParticles the number of particles used in SMC.
 * @param numBlocks kernel launch setting.
 * @param numThreadsPerBlock kernel launch setting.
 */
template <typename T>
HOST DEV void decideAncestors(resampler_t<T>& resampler, floating_t u, int numParticles, int numBlocks, int numThreadsPerBlock) {

    systematicCumulativeOffspringKernel<<<numBlocks, numThreadsPerBlock>>>(resampler.prefixSum, resampler.cumulativeOffspring, u, numParticles);

    cumulativeOffspringToAncestorKernel<<<numBlocks, numThreadsPerBlock>>>(resampler.cumulativeOffspring, resampler.ancestor, numParticles);

}

/**
 * Decides ancestors, propagates particles, and swaps pointers. (To avoid an additional copy kernel.)
 * 
 * @param particles the particles struct reference.
 * @param resampler the resampler struct reference.
 * @param u a sample from the standard uniform distribution.
 * @param numParticles the number of particles used in SMC.
 * @param numBlocks kernel launch setting.
 * @param numThreadsPerBlock kernel launch setting.
 */
template <typename T>
HOST DEV void postUniform(particles_t<T>& particles, resampler_t<T>& resampler, floating_t u, int numParticles, int numBlocks, int numThreadsPerBlock) {

    decideAncestors<T>(resampler, u, numParticles, numBlocks, numThreadsPerBlock);

    // Copy states
    copyStatesKernel<T><<<numBlocks, numThreadsPerBlock>>>(resampler.auxParticles, particles, resampler.ancestor, numParticles);
    cudaDeviceSynchronize();

    // Swap pointers
    particles_t<T> tempAux = resampler.auxParticles;
    resampler.auxParticles = particles;
    particles = tempAux;
}

#ifdef GPU
/**
 * Performs parallel resampling. Uses CUDA Dynamic Parallelism (launches kernels within kernels). Used in nested inference.
 * 
 * @param randState the curandState used to draw a random uniform sample on the GPU. 
 * @param particles the particles struct reference.
 * @param resampler the resampler struct reference.
 * @param numParticles the number of particles used in SMC.
 * @param numBlocks kernel launch setting.
 */
template <typename T>
DEV void resampleSystematicParNested(curandState* randState, particles_t<T>& particles, resampler_t<T>& resampler, int numParticles, int numBlocks) {
    
    floating_t u = uniform(randState, 0.0f, 1.0f);
    
    postUniform<T>(particles, resampler, u, numParticles, numBlocks, NUM_THREADS_PER_BLOCK_NESTED);
}
#endif

/**
 * Performs parallel resampling. Uses CUDA Dynamic Parallelism (launches kernels within kernels). Used in top-level inference. 
 * 
 * @param particles the particles struct reference.
 * @param resampler the resampler struct reference.
 * @param numParticles the number of particles used in SMC.
 * @param numBlocks kernel launch setting.
 */
template <typename T>
void resampleSystematicPar(particles_t<T>& particles, resampler_t<T>& resampler, int numParticles, int numBlocks) {

    floating_t u = uniformCPU(generatorRes);

    postUniform<T>(particles, resampler, u, numParticles, numBlocks, NUM_THREADS_PER_BLOCK);
}

#endif

#endif