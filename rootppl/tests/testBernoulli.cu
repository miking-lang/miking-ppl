/**
 * Tests the Beta distribution
 */

#include <iostream>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>

#include "inference/smc/smc.cuh"
#include "utils/math.cuh"
#include "dists/delayed.cuh"

const floating_t alpha_p = 3.0; // first shape
const floating_t beta_p = 3.0;  // second shape
const int K = 5  ; // number of trials
const int M = 100;// subsample size in callback

const std::string testName = "testBernoulli";

int numParts; // number of particles, supplied by first argument
int numRuns; // number of runs supplied by the command line


INIT_MODEL(floating_t, 1);

BBLOCK(testBernoulli, {
    
    int statistic = 0;
    for (int i = 0; i < K; i++) {
      floating_t p = SAMPLE(beta, alpha_p, beta_p);
      statistic += SAMPLE(bernoulli, p);
    }
    
  PSTATE = statistic;
  PC++;
});




CALLBACK(stats, {
    std::string fileName = "tests/" + testName + ".csv";
    std::ofstream resultFile (fileName, std::ios_base::app);
    if(resultFile.is_open()) {
    	floating_t maxWeight = WEIGHTS[0];
    	for (int i = 1; i < N; i++) if (WEIGHTS[i] > maxWeight) maxWeight = WEIGHTS[i];
	/* Use the weights to choose the subsample in a numerically stable way. */
    	floating_t probs[N]; 
    	for (int i = 0; i < N; i++) probs[i] = exp(WEIGHTS[i] - maxWeight) ;
    
    	for (int j = 0; j < M; j++) {
      		int k = SAMPLE(discrete, probs, N); //doesn't work on GPU
      		//int k = adiscrete(probs, N);
      		resultFile << PSTATES[k] << ", " << WEIGHTS[k] << "\n";
      		
    	}
    
 
    	resultFile.close();
    } else {
      printf("Couldnot open file %s\n", fileName.c_str());
    }
})



MAIN({
    if(argc > 2) { 
      numRuns = atoi(argv[2]);			
    }
    else {
      numRuns = 1;
    }
    
    ADD_BBLOCK(testBernoulli);
    SMC(stats);
  })
