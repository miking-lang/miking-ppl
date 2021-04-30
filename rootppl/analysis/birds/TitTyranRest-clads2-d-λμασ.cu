/*
 * ClaDS2 Model
 * - uses the framework
 * - delayed sampling for lambda0
 *
 */

#include <iostream>
#include <cstring>
#include <cassert>
#include <string>
#include <fstream>
#include <algorithm>

#include "inference/smc/smc.cuh"
#include "../../models/phylogenetics/tree-utils/tree_utils.cuh"
#include "utils/math.cuh"
#include "utils/stack.cuh"
#include "dists/delayed.cuh"

//typedef bisse32_tree_t tree_t;
//typedef primate_tree_t tree_t;
//typedef moth_div_tree_t tree_t;
typedef TitTyranRest_tree_t tree_t;
//typedef Alcedinidae_tree_t tree_t;

std::string analysisName = "TitTyranRest";
 
// Test settings

floating_t rho      = 0.6869565217391305;

floating_t k = 1;
floating_t theta = 1;
floating_t kMu = 1;
floating_t thetaMu = 0.5;

floating_t m0 = 0;
floating_t v = 1;
floating_t a = 1.0;
floating_t b = 0.2;

#include "../../models/phylogenetics/clads2/clads2-d-λμασ.cuh"

MAIN({

    ADD_BBLOCK(simClaDS2);
    ADD_BBLOCK(simTree);
    ADD_BBLOCK(conditionOnDetection);
    ADD_BBLOCK(justResample);
    ADD_BBLOCK(sampleFinalLambda);
    SMC(saveResults);
    //SMC(NULL)
})
 