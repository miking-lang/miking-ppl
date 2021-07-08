# CorePPL
To be written...

# RootPPL
RootPPL can be seen as an intermediate language for representing probabilistic models and comes with a framework that performs inference on the GPU in these models. These models are currently hard-coded, see examples in the folder rootppl/models. The idea is that high-level Miking probabilistic programming languages should be compiled to this intermediate language. 

## Getting Started
The instructions below are tested on Ubuntu 18.04 but should work for other Linux distributions, Windows, and Mac. 

### Prerequisites
Before building RootPPL programs, a C++/CUDA compiler is required. RootPPL works on CPU and Nvidia GPU:s. For the CPU version, a C++ compiler should suffice, e.g. g++.  In 
order to build for GPU, CUDA must be installed. See
CUDA installation guides: [Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/ "CUDA Installation Guide Linux"), 
[Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html "CUDA Installation Guide Windows"), 
[Mac](https://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html "CUDA Installation Guide Mac").

To run the CPU version in parallel, [OpenMP](https://www.openmp.org/resources/openmp-compilers-tools/) must be installed. 
OpenMP comes with recent gcc versions. However, OpenMP is not necessary if one does not want to execute programs in parallel on the CPU.

### Install rootppl
To install rootppl for the current user, first clone this repository and change directory to the rootppl folder. Then run:
```
make install
```
This will install rootppl to `$HOME/.local/bin` with resources copied to `$HOME/.local/lib/rootppl`. Some systems, e.g. Mac OS, will require manually adding `$HOME/.local/bin` to `$PATH`. 

### Build
Note: before building and after installation, especially if you are re-installing,
execute a `rootppl clean`, first.

To compile a model and build an executable:
```
rootppl path/to/model.cu
```
This will compile the model along with the inference framework for CPU. To compile it for GPU, add your GPU:s compute capability to the arch variable.
You can find your GPU:s compute capability in the [Wikipedia table](https://en.wikipedia.org/wiki/CUDA#GPUs_supported).
Here is an example that will compile the airplane example for a GPU with a minimum compute capability of 7.5
(simply remove `--arch 75` to compile for CPU):
```
rootppl models/airplane/airplane.cu --arch 75 -j 5
```
The optional argument `-j x` speeds up the compilation process by spawning `x` jobs, allowing for parallel compilation. 
The corresponding parallel CPU (OpenMP) example is:
```
rootppl models/airplane/airplane.cu --omp -j 5
```
Alternatively, the c++ compiler can be specified with CXX. This is often required on Mac OS to enable OpenMP, by using g++ instead of the default clang. On Mac OS, g++ can be installed with e.g. `brew install gcc`. Then, assuming the version installed was `gcc-10`: 
```
rootppl models/airplane/airplane.cu --omp -j 5 --cxx g++-10
```

The first build will compile the entire inference framework and can take 20 seconds or so when building for GPU. (Run `rootppl` with the `-j num_threads` to use multiple threads and speed up the build). 
__Note that if the inference framework is compiled for GPU, and then the model is compiled for CPU, 
there will be errors. So always perform a `rootppl clean` before switching between CPU and GPU. The same goes for switching to/from OpenMP.__

This should generate an executable named `program` (add `-o <exec_name>` to name it differently). Execute it with `./program num_particles`. For example:
```
./program 1000
```

An example output of this:
```
./program 1000
-119.422143
Num particles close to target: 98.9%, MinX: 56.2349, MaxX: 91.1572
```
First is the command that is executed, it executes the executable `program` with program argument `1000`.
The second row is the normalizing constant. The last row is a a print statement within the model. 

You can supply a second argument to `program`, which is the number of sweeps:
```
./program 1000 3
-115.482724
Num particles close to target: 89.7%, MinX: 51.5005, MaxX: 88.2039
-115.472101
Num particles close to target: 90.5%, MinX: 53.0577, MaxX: 90.0298
-115.496258
Num particles close to target: 88.7%, MinX: 48.1773, MaxX: 89.0957
```

### Creating a simple model
Models are divided into fragments to enable pausing the execution within models. 
These fragments are functions referred to as basic blocks (`BBLOCK`). 
To control the program execution flow, a program counter (`PC`) can be modified. 
If it remains unchanged in when the basic block returns, resampling will be done, and then
the same block will be executed again. The program counter corresponds to the index of the basic block 
to be executed. So, incrementing it means that the next block will be executed after resampling. An 
example of this can be seen in the example below. However, if no following blocks are defined, 
the inference will terminate as the model program has been executed. 

Any interesting model will contain random choices, i.e. sampling from distributions. 
Below is a coin flip example which flips a biased coin. `SAMPLE` is a macro that can take variable number of arguments.
First is the distribution name, followed by the distribution specific arguments. 

Sampling is done differently on the CPU and the GPU,but these differences are hidden in the model with the help 
of this `SAMPLE` macro. All of these RootPPL keywords are macros similar to `SAMPLE`, they provide higher-level 
constructs that can be used in the same way regardless of if the model will be executed on the CPU or the GPU.   

```CUDA
BBLOCK(coinFlip, {
    int coinSample = SAMPLE(bernoulli, 0.6);
    PC++;
})
```

In the coin flip example above, there is one problem, however. The sample is only stored in a local
variable and is never used. To store data that remains when the next block is executed, the program
state (`PSTATE`) should be used. Before defining the blocks, the model must be
initialized with the macro `INIT_MODEL` that takes two arguments. First the type of the program
state (this could be any type, e.g. `int` or a structure), then the number of basic blocks in
the program. So, adding it to the above example:

```CUDA
INIT_MODEL(int, 1)
  
BBLOCK(coinFlip, {
    PSTATE = SAMPLE(bernoulli, 0.6);
    PC++;
})
```

Now to run this program, only one thing remains to be defined, the main function.
This is done with the `MAIN` macro, taking a code block as argument.
The code below shows what we need in order to run the program and perform SMC. 

```CUDA
MAIN({
    ADD_BBLOCK(coinFlip);
 
    SMC(NULL);
})
```

First, the block must be added to the array of blocks to be executed. The order in which
blocks are added with `ADD_BLOCK`, defines their order in the array and thus
defines the order of execution together with the program counter. Secondly, the
`SMC` macro (its parameter is explained below) starts the inference. 

Now the model can be compiled and executed! However, the result of the coin flips is
stored, but never used. To aggregate the results of all the particles' samples, 
a callback function (`CALLBACK`) can be used. The callback will be called after
inference, but before clean up of the particles. This way, the results can be used to
generate desired distributions or values before particles are deleted. 
Within the `CALLBACK` macro, the array of program states is accessed 
with `PSTATES` and the number of particles is accessed with the 
parameter `N` (which is hidden within the macro). Also, the log-weights array can be accessed with
the `WEIGHTS` macro. These weights are normalised (so that the exponent of these log-weights sum to 1).
Here is an example callback, called "sampleMean" that calculates and prints the mean 
of the samples. 

```CUDA
CALLBACK(sampleMean, {
    double sum = 0;
    for(int i = 0; i < N; i++)
        sum += PSTATES[i];
    double mean = sum / N;
    printf("Sample mean: %f\n", mean);
})
```

For this to be used, the `SMC` macro call in main must be 
changed to: `SMC(sampleMean);`

This example can be found in
[rootppl/models/simple-examples/coin_flip_mean.cu](rootppl/models/simple-examples/coin_flip_mean.cu)
and, being in the rootppl directory, compiled with:
```
rootppl models/simple-examples/coin_flip_mean.cu
```

Then it can be executed with the executable followed by the
number of particles, for example: `./program 1000`. 

An example output is then:
```
Sample mean: 0.608000
log normalization constant = 0.000000
```

First we see our callback function's output. Then on the next line, is the logarithm of the
normalization constant approximated by the inference. This is simply 0 here, since the model contains
no statements that alter the weights of the particles. 

### Supplementary Examples

Below follows some examples, these are all models that are defined within one single block. 

#### Coin Flip Posterior
Full example: [rootppl/models/simple-examples/coin_flip.cu](rootppl/models/simple-examples/coin_flip.cu)

In this model, a bias for a coin is sampled from the prior beta distribution. Then we observe that the coin flip is true. This model thus infers the
posterior distribution of the bias, conditioned on the observation. 
```CUDA
BBLOCK(coinFlip, {
    double x = SAMPLE(beta, 2, 2);
    OBSERVE(bernoulli, x, true);

    PSTATE = x;
    PC++;
})
```

#### Gaussian Mixture Model
Full example: [rootppl/models/simple-examples/mixture.cu](rootppl/models/simple-examples/mixture.cu)

This model demonstrates an example of *stochastic branching*, meaning that different code is executed depending on the outcome of the sample. 
```CUDA
BBLOCK(mixture, {
    double x;
    if(SAMPLE(bernoulli, 0.7))
        x = SAMPLE(normal, -2, 1);
    else
        x = SAMPLE(normal, 3, 1);
 
    PSTATE = x;
    PC++;
})
```

#### Geometric Distribution (Recursive)
Full example: [rootppl/models/simple-examples/geometric_recursive.cu](rootppl/models/simple-examples/geometric_recursive.cu)

This model combines stochastic branching with recursion. Basic blocks do not fully support recursion themselves, as they take no custom arguments or return values. Instead, a helper function is used to express the recursive model:

```CUDA
BBLOCK_HELPER(geometricRecursive, {
    if(SAMPLE(bernoulli, p))
        return 1;
    else
        return BBLOCK_CALL(geometricRecursive, p) + 1;

}, int, double p)
```
```CUDA
BBLOCK(geometric, {
    PSTATE = BBLOCK_CALL(geometricRecursive, 0.6);
    PC++;
})
```

Note that the helper function takes its return value and parameters comma-separated after the function body. 

While recursive functions is supported by CUDA, iterative solutions are encouraged. Below is the same model, implemented with a loop instead.
#### Geometric Distribution (Iterative)
Full example: [rootppl/models/simple-examples/geometric_iterative.cu](rootppl/models/simple-examples/geometric_iterative.cu)
```CUDA
BBLOCK(geometric, {
    int numFlips = 1;
    while(! SAMPLE(bernoulli, 0.6))
        numFlips++;
    PSTATE = numFlips;
    PC++;
})
```

### Phylogenetic Models

More sophisticated models can be found in the [phylogenetics directory](rootppl/models/phylogenetics). These probabilistic
models to inference on observed phylogenetic trees. These observed trees can be found in [phylogenetics/tree-utils/trees.cuh](rootppl/models/phylogenetics/tree-utils/trees.cuh).
The correct phylogenetic models contain a link in the top of the file to the WebPPL source code used as reference when implementing them.
These models contain a number of new things, e.g.: 
- Multiple basic blocks
- Structures as program states
- Recursive simulations
- Global data used by the model
- Resampling throughout the traversal of the observed tree

#### Constant-rate birth-death
In [phylogenetics/crbd](rootppl/models/phylogenetics/crbd), the constant-rate birth-death models can be found. 
The most interesting file here is [crbd.cu](rootppl/models/phylogenetics/crbd/crbd.cu) as it is a correct model
used by evolutionary biologists. This model uses a pre-processed DFS traversal path over the observed tree, rather than using a call stack. 

#### Further phylogenetic models
Further phylogenetic models are found in the package
[`phyrppl`](https://github.com/vsenderov/phyrppl).

### The RootPPL Architecture

The architecture's main goal is to decouple the inference from the probabilistic models, this is the case for probabilistic
programming languages. In the `inference` directory, the code for inference methods can be found. Currently, only SMC is supported. 
Within the `smc` directory, the `resampling` directory contain resampling strategies. Currently only Systematic Resampling is supported. The resampling has 
a parallel implementation for the GPU, and a sequential and parallel implementation for the CPU. 

- The `dists` directory contains distributions and scoring functions. Both models and inference can rely on these. 
- The `macros` directory contains all the macros/constructs that are mostly used by models but also in inference code. 
- The `utils` directory contains code that RootPPL uses, but can be used by models as well. 
- The `models` directory contains example models that use the inference. 

### Compile with the built-in stack
To compile with the `progStateStack_t` program state, the macro `INIT_MODEL_STACK(num_bblocks)` should be used instead of `INIT_MODEL(progState_t, num_bblocks)`.
Then the stack size is specified in bytes with `--stack_size num_bytes`, e.g. `rootppl my_stack_model.cu --stack_size 10000`.

### Building a more interesting model
TODO, stuff not yet demonstrated explicitly in README: 
- WEIGHT macro
- Multiple BBLOCK models
- Global data accessible by bblocks on GPU
- Program States containing more than a primitive datatype, e.g. structs. 

### RootPPL ChangeLog
7/7/2021: added BetaBernoulli distribution