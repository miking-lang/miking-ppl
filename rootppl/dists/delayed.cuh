/**
 * Beta distribution type
 *
 * @param alpha first shape.
 * @param beta second shape.
 */
struct beta_t {
  floating_t alpha;
  floating_t beta;
  
  DEV beta_t(){}; 
  DEV beta_t(floating_t alpha_, floating_t beta_) {
    alpha = alpha_;
    beta = beta_;
  }
};


/**
 * Gamma Type
 *
 * @param k shape.
 * @param theta scale.
 */
struct gamma_t {
  floating_t k;
  floating_t theta;
  
  DEV gamma_t(){}; 
  DEV gamma_t(floating_t k_, floating_t theta_) {
    k = k_;
    theta = theta_;
  }
};


/**
 * NormalInverseGamma Type
 *
 * σ^2 | a,b ~ InverseGamma(a, b)
 * m ~ N(m0, v σ^2)
 * 
 * <=> σ^2, m ~ NormalInverseGamma(m0, v, a, b) 
 *
 * @param m0 initial mean.
 * @param v scale by which the variance is multiplied.
 * @param a gamma-parameter of the variance (shape).
 * @param b gamma-parameter of the variance (scale).
 */
struct normalInverseGamma_t {
  floating_t m0;
  floating_t v;
  floating_t a;
  floating_t b;

  DEV normalInverseGamma_t(){};
  DEV normalInverseGamma_t(floating_t m0_, floating_t v_, floating_t a_, floating_t b_) {
    m0 = m0_;
    v = v_;
    a = a_;
    b = b_;
  }
};


/**
 * Returns a sample from the Bernoulli distribution with a Beta prior (BetaBernoulli).
 * 
 * @param p beta_t distribution of the success probability.
 * @return 1 with probability sampled from p, 0 otherwise.
 */
DEV floating_t betaBernoulli(RAND_STATE_DECLARE beta_t& p);


/**
 * Score of betaBernoulli
 * 
 * @param p beta_t distribution of the success probability.
 * @param x variate.
 *
 * @retun log-probability.
 */
HOST DEV floating_t betaBernoulliScore(beta_t& p, int x);


/**
 * Delayed Sampling from a GammaExponential Mixture
 *
 * E.g. to sample a waiting time.
 *
 * t - returns waiting time
 * rate - a reference to the prior (will modify)
 * f - factor by which the scale may be multiplied
 */
DEV floating_t sample_GammaExponential(RAND_STATE_DECLARE gamma_t& rate, floating_t f);


/**
 * Log-score (delayed) of observation from a GammaExponential
 * 
 * E.g. to observe a waiting time.
 * 
 * score - returns the log score
 * x - observation
 * rate - reference to rate (will modify)
 * f - factor by which the scale may be multiplied
 */
DEV floating_t score_GammaExponential(floating_t x, gamma_t& rate,  floating_t f);



/**
 * Log-score (delayed) of observation from a GammaPoisson
 * 
 * E.g. to observe a number of speciation points.
 * 
 * score - returns the log score
 * x - observation
 * rate - reference to rate (will modify)
 * f - factor by which the scale may be multiplied
 */
DEV floating_t score_GammaPoisson(floating_t x, floating_t t, gamma_t& rate, floating_t f);


/**
 * Returns a Sample from NormalInverseGammaNormal (delayed)
 *
 * E.g. to sample a factor by which the speciation rate is multiplied.
 *
 * m, σ^2 ~ NIG(m0, v, a, b)
 * f ~ N(m, σ^2)
 *
 * @param prior parameters (will be updated).
 */
DEV floating_t sample_NormalInverseGammaNormal(RAND_STATE_DECLARE normalInverseGamma_t& prior);
