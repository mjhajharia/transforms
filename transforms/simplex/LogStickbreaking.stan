functions {
   vector log_break_simplex_constrain_lp(vector y, data vector log_linspaced_vector) {
    int N = num_elements(y) + 1;
    vector[N] log_x;
    vector[N - 1] z = log_inv_logit(y - log_linspaced_vector);

    log_x[1] = z[1];
    log_x[2:N] = cumulative_sum(log1m_exp(z));
    log_x[2:N - 1] += z[2:N - 1];

    // The jacobian adjustment is in fact 0.
    // The stick breaking transform has a log det jac of:
    //
    // target += sum(log_x);
    // 
    // when we take the log of that transform the log jacobian adjustment is:
    //
    // target += -sum(log_x);
    //
    // So we end up with a net 0 adjustment.

    return log_x;
  }

  // we can stay on the log-scale with a log Dirichlet pdf
  // the adjustment is |d g^-1(x) / dy | = exp(x)
  // in the Dirichlet this simplifies to prod( x .^ alpha )
    real log_dirichlet_lpdf(vector log_x, vector alpha) {
      real norm_constant = - sum(lgamma(alpha)) + lgamma(sum(alpha)); 
      return log_x' * alpha + norm_constant;
  }

    real log_dirichlet_lupdf(vector log_x, vector alpha) {
      return log_x' * alpha;
  }
}
data {
  int<lower=0> N;
  vector<lower=0>[N] alpha;
}
transformed data {
  vector[N - 1] log_linspaced_vector = log(reverse(linspaced_vector(N - 1, 1, N - 1)));
}
parameters {
  vector[N - 1] y;
}
transformed parameters {
  vector[N] log_x = log_break_simplex_constrain_lp(y, log_linspaced_vector);
}
model {
  target += target_density_lp(log_x, alpha);
}
