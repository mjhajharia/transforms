functions {
  vector cumulative_logsumexp(vector x) {
    real running_max = negative_infinity();
    real r = 0;
    int N = num_elements(x);
    vector[N] y;
    
    for (n in 1 : N) {
      if (x[n] <= running_max) {
        r += exp(x[n] - running_max);
      } else {
        r *= exp(running_max - x[n]);
        r += 1.;
        running_max = x[n];
      }
      
      y[n] = log(r) + running_max;
    }
    return y;
  }
}
data {
  int<lower=0> N;
  vector<lower=0>[N] alpha;
}
transformed data {
  vector[N - 1] center_vec = log(reverse(linspaced_vector(N - 1, 1, N - 1)));
}
parameters {
  vector[N - 1] y;
}
transformed parameters {
  vector[N - 1] z = y - center_vec;
  real logabsjac = 0;
  vector[N] log_x;
  simplex[N] x;
  
  log_x[1 : N - 1] = log_inv_logit(z);
  log_x[N] = 0;
  logabsjac += sum(log_x[1 : N - 1]);
  log_x[2 : N] += cumulative_sum(log1m_inv_logit(z));
  logabsjac += log_x[N]
               + sum(log1m_exp(cumulative_logsumexp(log_x[1 : N - 2])));
  x = exp(log_x);
}
model {
  target += logabsjac;
  target += target_density_lp(x, alpha);
}