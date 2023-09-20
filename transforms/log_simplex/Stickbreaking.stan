functions {
  vector stickbreaking_log_simplex_constrain_lp(vector y) {
    int N = rows(y) + 1;
    vector[N-1] log_z = log_inv_logit(y - log(reverse(linspaced_vector(N - 1, 1, N - 1))));
    vector[N] log_x;
    log_x[1] = log_z[1];
    real log_cum_sum = negative_infinity();
    for (i in 2:N - 1) {
      log_cum_sum = log_sum_exp(log_cum_sum, log_x[i - 1]);
      log_x[i] = log1m_exp(log_cum_sum) + log_z[i];
    }   
    log_x[N] = log1m_exp(log_sum_exp(log_cum_sum, log_x[N-1]));
    target += log_x[N];
    return log_x;    
  }
}
data {
  int<lower=0> N;
  vector<lower=0>[N] alpha;
}
parameters {
  vector[N - 1] y;
}
transformed parameters {
  vector<upper=0>[N] log_x = stickbreaking_log_simplex_constrain_lp(y);
  simplex[N] x = exp(log_x);
}
model {
  target += target_density_lp(log_x, alpha);
}
