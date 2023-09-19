functions{
  vector inv_alr_log_simplex_constrain_lp(vector y){
    int N = rows(y) + 1;
    vector[N] log_x;
    real log_r = log1p_exp(log_sum_exp(y));
    for (i in 1:(N - 1))
      log_x[i] = y[i] - log_r;
    log_x[N] = -log_r;
    target += -log_r;
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
  vector<upper=0>[N] log_x = inv_alr_log_simplex_constrain_lp(y);
}
model {
  target += target_density_lp(log_x, alpha);
}
