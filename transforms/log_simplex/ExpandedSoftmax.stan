functions{
  vector augmented_softmax_log_simplex_constrain_lp(vector y) {
    int N = nrows(y);
    real log_r = log_sum_exp(y);
    vector[N] log_x = y - log_r;
    target += -log_r;
    target += std_normal_lupdf(log_r - log(N));
    return log_x;
  }
}
data {
  int<lower=0> N;
  vector<lower=0>[N] alpha;
}
parameters {
  vector[N] y;
}
transformed parameters {
  vector<upper=0>[N] log_x = augmented_softmax_log_simplex_constrain_lp(y);
  simplex[N] x = exp(log_x);
}
model {
  target += target_density_lp(x, alpha);
}
