data {
 int<lower=0> N;
 vector<lower=0>[N] alpha;
}
transformed data {
  matrix[N - 1, N - 1] Vinv = construct_vinv(N);
  real logN = log(N);
}
parameters {
 vector[N - 1] y;
}
transformed parameters {
  vector[N] s = append_row(Vinv * y, 0);
  real logr = log_sum_exp(s);
  simplex[N] x = exp(s - logr);
}
model {
  target += sum(s) - N * logr + logN;
  target += target_density_lp(x, alpha);
}
