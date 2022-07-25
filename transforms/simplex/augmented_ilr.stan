data {
 int<lower=0> N;
 vector<lower=0>[N] alpha;
}
transformed data {
  real half_logN = 0.5 * log(N);
}
parameters {
 vector[N - 1] y;
}
transformed parameters {
 real<lower=0> logr = log_sum_exp(append_row(y, 0));
 simplex[N] x = append_row(exp(y - logr), exp(-logr));
}
model {
 target += sum(y) - N * logr + half_logN;
 target += target_density_lp(x, alpha);
}
