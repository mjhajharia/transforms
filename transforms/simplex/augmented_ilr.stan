data {
 int<lower=0> N;
 matrix[N, N - 1] Vinv;
 vector<lower=0>[N] alpha;
}
transformed data {
  real logN = log(N);
}
parameters {
 vector[N - 1] y;
 simplex[N] z;
}
transformed parameters {
  vector[N] s = Vinv * y;
  real alpha = log_sum_exp(s);
  simplex[N] x = exp(s - alpha);
}
model {
  target += sum(s) - N * alpha + logN;
  target += target_density_lp(x, alpha);
}
