data {
  int<lower=0> N;
  vector<lower=0>[N] alpha;
}
parameters {
  vector[N-1] y;
}
transformed parameters {
  simplex[N] x;
  vector[N-1] z;
  z[1:N-1] = inv_logit(y[1:N-1] - reverse(log([1:N-1])));
  x[1:N-1] = (1-append_row(0,cumulative_sum(x[1:N-2])))*z[1:N-1];
  x[N] = 1-sum(x[1:N-1]);
}
model {
 target += log(z[1:N-1]) + log1m(z[1:N-1]) + log1m(append_row(0,cumulative_sum(x[1:N-2])));
 target += target_density_lp(x, alpha);
}

