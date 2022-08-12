data {
  int<lower=0> N;
  vector<lower=0>[N] alpha;
}
parameters {
  vector[N - 1] y;
}
transformed parameters {
  vector[N-1] z = inv_logit(y[1:N - 1] - log(reverse(linspaced_vector(N - 1, 1, N - 1))));
  simplex[N] x;
  x[1] = z[1];
  real cum_sum = 0;
  for (n in 2:N - 1) {
    cum_sum += x[n - 1];
    x[n] = (1 - cum_sum) * z[n];
  }   
  x[N] = 1 - (cum_sum+x[N-1]);
}
model {
 target += log(z[1:N - 1]) + log1m(z[1:N - 1]) + log1m(cumulative_sum(append_row(0, x[1:N - 2])));
 target += target_density_lp(x, alpha);
}

