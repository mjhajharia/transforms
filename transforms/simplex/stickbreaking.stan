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
  z = inv_logit(y[1:N-1] - log(reverse(linspaced_vector(N-1, 1,N-1))));
  for (n in 1:N-1) {
      x[n] = (1-sum(x[1:n-1]))*z[n];
      }   
  x[N] = 1-sum(x[1:N-1]);
}
model {
 target += log(z[1:N-1]) + log1m(z[1:N-1]) + log1m(cumulative_sum(append_row(0, x[1:N-2])));
 target += target_density_lp(x, alpha);
}

