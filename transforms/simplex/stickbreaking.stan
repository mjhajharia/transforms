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
  for (n in 1:N-1) {
      z[n] = inv_logit(y[n] - log(N-n));
      x[n] = (1-sum(x[1:n-1]))*z[n];
      }   
  x[N] = 1-sum(x[1:N-1]);
}
model {
  for (n in 1:N-1) {
      target += log(z[n]) + log1m(z[n]) + log1m(sum(x[1:n-1]));
    }
 target += target_density_lp(x, alpha);
}

