data {
  int<lower=0> N;
  vector<lower=0>[N] alpha;
}
parameters {
  vector[N] y;
}
transformed parameters {
  simplex[N] x;
  for (n in 1:N) {
    x[n] = y[1] + sum()
      }   
  x[N] = 1-sum(x[1:N-1]);
}
model {
  for (n in 1:N-1) {
      target += log(z[n]) + log1m(z[n]) + log1m(sum(x[1:n-1]));
    }
 target += target_density_lp(x, alpha);
}

