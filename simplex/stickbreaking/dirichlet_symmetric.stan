data {
  int<lower=0> K;
}
parameters {
  vector[K-1] y;
}
transformed parameters {
  simplex[K] x;
  vector[K-1] z;
  for (k in 1:K-1) {
      z[k] = inv_logit(y[k] - log(K-k));
      x[k] = (1-sum(x[1:k-1]))*z[k];
      }   
  x[K] = 1-sum(x[1:K-1]);
}
model {
  for (k in 1:K-1) {
      target += log(z[k]) + log1m(z[k]) + log1m(sum(x[1:k-1]));
    }
}
