data {
  int<lower=0> K;
}
parameters {
  vector[K-1] y;
}
transformed parameters {
  simplex[K] x;
  vector[K-1] Z;
  for (i in 1:K-1) {
      Z[i] = inv_logit(y[i] + (1.0/(K-i)));
      x[i] = (1-sum(x[1:i-1]))*Z[i];
      }   
  x[K] = 1-sum(x[1:K-1]);
}
model {
  for (i in 1:K-1) {
    for (j in 1:i) {
      target += log(Z[i]*(1-Z[i])*(1-sum(x[1:j-1])));
    }
  }
}