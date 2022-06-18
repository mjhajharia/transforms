data {
  int<lower=0> N;
  vector<lower=0>[N] alpha;
}
parameters {
  simplex[N] x;
}
model {
  x ~ dirichlet(alpha);
}
