data {
  int<lower=0> K;
  vector<lower=0>[K] alpha;
}
parameters {
  simplex[K] x;
}
model{
  target += dirichlet_lupdf(x | alpha);
}
