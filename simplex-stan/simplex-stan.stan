data {
  int<lower=0> K;
}
parameters {
  simplex[K] x;
}