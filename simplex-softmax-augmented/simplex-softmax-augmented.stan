data {
 int<lower=0> K;
 vector<lower=0>[K] alpha;
}
parameters {
 vector[K] z;
}
transformed parameters {
 real<lower=0> logr = log_sum_exp(z);
 simplex[K] x = exp(z - logr);
}
model {
 target += sum(z) - exp(2 * logr) / 2;
 target += dirichlet_lupdf(x | alpha);
}
