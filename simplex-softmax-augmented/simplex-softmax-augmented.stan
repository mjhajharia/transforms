data {
 int<lower=0> N;
}
parameters {
 vector[N] z;
}
transformed parameters {
 real<lower=0> logr = log_sum_exp(z);
 simplex[N] x = exp(z - logr);
}
model {
 target += sum(z) - exp(2 * logr) / 2;
}
