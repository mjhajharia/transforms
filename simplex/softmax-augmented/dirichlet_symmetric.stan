data {
 int<lower=0> N;
 vector<lower=0>[N] alpha;
 real<lower=0> p;
}
parameters {
 vector[N] y;
}
transformed parameters {
 real<lower=0> logr = log_sum_exp(y);
 simplex[N] x = exp(y - logr);
}
model {
 target += sum(y) - exp(p * logr) / p;
 target += dirichlet_lupdf(x | alpha);
}
