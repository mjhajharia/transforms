data {
 int<lower=0> N;
 vector<lower=0>[N] alpha;
}
parameters {
 vector[N] y;
}
transformed parameters {
 real<lower=0> logr = log_sum_exp(y);
 simplex[N] x = exp(y - logr);
}
model {
 target += sum(y) - exp(0.5 * logr) / 0.5;
 target += dirichlet_lupdf(x | alpha);
}
