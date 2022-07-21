data {
 int<lower=0> N;
 vector<lower=0>[N] alpha;
}
parameters {
 real p=1/N;
 vector[N] y;
}
transformed parameters {
 real<lower=0> logr = log_sum_exp(y);
 simplex[N] x = exp(y - logr);
}
model {
 target += sum(y) - exp(p * logr) / p;
 target += target_density_lp(x, alpha);
}
