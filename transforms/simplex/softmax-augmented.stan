data {
 int<lower=0> N;
 vector<lower=0>[N] alpha;
 int eval_model;
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
 target += target_density_lp(x, alpha);
}
