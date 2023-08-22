data {
 int<lower=0> N;
 vector<lower=0>[N] alpha;
}
parameters {
 vector[N] y;
}
transformed parameters {
 real logr = log_sum_exp(y);
 simplex[N] x = exp(y - logr);
}
model {
 target += sum(y) - N * logr - (logr - log(N))^2 / 2;
 target += target_density_lp(x, alpha);
}
