data {
 int<lower=0> N;
 vector<lower=0>[N] alpha;
}
transformed data {
 real half_logN = 0.5 * log(N);
}
parameters {
 vector[N - 1] y;
}
transformed parameters {
 simplex[N] x = softmax(append_row(y,0));
}
model {
 target += sum(y) - N * log_sum_exp(append_row(y, 0)) + half_logN;
 target += target_density_lp(x, alpha);
}
