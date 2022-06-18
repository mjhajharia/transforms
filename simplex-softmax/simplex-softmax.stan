data {
 int<lower=0> N;
 vector<lower=0>[N] alpha;
}
parameters {
 vector[N-1] y;
}
transformed parameters {
 simplex[N] x = softmax(append_row(y,0));
}
model {
 target += -N * log1p_exp(log_sum_exp(y)) + sum(y);
 target += dirichlet_lupdf(x | alpha);
}
