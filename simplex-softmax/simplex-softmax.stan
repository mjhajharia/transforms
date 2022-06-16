data {
 int<lower=0> K;
}
parameters {
 vector[K-1] x_unc;
}
transformed parameters {
 simplex[K] x = softmax(append_row(x_unc,0));
}
model {
 target += -K * log1p(sum(exp(x_unc))) + sum(x_unc);
}
