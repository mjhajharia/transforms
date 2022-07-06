parameters {
 real y;
}
transformed parameters {
 real<lower=0> x = inv_logit(y);
}
model {
 target += log(inv_logit(y)*(1-inv_logit(y)));
}
