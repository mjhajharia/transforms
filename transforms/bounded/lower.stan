data {
 real a;
}
parameters {
 real y;
}
transformed parameters {
 real<lower=a> x = exp(y)+a;
}
model {
 target += exp(y);
}
