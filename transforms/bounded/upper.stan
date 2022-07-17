data {
 real b;
}
parameters {
 real y;
}
transformed parameters {
 real<upper=b> x = b-exp(y);
}
model {
 target += exp(y);
}
