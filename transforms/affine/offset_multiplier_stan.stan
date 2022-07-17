data {
 real mu;
 real sigma;
}
parameters {
 real y;
}
transformed parameters {
 real <offset=mu, multiplier=sigma> x = mu + sigma*y;
}
model {
 target += sigma;
}
