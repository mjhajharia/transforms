data {
 int<lower=0> N;
 vector<lower=0>[N] alpha;
}
parameters {
 vector[N] y;
}
transformed parameters {
 simplex[N] x = y^2;
}
model {
 target += log_determinant(2*y*transpose(y));
 target += target_density_lp(x, alpha);
}
