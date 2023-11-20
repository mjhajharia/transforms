data {
  int<lower=0> N;
  int<lower=0, upper=1> dirichlet_target;

  vector<lower=0>[dirichlet_target ? 0 : N-1] mu;
  cov_matrix[dirichlet_target ? 1 : N-1] sigma;
  vector<lower=0>[dirichlet_target ? N : 0] alpha;
}

parameters {
 vector[N] y;
}
transformed parameters {
 real logr = log_sum_exp(y);
 simplex[N] x = exp(y - logr);
}
model {
 target += sum(y) - N * logr;  // target += log(prod(x))
 target += std_normal_lupdf(logr - log(N));
 if (dirichlet_target==1){
    target += target_density_lp(x, alpha);
 }
 if (dirichlet_target==0){
    x ~ multi_logit_normal(mu, sigma);
 }

}