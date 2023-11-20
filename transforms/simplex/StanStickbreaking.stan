data {
  int<lower=0> N;
  int<lower=0, upper=1> dirichlet_target;

  vector<lower=0>[dirichlet_target ? 0 : N-1] mu;
  cov_matrix[dirichlet_target ? 1 : N-1] sigma;
  vector<lower=0>[dirichlet_target ? N : 0] alpha;
}
parameters {
  simplex[N] x;
}
model {
 if (dirichlet_target==1){
    target += target_density_lp(x, alpha);
 }
 if (dirichlet_target==0){
    x ~ multi_logit_normal(mu, sigma);
 }

}