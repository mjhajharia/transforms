data {
  int<lower=0> N;
  int<lower=0, upper=1> dirichlet_target;

  vector<lower=0>[dirichlet_target ? 0 : N-1] mu;
  cov_matrix[dirichlet_target ? 1 : N-1] sigma;
  vector<lower=0>[dirichlet_target ? N : 0] alpha;
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
 if (dirichlet_target==1){
    target += target_density_lp(x, alpha);
 }
 if (dirichlet_target==0){
    x ~ multi_logit_normal(mu, sigma);
 }

}