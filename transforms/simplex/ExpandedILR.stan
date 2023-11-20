data {
  int<lower=0> N;
  int<lower=0, upper=1> dirichlet_target;

  vector<lower=0>[dirichlet_target ? 0 : N-1] mu;
  cov_matrix[dirichlet_target ? 1 : N-1] sigma;
  vector<lower=0>[dirichlet_target ? N : 0] alpha;
}
transformed data {
  matrix[N - 1, N - 1] Vinv = construct_vinv(N);
  real logN = log(N);
}
parameters {
 vector[N - 1] y;
}
transformed parameters {
  vector[N] s = append_row(Vinv * y, 0);
  real logr = log_sum_exp(s);
  simplex[N] x = exp(s - logr);
}
model {
  target += sum(s) - N * logr + logN;
  if (dirichlet_target==1){
      target += target_density_lp(x, alpha);
  }
  if (dirichlet_target==0){
      x ~ multi_logit_normal(mu, sigma);
  }

  }
