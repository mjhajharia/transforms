data {
  int<lower=0> N;
  int<lower=0, upper=1> dirichlet_target;

  vector<lower=0>[dirichlet_target ? 0 : N-1] mu;
  cov_matrix[dirichlet_target ? 1 : N-1] sigma;
  vector<lower=0>[dirichlet_target ? N : 0] alpha;
}
parameters {
  vector[N - 1] y;
}
transformed parameters {
  simplex[N] x;
  real log_det_jacobian = 0;
  {
    real yi;
    real zi, zprod = 1, zprod_new;
  
    for (i in 1:(N-1)) {
      yi = y[i];
      zi = Phi(yi);
      log_det_jacobian += std_normal_lpdf(yi |);
      zprod_new = zprod * zi;
      log_det_jacobian += log(zprod);
      x[i] = zprod - zprod_new;
      zprod = zprod_new;
    }
    x[N] = zprod;
  }
}
model {
  target += log_det_jacobian; 
  if (dirichlet_target==1){
      target += target_density_lp(x, alpha);
  }
  if (dirichlet_target==0){
      x ~ multi_logit_normal(mu, sigma);
  }

  }