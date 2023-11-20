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
  vector[N-1] z = inv_logit(y[1:N - 1] - log(reverse(linspaced_vector(N - 1, 1, N - 1))));
  simplex[N] x;
  x[1] = z[1];
  real cum_sum = 0;
  for (n in 2:N - 1) {
    cum_sum += x[n - 1];
    x[n] = (1 - cum_sum) * z[n];
  }   
  x[N] = 1 - (cum_sum+x[N-1]);
}
model {
//  target += log(z[1:N - 1])  + log1m(z[1:N - 1]) + log1m(cumulative_sum(append_row(0.0, x[1:N - 2])));
//commented out version causes an error for autodiff, equivalent modification below
 target += sum(log(z[1:N - 1])  + log1m(z[1:N - 1]) + log1m(cumulative_sum(append_row(0.0, x[1:N - 2]))));
 if (dirichlet_target==1){
    target += target_density_lp(x, alpha);
 }
 if (dirichlet_target==0){
    x ~ multi_logit_normal(mu, sigma);
 }

}