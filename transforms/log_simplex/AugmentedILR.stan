functions{
  vector inv_ilr_log_simplex_constrain_lp(vector y, matrix Vinv){
    int N = rows(y) + 1;
    vector[N - 1] s = Vinv * y;
    real log_r = log1p_exp(log_sum_exp(s));
    vector[N] log_x;
    log_x[1:N - 1] = s - log_r;
    log_x[N] = -log_r;
    target += log(N) - log_r;
    return log_x;
  }
}
data {
  int<lower=0> N;
  vector<lower=0>[N] alpha;
}
transformed data {
  matrix[N - 1, N - 1] Vinv = construct_vinv(N);
}
parameters {
  vector[N - 1] y;
}
transformed parameters {
  vector<upper=0>[N] log_x = inv_ilr_log_simplex_constrain_lp(y, Vinv);
  simplex[N] x = exp(log_x);
}
model {
  target += target_density_lp(log_x, alpha);
}
