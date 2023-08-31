functions {
   vector log_break_simplex_constrain_lp(vector y, data vector log_linspaced_vector) {
    int N = num_elements(y) + 1;
    vector[N] log_x;
    vector[N - 1] z = log_inv_logit(y - log_linspaced_vector);

    log_x[1] = z[1];
    log_x[2:N] = cumulative_sum(log1m_exp(z));
    log_x[2:N - 1] += z[2:N - 1];

    target += sum(log_x);

    return log_x;
  }
}
data {
  int<lower=0> N;
  vector<lower=0>[N] alpha;
}
transformed data {
  vector[N - 1] log_linspaced_vector = log(reverse(linspaced_vector(N - 1, 1, N - 1)));
}
parameters {
  vector[N - 1] y;
}
transformed parameters {
  vector[N] log_x = log_break_simplex_constrain_lp(y, log_linspaced_vector);
}
model {
  target += target_density_lp(log_x, alpha);
}
