functions {
  vector stick_break_simplex_constrain_lp(vector y, data vector log_linspaced_vector) {
    int Nm1 = rows(y);
    vector[Nm1] z = inv_logit(y - log_linspaced_vector);
    real logabs_jacobian = sum(log1m(z));
    real cum_sum = 0;
    vector[Nm1 + 1] x; // simplex vector

    for n in 2:Nm1 {
      cum_sum += z[n - 1];
      z[n] *= (1 - cum_sum);
    }
    
    x = append_row(z, 1 - (cum_sum + z[Nm1]));

    logabs_jacobian += sum(log(x[1:Nm1]));

    target += logabs_jacobian;

    return x;
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
  simplex[N] x = stick_break_simplex_constrain_lp(y, log_linspaced_vector);
}
model {
 target += target_density_lp(x, alpha);
}
