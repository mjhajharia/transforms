functions {
  matrix helmert_matrix(int D) {
    matrix[D, D] helmert_mat;
    real inv_nrm2 = inv_sqrt(D);
    helmert_mat[1, 1:D] = rep_row_vector(inv_nrm2, D);
    for (d in 2:D) {
      inv_nrm2 = inv_sqrt(d * (d - 1));
      helmert_mat[d, 1:(d - 1)] = rep_row_vector(inv_nrm2, d - 1);
      helmert_mat[d, d] = -(d - 1) * inv_nrm2;
      helmert_mat[d, (d + 1):D] = rep_row_vector(0, D - d);
    }
    return helmert_mat;
  }

  matrix semiorthogonal_matrix(int N) {
    return helmert_matrix(N)[2:N]';
  }

  vector inv_ilr_simplex_constrain_lp(vector y, matrix V) {
    int N = rows(y) + 1;
    vector[N] z = V * y;
    real logr = log_sum_exp(z);
    vector[N] x = exp(z - logr);
    target += sum(z) - N * logr + 0.5 * log(N);
    return x;
  }
}
data {
  int<lower=0> N;
  vector<lower=0>[N] alpha;
}
transformed data {
  matrix[N, N - 1] V = semiorthogonal_matrix(N);
}
parameters {
  vector[N - 1] y;
}
transformed parameters {
  simplex[N] x = inv_ilr_simplex_constrain_lp(y, V);
}
model {
  target += target_density_lp(x, alpha);
}
