functions {
  matrix helmert_matrix(int D) {
    int Dm1 = D - 1;
    matrix[D, D] helmert_mat;
    real inv_nrm2 = inv_sqrt(D);
    for (j in 1:D)
      helmert_mat[1, j] = inv_nrm2;
    for (i in 2:D) {
      inv_nrm2 = inv_sqrt(i * (i - 1));
      for (j in 1:i)
        helmert_mat[i, j] = inv_nrm2;
      helmert_mat[i, i] = -(i - 1) * inv_nrm2;
      for (j in (i + 1):D)
        helmert_mat[i, j] = 0;
    }
    return helmert_mat;
  }

  matrix semiorthogonal_matrix(int N) {
    matrix[N, N] H = helmert_matrix(N);
    matrix[N, N - 1] V = transpose(H[2:N, 1:N]);
    return V;
  }

  vector inv_ilr_simplex_constrain_lp(vector y, matrix V) {
    int N = rows(y) + 1;
    vector[N] z = V * y;
    real logr = log_sum_exp(z);
    vector[N] x = exp(z - logr);
    target += sum(z) - N * logr + log(N);
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
