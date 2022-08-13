
data {
  int<lower=0> N;
  matrix[N, N] U;
}
parameters {
  vector<lower=-2, upper=2>[N] lambda;
}
transformed parameters {
  matrix[N, N] x = crossprod(diag_pre_multiply(exp(lambda), U));
  matrix[N, N] L = cholesky_decompose(quad_form_diag(x, 1 / sqrt(diagonal(x))));
}
model {
  L ~ lkj_corr_cholesky(1);
}
generated quantities {
  matrix[N, N] x_cor = multiply_lower_tri_self_transpose(L);
}
