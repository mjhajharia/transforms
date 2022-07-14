data {
  int<lower=0> K;
}
transformed data {
   
}
parameters {
  vector[(K*(K-1)) %/% 2] y;
}
transformed parameters {
  matrix[K, K] L = diag_matrix(rep_vector(1, K));
  real log_det_jacobian;
  {
    int counter = 1;

    for (i in 2:K) {
      L[i, 1] = y[counter];
      counter += 1;
      for (j in 2:(i-1)) {
        L[i, j] = y[counter];
        counter += 1;
      }
      L[i , :i] = L[i , :i] / sqrt(sum(square(L[i , :i ])));
      log_det_jacobian += (K - i + 1) * log(L[i,i]);
    }
  }
}
model {
  target += log_det_jacobian;
  target += lkj_corr_cholesky_lpdf(L | 2.);
}
