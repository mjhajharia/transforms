data {
  int<lower=0> K;
}
parameters {
  vector[(K*(K-1)) %/% 2] y;
}
transformed parameters {
  matrix[K, K] L = diag_matrix(rep_vector(1, K));
  vector<lower=-1, upper=1>[(K*(K-1)) %/% 2] z = tanh(y);
  real log_det_jacobian = 0;
  {
    int counter = 1;
    real sum_sqs;

    for (i in 2:K) {
      L[i, 1] = z[counter];
      counter += 1;
      sum_sqs = square(L[i, 1]);
      for (j in 2:(i-1)) {
        log_det_jacobian += 0.5 * log1m(sum_sqs);
        L[i, j] = z[counter] * sqrt(1 - sum_sqs);
        counter += 1;
        sum_sqs = sum_sqs + square(L[i, j]);
      }
      L[i, i] = sqrt(1 - sum_sqs);
    }
  }
}
model {
  target += log_det_jacobian;
  // target += target_density(L);
}
