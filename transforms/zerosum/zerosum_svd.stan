data {
  int K;
  matrix[K, K] U_alpha;
  vector[K] d_alpha;
}
parameters {
  vector[K-1] alpha1;
}
transformed parameters {
  vector[K] alpha0 = diag_post_multiply(U_alpha, d_alpha) * append_row(alpha1, 0);
}
model {
  alpha0 ~ normal(0, inv(sqrt(1 - inv(K))));
}
generated quantities {
  vector[K] alpha = softmax(alpha0);
}