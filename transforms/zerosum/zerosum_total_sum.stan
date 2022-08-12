parameters {
  int K;
  vector[K-1] alpha1;
}
transformed parameters {
  vector[K] alpha0 = append_row(alpha1, -sum(alpha1));
}
model {
  alpha0 ~ normal(0, 10);
}
generated quantities {
  vector[K] alpha = softmax(alpha0);
}
