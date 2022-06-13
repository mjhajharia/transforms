data {
 int<lower=0> N;
}
parameters {
 vector[N-1] alpha_unc;
}
transformed parameters {
 simplex[N] alpha = softmax(append_row(alpha_unc, 0));
}
model {
 target += log_determinant(diag_matrix((alpha)-(alpha'*alpha)));
}
