data {
 int<lower=2> N;
}
transformed data {
  matrix[N, N] d = rep_matrix(-1.0 / (N - 1), N, N);
  matrix[N, N] V;
  vector[N] eV;
  d = add_diag(d, 1 - diagonal(d));
  
  V = eigenvectors_sym(d);
  eV = eigenvalues_sym(d);
  
  for (i in 1:N) {
    if (eV[i] < 0) eV[i] = 0;
    eV[i] = sqrt(eV[i]); 
  }
}
parameters {
  vector[N] z;
}
transformed parameters {
  vector[N] y = diag_post_multiply(V, eV) * z;
}
model {
 // user may choose or estimate mu or sigma
 z ~ normal(0, 1);
 target += sum(log(eV));
}
