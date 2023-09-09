functions {
  vector sum_rows(matrix A) {
    vector[rows(A)] row_sums;
    for (i in 1 : rows(A)) {
      row_sums[i] = sum(row(A, i));
    }
    return row_sums;
  }
  
  // lupdf of a distribution with dirichlet and logistic-normal special cases
  // alpha: size N vector
  // Omega: size (N-1, N-1) positive semi-definite matrix
  // when Omega=0 and alpha[i]>0 for all i, this is the dirichlet distribution with parameters alpha
  // when sum(alpha)=0 and Omega is positive definite, this is the logistic-normal 
  // with precision matrix Omega and location parameter inv(Omega) * head(alpha, N-1)
  // J Aitchison. A General Class of Distributions on the Simplex.
  // Journal of the Royal Statistical Society. Series B (Methodological), 41(1): 1-7, 136-146.
  // doi: 10.1111/j.2517-6161.1985.tb01341.x url: https://www.jstor.org/stable/2345555
  real dirichlet_logisticnormal_lpdf(vector x, vector alpha, matrix Omega) {
    int N = rows(x);
    vector[N - 1] Omega_colsum = sum_rows(Omega);
    real Omega_sum = sum(Omega_colsum);
    vector[N] z = log(x);
    vector[N - 1] z_minus = head(z, N - 1);
    real z_N = z[N];
    real lp = dot_product(alpha, z) - sum(z) -
              (quad_form_sym(Omega, z_minus) + Omega_sum * z_N ^ 2) / 2 +
              z_N * dot_product(Omega_colsum, z_minus);
    return lp;
  }
  
  real target_density_lp(vector x, vector alpha, matrix Omega) {
    return dirichlet_logisticnormal_lpdf(x | alpha);
  }
}
