/**
 * Return the multivariate logistic normal density for the specified log simplex.
 *
 * See: https://en.wikipedia.org/wiki/Logit-normal_distribution#Multivariate_generalization
 * 
 * @param theta a simplex (N rows)
 * @param mu location of normal (N-1 rows)
 * @param L_Sigma Cholesky factor of covariance (N-1 rows, N-1 cols)
 */
real multi_logit_normal_cholesky_lpdf(vector theta, vector mu, matrix L_Sigma) {
  real lp;
  lp += sum(-log(theta));
  lp += multi_normal_cholesky_lpdf(log(theta[1:N-1] / theta[N]) | mu, L_Sigma);
  return lp;
}
