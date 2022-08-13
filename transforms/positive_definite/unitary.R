library(cmdstanr)

mod <- cmdstan_model("unitary.stan")

N <- 20
Z <- matrix(rexp(N^2), N, N)
# Z <- matrix( (runif(N^2) + runif(N^2)) / sqrt(2), N, N)

qr_out <- qr(Z)
Q <- qr.Q(qr_out)
R <- qr.R(qr_out)

diag_R <- diag(R)
lambda <- diag_R / abs(diag_R)

unitary <- Q %*% diag(lambda)
crossprod(unitary)

mod_out <- mod$sample(
  data = list(N = N,
              U = t(unitary)),
  parallel_chains = 4
)

exp(mod_out$summary("lambda")$mean)

mat <- matrix(mod_out$summary("x")$mean, N, N)
cov2cor(mat)

mat


chol(mat)

L <- matrix(mod_out$summary("L")$mean, N, N)
X_cor <- matrix(mod_out$summary("x")$mean, N, N)
hist(X_cor[lower.tri(X_cor)])

hist(L[lower.tri(L)])
