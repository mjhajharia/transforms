data {
 int<lower=0> N;
// vector<lower=0>[N] alpha;
}
transformed data {
  real half_logN = 0.5 * log(N);
  int Nm1 = N - 1;
}
parameters {
 vector[Nm1] y;
 
}
transformed parameters {
 simplex[N] x ;
 real log_abs_det = 0;

 {
   real remaining = 1; // Remaining amount to be distributed
   real lb = 0; // The minimum for the next element
   real ub = 1;
 
    for(i in 1:Nm1) {
      int N_prime = Nm1 + 2 - i; // Number of remaining elements
      real off_set = logit(1/N_prime^2);
      //First constrain to [0; 1 / N_prime]
      ub = inv(N_prime);
      real xcons = ub * inv_logit(off_set + y[i]);
      log_abs_det += log(ub) + log_inv_logit(off_set + y[i]) + log1m_inv_logit(off_set + y[i]);

      // Add the lowest element log density
      log_abs_det += log(N_prime - 1) +  log(N_prime) + (N_prime - 2) * log1m(N_prime * xcons);
      
      x[i] = lb + remaining * xcons;
      lb = x[i];
      //We added  remaining * x_cons to each of the N_prime elements yet to be processed
      remaining -= remaining * xcons * N_prime; 
    }
    x[N] = lb + remaining;
 }

}
model {
 target += log_abs_det;
// target += target_density_lp(x, alpha);
}