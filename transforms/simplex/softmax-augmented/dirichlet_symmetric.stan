functions{
    real f_lp(int eval_model, vector x, vector alpha)
    {
        if (eval_model==1)
            return dirichlet_lpdf(x | alpha);
        else
            return 0; 
            }
}
data {
 int<lower=0> N;
 vector<lower=0>[N] alpha;
 int eval_model;
}
parameters {
 vector[N] y;
}
transformed parameters {
 real<lower=0> logr = log_sum_exp(y);
 simplex[N] x = exp(y - logr);
}
model {
 target += sum(y) - exp(0.5 * logr) / 0.5;
 target += f_lp(eval_model, x, alpha);
}
