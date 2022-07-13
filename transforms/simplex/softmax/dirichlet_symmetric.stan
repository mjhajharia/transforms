// functions{
//     real f_lp(int eval_model, vector x, vector alpha)
//     {
//         if (eval_model==1)
//             return dirichlet_lpdf(x | alpha);
//         else
//             return 0; 
//             }
// }
data {
 int<lower=0> N;
 vector<lower=0>[N] alpha;
//  int eval_model;
}
parameters {
 vector[N-1] y;
}
transformed parameters {
 simplex[N] x = softmax(append_row(y,0));
}
model {
 target += -N * log1p_exp(log_sum_exp(y)) + sum(y);
 target += dirichlet_lupdf(x | alpha);
//  target += f_lp(eval_model, x, alpha);
}
