functions {
  // number of entries in lower/upper triangle of NxN matrix
  int length_tri(int N) {
    return (N * (N + 1)) %/% 2;
  }

  // given the length_tri(N)-long vector x, construct the NxN symmetric matrix
  // whose lower triangle is x
  matrix symmetrize_from_vec(vector x, int N) {
    matrix[N,N] S;
    real xk;
    int k = 1;
    for (i in 1:N) {
      for (j in 1:(i-1)) {
        xk = x[k];
        S[i,j] = xk;
        S[j,i] = xk;
        k += 1;
      }
      S[i,i] = x[k];
      k += 1;
    }
    return S;
  }
}
