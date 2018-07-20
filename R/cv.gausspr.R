cv.gausspr <- function(X, y, K=5, sigma){

  if (is.matrix(X) == FALSE) {
    X <- as.matrix(X)
  }

  d <- ncol(X)
  sigma.mx <- matrix(sigma, ncol=1)
  error.vt <- apply(sigma.mx, 1, function(para) cross(gausspr(X, as.factor(y), kpar=list(sigma=para), type= "classification", scaled = FALSE, cross = K, fit = FALSE)))

  return(sigma.mx[which.min(error.vt),])
}

