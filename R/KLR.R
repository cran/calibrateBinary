#' Kernel Logistic Regression
#'
#' @description The function performs a kernel logistic regression for binary outputs.
#'
#' @param X a design matrix with dimension \code{n} by \code{d}.
#' @param y a response vector with length \code{n}. The values in the vector are 0 or 1.
#' @param xnew a testing matrix with dimension \code{n_new} by \code{d} in which each row corresponds to a predictive location.
#' @param lambda a positive value specifing the tuning parameter for KLR. The default is 0.01.
#' @param kernel "matern" or "exponential" which specifies the matern kernel or power exponential kernel. The default is "matern".
#' @param nu a positive value specifying the order of matern kernel if \code{kernel} == "matern". The default is 1.5 if matern kernel is chosen.
#' @param power a positive value (between 1.0 and 2.0) specifying the power of power exponential kernel if \code{kernel} == "exponential". The default is 1.95 if power exponential kernel is chosen.
#' @param rho a positive value specifying the scale parameter of matern and power exponential kernels. The default is 0.1.
#'
#' @details This function performs a kernel logistic regression, where the kernel can be assigned to Matern kernel or power exponential kernel by the argument \code{kernel}. The arguments \code{power} and \code{rho} are the tuning parameters in the power exponential kernel function, and \code{nu} and \code{rho} are the tuning parameters in the Matern kernel function. The power exponential kernel has the form \deqn{K_{ij}=\exp(-\frac{\sum_{k}{|x_{ik}-x_{jk}|^{power}}}{rho}),} and the Matern kernel has the form \deqn{K_{ij}=\prod_{k}\frac{1}{\Gamma(nu)2^{nu-1}}(2\sqrt{nu}\frac{|x_{ik}-x_{jk}|}{rho})^{nu} \kappa(2\sqrt{nu}\frac{|x_{ik}-x_{jk}|}{rho}).} The argument \code{lambda} is the tuning parameter for the function smoothness.
#'
#' @return Predictive probabilities at given locations \code{xnew}.
#'
#' @seealso \code{\link{cv.KLR}} for performing cross-validation to choose the tuning parameters.
#' @references Zhu, J. and Hastie, T. (2005). Kernel logistic regression and the import vector machine. Journal of Computational and Graphical Statistics, 14(1), 185-205.
#'
#' @author Chih-Li Sung <iamdfchile@gmail.com>
#'
#' @import GPfit
#' @import gelnet
#'
#' @examples
#' library(calibrateBinary)
#'
#' set.seed(1)
#' np <- 10
#' xp <- seq(0,1,length.out = np)
#' eta_fun <- function(x) exp(exp(-0.5*x)*cos(3.5*pi*x)-1) # true probability function
#' eta_x <- eta_fun(xp)
#' yp <- rep(0,np)
#' for(i in 1:np) yp[i] <- rbinom(1,1, eta_x[i])
#'
#' x.test <- seq(0,1,0.001)
#' etahat <- KLR(xp,yp,x.test)
#'
#' plot(xp,yp)
#' curve(eta_fun, col = "blue", lty = 2, add = TRUE)
#' lines(x.test, etahat, col = 2)
#'
#' #####   cross-validation with K=5    #####
#' ##### to determine the parameter rho #####
#'
#' cv.out <- cv.KLR(xp,yp,K=5)
#' print(cv.out)
#'
#' etahat.cv <- KLR(xp,yp,x.test,lambda=cv.out$lambda,rho=cv.out$rho)
#'
#' plot(xp,yp)
#' curve(eta_fun, col = "blue", lty = 2, add = TRUE)
#' lines(x.test, etahat, col = 2)
#' lines(x.test, etahat.cv, col = 3)
#'
#' @export

KLR <- function(X, y, xnew, lambda = 0.01,
                kernel = c("matern","exponential")[1],
                nu = 1.5, power = 1.95, rho = 0.1){

  if (!is.matrix(X)) {
    X <- as.matrix(X)
  }
  if (!is.factor(y)) {
    y <- as.factor(y)
  }
  if (!is.matrix(xnew)) {
    xnew <- as.matrix(xnew)
  }

  # computer K
  beta <- -log10(rep(rho,ncol(X)))
  if(kernel == "matern") {
    K <- corr_matrix(X, beta, corr = list(type = "matern", nu = nu))
  }else{
    K <- corr_matrix(X, beta, corr = list(type = "exponential", power = power))
  }
  fit <- gelnet.ker(K, y, lambda, silent = TRUE)
  n <- nrow(X)

  phi_new <- apply(xnew, 1, function(xn){
    xn <- matrix(xn, nrow = 1)
    if(kernel == "matern") {
      temp <- 10^beta
      temp <- matrix(temp, ncol = ncol(xn), nrow = (length(X)/ncol(xn)),
                     byrow = TRUE)
      temp <- 2 * sqrt(nu) * abs(X - as.matrix(rep(1, n)) %*%
                                   (xn)) * (temp)
      ID <- which(temp == 0)
      rd <- (1/(gamma(nu) * 2^(nu - 1))) * (temp^nu) * besselK(temp,
                                                               nu)
      rd[ID] <- 1
      return(matrix(apply(rd, 1, prod), ncol = 1))
    }else{
      return(exp(-(abs(X - as.matrix(rep(1, n)) %*% (xn))^power) %*%
                (10^beta)))
    }
  })

  f <- c(fit$v %*% phi_new)+fit$b
  p <- exp(f)/(1+exp(f)) # probability of 0
  p <- 1 - p # probability of 1

  return(p)
}
