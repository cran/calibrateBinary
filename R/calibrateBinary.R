#' Calibration for Binary Outputs
#'
#' @description The function performs the L2 calibration method for binary outputs.
#'
#' @param Xp a design matrix with dimension \code{np} by \code{d}.
#' @param yp a response vector with length \code{np}. The values in the vector are 0 or 1.
#' @param Xs1 a design matrix with dimension \code{ns} by \code{d}. These columns should one-by-one correspond to the columns of \code{Xp}.
#' @param Xs2 a design matrix with dimension \code{ns} by \code{q}.
#' @param ys a response vector with length \code{ns}. The values in the vector are 0 or 1.
#' @param K a positive integer specifying the number of folds for fitting kernel logistic regression and generalized Gaussian process. The default is 5.
#' @param lambda a vector specifying lambda values at which CV curve will be computed for fitting kernel logistic regression. See \code{\link{cv.KLR}}.
#' @param kernel input for fitting kernel logistic regression. See \code{\link{KLR}}.
#' @param nu input for fitting kernel logistic regression. See \code{\link{KLR}}.
#' @param power input for fitting kernel logistic regression. See \code{\link{KLR}}.
#' @param rho \code{rho} value at which CV curve will be computed for fitting kernel logistic regression. See \code{\link{KLR}}.
#' @param sigma a vector specifying values of the tuning parameter \eqn{\sigma} at which CV curve will be computed for fitting generalized Gaussian process. See Details.
#' @param lower a vector of size \code{p+q} specifying lower bounds of the input space for \code{rbind(Xp,Xs1)} and \code{Xs2}.
#' @param upper a vector of size \code{p+q} specifying upper bounds of the input space for \code{rbind(Xp,Xs1)} and \code{Xs2}.
#' @param verbose logical. If \code{TRUE}, additional diagnostics are printed. The default is \code{TRUE}.
#'
#' @details The function performs the L2 calibration method for computer experiments with binary outputs. The input and ouput of physical data are assigned to \code{Xp} and \code{yp}, and the input and output of computer data are assigned to \code{cbind(Xs1,Xs2)} and \code{ys}. Note here we separate the input of computer data by \code{Xs1} and \code{Xs2}, where \code{Xs1} is the shared input with \code{Xp} and \code{Xs2} is the calibration input. The idea of L2 calibration is to find the calibration parameter that minimizes the discrepancy measured by the L2 distance between the underlying probability functions in the physical and computer data. That is, \deqn{\hat{\theta}=\arg\min_{\theta}\|\hat{\eta}(\cdot)-\hat{p}(\cdot,\theta)\|_{L_2(\Omega)},} where \eqn{\hat{\eta}(x)} is the fitted probability function for physical data, and \eqn{\hat{p}(x,\theta)} is the fitted probability function for computer data. In this L2 calibration framework, \eqn{\hat{\eta}(x)} is fitted by the kernel logistic regression using the input \code{Xp} and the output \code{yp}. The tuning parameter \eqn{\lambda} for the kernel logistic regression can be chosen by k-fold cross-validation, where k is assigned by \code{K}. The choices of the tuning parameter are given by the vector \code{lambda}. The kernel function for the kernel logistic regression can be given by \code{kernel}, where Matern kernel or power exponential kernel can be chosen. The arguments \code{power}, \code{nu}, \code{rho} are the tuning parameters in the kernel functions. See \code{\link{KLR}}. For computer data, the probability function \eqn{\hat{p}(x,\theta)} is fitted by the Bayesian Gaussian process in Williams and Barber (1998) using the input \code{cbind(Xs1,Xs2)} and the output \code{ys}, where the Gaussian correlation function, \deqn{R_{\sigma}(\mathbf{x}_i,\mathbf{x}_j)=\exp\{-\sum^{d}_{l=1}\sigma(x_{il}-x_{jl})^2 \},} is used here. The vector \code{sigma} is the choices of the tuning parameter \eqn{\sigma}, and it will be chosen by k-fold cross-validation. More details can be seen in Sung et al. (unpublished). The arguments \code{lower} and \code{upper} are lower and upper bounds of the input space, which will be used in scaling the inputs and optimization for \eqn{\theta}. If they are not given, the default is the range of each column of \code{rbind(Xp,Xs1)}, and \code{Xs2}.
#' @return a matrix with number of columns \code{q+1}. The first \code{q} columns are the local (the first row is the global) minimal solutions which are the potential estimates of calibration parameters, and the \code{(q+1)}-th column is the corresponding L2 distance.
#'
#' @seealso \code{\link{KLR}} for performing a kernel logistic regression with given \code{lambda} and \code{rho}. \code{\link{cv.KLR}} for performing cross-validation to estimate the tuning parameters.
#' @author Chih-Li Sung <iamdfchile@gmail.com>
#'
#' @import GPfit
#' @import gelnet
#' @import kernlab
#' @import randtoolbox
#' @importFrom stats optim
#'
#' @examples
#' library(calibrateBinary)
#'
#' set.seed(1)
#' #####   data from physical experiment   #####
#' np <- 10
#' xp <- seq(0,1,length.out = np)
#' eta_fun <- function(x) exp(exp(-0.5*x)*cos(3.5*pi*x)-1) # true probability function
#' eta_x <- eta_fun(xp)
#' yp <- rep(0,np)
#' for(i in 1:np) yp[i] <- rbinom(1,1, eta_x[i])
#'
#' #####   data from computer experiment   #####
#' ns <- 20
#' xs <- matrix(runif(ns*2), ncol=2)  # the first column corresponds to the column of xp
#' p_xtheta <- function(x,theta) {
#'      # true probability function
#'      exp(exp(-0.5*x)*cos(3.5*pi*x)-1) - abs(theta-0.3) *exp(-0.5*x)*cos(3.5*pi*x)
#' }
#' ys <- rep(0,ns)
#' for(i in 1:ns) ys[i] <- rbinom(1,1, p_xtheta(xs[i,1],xs[i,2]))
#'
#' #####    check the true parameter    #####
#' curve(eta_fun, lwd=2, lty=2, from=0, to=1)
#' curve(p_xtheta(x,0.3), add=TRUE, col=4)   # true value = 0.3: L2 dist = 0
#' curve(p_xtheta(x,0.9), add=TRUE, col=3)   # other value
#'
#' ##### calibration: true parameter is 0.3 #####
#' \donttest{
#' calibrate.result <- calibrateBinary(xp, yp, xs[,1], xs[,2], ys)
#' print(calibrate.result)
#' }
#' @export

calibrateBinary <- function(Xp, yp, Xs1, Xs2, ys,
                        K = 5, lambda = seq(0.001,0.1,0.005),
                        kernel = c("matern","exponential")[1],
                        nu = 1.5, power = 1.95,
                        rho = seq(0.05,0.5,0.05),
                        sigma = seq(100,20,-1),
                        lower, upper, verbose = TRUE){

  if (is.matrix(Xp) == FALSE) {
    Xp <- as.matrix(Xp)
  }
  if (is.matrix(Xs1) == FALSE) {
    Xs1 <- as.matrix(Xs1)
  }
  if (is.matrix(Xs2) == FALSE) {
    Xs2 <- as.matrix(Xs2)
  }
  if (is.factor(ys) == FALSE) {
    ys <- as.factor(ys)
  }

  if(ncol(Xs1) != ncol(Xp)){
    stop("ncol(Xs1) != ncol(Xp)")
  }

  if(nrow(Xs1) != nrow(Xs2)){
    stop("nrow(Xs1) != nrow(Xs2)")
  }

  if(missing(lower)){
    lower <- c(apply(rbind(Xs1, Xp), 2, min), apply(Xs2, 2, min))
  }

  if(missing(upper)){
    upper <- c(apply(rbind(Xs1, Xp), 2, max), apply(Xs2, 2, max))
  }

  Xs <- cbind(Xs1, Xs2)
  ns <- nrow(Xs)
  np <- nrow(Xp)
  d <- ncol(Xp)
  q <- ncol(Xs) - d

  if(verbose){
    cat("number of input variables =", d+q,  "where", q, "of them", ifelse(q==1, "is", "are"), "calibration input.\n")
  }

  #####     scale the input data to [0,1]                #####
  if(verbose) cat("scaling input data to region [0,1] for model fitting.\n")
  Xp <- t((t(Xp) - lower[1:d])/(upper[1:d] - lower[1:d]))
  Xs <- t((t(Xs) - lower)/(upper - lower))

  x.test <- sobol(20^d, dim = d)  # for integration
  if(d == 1) x.test <- matrix(x.test, ncol=1)

  #####     performs kernel logistic regression          #####
  #####     for the data from physical experiments       #####
  if(verbose) cat("running kernel logistic regression via cross-validation:\n")
  cv.out <- cv.KLR(Xp, yp, K = K, lambda = lambda, kernel = kernel, nu = nu, power = power, rho = rho)
  if(verbose) cat("        lambda =", cv.out$lambda, "and rho =", cv.out$rho, "\n")
  etahat <- KLR(Xp, yp, x.test, lambda = cv.out$lambda, kernel = kernel, nu = nu, power = power, rho = cv.out$rho)

  #####     performs generalized Gaussian regression     #####
  #####     for the data from computer  experiments      #####
  if(verbose) cat("running generalized Gaussian regression via cross-validation:\n")
  sigma.hat <- cv.gausspr(Xs, ys, K = K, sigma = sigma)
  if(verbose) cat("        sigma =", sigma.hat, "\n")
  gp.fit <- gausspr(Xs, ys,
                    kpar = list(sigma = sigma.hat),
                    type = "classification", scaled = FALSE, cross = 0, fit = FALSE)

  L2_fun <- function(theta) {
    x.new <- cbind(x.test, matrix(theta, ncol = q, nrow = nrow(x.test), byrow = TRUE))
    sqrt(mean((etahat - predict(gp.fit, x.new, type="probabilities")[,2])^2))
  }

  if(verbose) cat("running optimization:\n")
  ini.val <- sobol(3^q, q)
  if(q==1) ini.val <- matrix(ini.val,ncol=1)
  opt.out <- vector("list", 3^q)
  for(i in 1:3^q) opt.out[[i]] <- optim(ini.val[i,], L2_fun, lower = rep(0,q), upper = rep(1,q), method = "L-BFGS-B")
  if(verbose) cat("        done.\n")

  opt.val <- sapply(opt.out, function(x) x$value)
  opt.sol <- sapply(opt.out, function(x) x$par)
  if(q > 1) opt.sol <- t(opt.sol)

  out <- cbind(opt.sol, opt.val)
  colnames(out) <- c(paste0("par.",1:q), "l2 dist")
  out <- out[sort.int(out[,"l2 dist"], index.return = TRUE)$ix,]
  out <- unique(round(out, 5))

  #####      scale back to original range     #####
  if(verbose) cat("scaling back to the original region.\n")
  out[, 1:q] <- t((t(out[, 1:q]) * (upper[(d+1):(d+q)] - lower[(d+1):(d+q)]) + lower[(d+1):(d+q)]))

  return(out)
}
