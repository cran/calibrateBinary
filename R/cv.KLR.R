#' K-fold cross-validation for Kernel Logistic Regression
#'
#' @description The function performs k-fold cross validation for kernel logistic regression to estimate tuning parameters.
#'
#' @param X input for \code{KLR}.
#' @param y input for \code{KLR}.
#' @param K a positive integer specifying the number of folds. The default is 5.
#' @param lambda a vector specifying lambda values at which CV curve will be computed.
#' @param kernel input for \code{KLR}.
#' @param nu input for \code{KLR}.
#' @param power input for \code{KLR}.
#' @param rho rho value at which CV curve will be computed.
#'
#' @details This function performs the k-fold cross-valibration for a kernel logistic regression. The CV curve is computed at the values of the tuning parameters assigned by \code{lambda} and \code{rho}. The number of fold is given by \code{K}.
#'
#' @return
#' \item{lambda}{value of \code{lambda} that gives minimum CV error.}
#' \item{rho}{value of \code{rho} that gives minimum CV error.}
#'
#' @seealso \code{\link{KLR}} for performing a kernel logistic regression with given \code{lambda} and \code{rho}.
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

cv.KLR <- function(X, y, K = 5, lambda = seq(0.001,0.2,0.005),
                   kernel = c("matern","exponential")[1],
                   nu = 1.5, power = 1.95, rho = seq(0.05,0.5,0.05)){

  if (!is.matrix(X)) {
    X <- as.matrix(X)
  }
  if(is.factor(y)){
    y <- as.factor(y)
  }

  para.mx <- expand.grid(rho, lambda)[,2:1]
  n <- length(y)
  all.folds <- split(sample(1:n), rep(1:K, length = n))
  pmse.mx <- matrix(0, K, nrow(para.mx))
  for (i in seq(K)) {
    omit <- all.folds[[i]]
    for(j in 1:nrow(para.mx)){
      phat <- try(KLR(X[-omit, , drop = FALSE], y[-omit], X[omit, , drop = FALSE],
                  lambda = para.mx[j,1], kernel = kernel, nu = nu, power = power,
                  rho = para.mx[j,2]),  silent = TRUE)
      if(class(phat)=="try-error"){
        pmse.mx[i,j] <- NA
      }else{
        phat[phat>0.5] <- 1
        phat[phat<=0.5] <- 0
        pmse.mx[i,j] <- mean((as.numeric(y[omit])-phat)^2)
      }
    }
  }

  if(all(is.na(pmse.mx))){
    stop("Try other lambda and rho.")
  }else{
    return(list(lambda = para.mx[which.min(apply(pmse.mx,2,mean,na.rm=TRUE)),1],
                rho = para.mx[which.min(apply(pmse.mx,2,mean,na.rm=TRUE)),2]))
  }
}
