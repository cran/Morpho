##' Create Matrices necessary for Thin-Plate Spline
#' 
#' Create (Bending Engergy) Matrices necessary for Thin-Plate Spline, and
#' sliding of Semilandmarks
#' 
#' 
#' @param matrix k x 3 or k x 2 matrix containing landmark coordinates.
#' @param lambda numeric: regularization factor
#' @param blockdiag logical: request blockdiagonal matrix Lsubk3 needed for
#' sliding of semilandmarks.
#' @return
#' \item{L }{Matrix L as specified in Bookstein (1989)}
#' \item{Linv }{Inverse of matrix L as specified in Bookstein (1989)}
#' \item{Lsubk }{uper left k x k submatrix of \code{Linv}}
#' \item{Lsubk3 }{Matrix used for sliding in \code{\link{slider3d}} and \code{\link{relaxLM}}. Only available if \code{blockdiag = TRUE}}
#' @note This function is not intended to be called directly - except for
#' playing around to grasp the mechansims of the Thin-Plate Spline.
#' @seealso \code{\link{tps3d}, \link{warp.mesh}}
#' @references Gunz, P., P. Mitteroecker, and F. L. Bookstein. 2005.
#' Semilandmarks in Three Dimensions, in Modern Morphometrics in Physical
#' Anthropology. Edited by D. E. Slice, pp. 73-98. New York: Kluwer
#' Academic/Plenum Publishers.
#' 
#' Bookstein FL. 1989. Principal Warps: Thin-plate splines and the
#' decomposition of deformations. IEEE Transactions on pattern analysis and
#' machine intelligence 11(6).
#' 
#' @examples
#' 
#' require(rgl)
#' data(boneData)
#' L <- CreateL(boneLM[,,1])
#' ## calculate Bending energy between first and second specimen:
#' be <- t(boneLM[,,2])%*%L$Lsubk%*%boneLM[,,2]
#' ## calculate Frobenius norm 
#' sqrt(sum(be^2))
#' ## the amount is dependant on on the squared scaling factor
#' # scale landmarks by factor 5 and compute bending energy matrix
#' be2 <- t(boneLM[,,2]*5)%*%L$Lsubk%*%(boneLM[,,2]*5)
#' sqrt(sum(be2^2)) # exactly 25 times the result from above
#' ## also this value is not symmetric:
#' L2 <- CreateL(boneLM[,,2])
#' be3 <- t(boneLM[,,1])%*%L2$Lsubk%*%boneLM[,,1]
#' sqrt(sum(be3^2))
#' 
#' @export
CreateL <- function(matrix,lambda=0, blockdiag=TRUE)
{
    if (dim(matrix)[2] == 3) {
        k <- dim(matrix)[1]
        Q <- cbind(1,matrix)
        O <- matrix(0,4,4)
        if (!is.matrix(matrix) || !is.numeric(matrix))
        stop("matrix must be a numeric matrix")
        K <- .Call("createL",matrix)
        
        diag(K) <- lambda
        L <- rbind(cbind(K,Q),cbind(t(Q),O))
        L1 <- try(solve(L),silent=TRUE)
        if (class(L1)=="try-error") {
            cat("CreateL: singular matrix: general inverse will be used.\n")
            L1 <- armaGinv(L)		
        }
        Lsubk <- L1[1:k,1:k]
        Lsubk3 <- NULL
        if (blockdiag) {
            Lsubk3 <- matrix(0,3*k,3*k)
            Lsubk3[1:k,1:k] <- Lsubk
            Lsubk3[(k+1):(2*k),(k+1):(2*k)] <- Lsubk
            Lsubk3[(2*k+1):(3*k),(2*k+1):(3*k)] <- Lsubk
        }
        return(list(L=L,Linv=L1,Lsubk=Lsubk,Lsubk3=Lsubk3))
    } else if (dim(matrix)[2] == 2) {
        out <- CreateL2D(matrix, lambda, blockdiag=blockdiag)
        return(out)
    } else
        stop("only works for matrices with 2 or 3 columns")
}
CreateL2D <- function(matrix, lambda=0, blockdiag=TRUE)
{
    k <- dim(matrix)[1]
    K <- matrix(0,k,k)
    Q <- cbind(1,matrix)
    O <- matrix(0,3,3)

    for (i in 1:k) {
        for (j in 1:k) {
            r2 <- sum((matrix[i,]-matrix[j,])^2)
            K[i,j] <- r2*log(r2)
        }
    }
    K[which(is.na(K))] <- 0
    diag(K) <- lambda
    L <- rbind(cbind(K,Q),cbind(t(Q),O))
    
	L1 <- try(solve(L),silent=TRUE)
    	if (class(L1)=="try-error") {
            cat("singular matrix: general inverse will be used.\n")
            L1 <- armaGinv(L)		
        }
    Lsubk <- L1[1:k,1:k]
    Lsubk3 <- NULL
    if (blockdiag)
        Lsubk3 <- rbind(cbind(Lsubk,matrix(0,k,k)),cbind(matrix(0,k,k),Lsubk))
    return(list(L=L,Linv=L1,Lsubk=Lsubk,Lsubk3=Lsubk3))
}
