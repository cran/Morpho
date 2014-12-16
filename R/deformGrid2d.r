#' visualise differences between two superimposed sets of 2D landmarks
#' 
#' visualise differences between two superimposed sets of 2D landmarks by
#' deforming a cubic grid based on a thin-plate spline interpolation
#' 
#' 
#' @param matrix reference matrix containing 2D landmark coordinates or mesh of class "mesh3d"
#' @param tarmatrix target matrix containing 2D landmark coordinates or mesh of class "mesh3d"
#' @param ngrid number of grid lines to be plotted; ngrid=0 suppresses grid
#' creation.
#' @param lwd width of lines connecting landmarks.
#' @param show integer (vector): if c(1:2) both configs will be plotted, show = 1 only plots the reference and show = 2 the target.
#' plotted. Options are combinations of 1,2 and 3.
#' @param lines logical: if TRUE, lines between landmarks will be plotted.
#' @param lcol color of lines
#' @param col1 color of "matrix"
#' @param col2 color of "tarmat"
#' @param pcaxis logical: align grid by shape's principal axes.
#' @param ... additional parameters passed to plot
#' @author Stefan Schlager
#' @seealso \code{\link{tps3d}}
#' 
#' @examples
#' require(shapes)
#' proc <- procSym(gorf.dat)
#' 
#' deformGrid2d(proc$mshape,proc$rotated[,,1],ngrid=5,pch=19)
#' 
#' @export

deformGrid2d <- function(matrix,tarmatrix,ngrid=0,lwd=1,show=c(1:2),lines=TRUE,lcol=1,col1=2,col2=3,pcaxis=FALSE,...)
{
    k <- dim(matrix)[1]
    x0 <- NULL
    if (ngrid > 1) {

        x2 <- x1 <- c(0:(ngrid-1))/ngrid;
        x0 <- as.matrix(expand.grid(x1,x2))
    
        cent.mat <- apply(matrix,2,scale,scale=F)
        mean.mat <- apply(matrix,2,mean)
        xrange <- diff(range(matrix[,1]))
        yrange <- diff(range(matrix[,2]))
        
        xrange1 <- diff(range(tarmatrix[,1]))
        yrange1 <- diff(range(tarmatrix[,2]))
        
        maxi <- max(c(xrange,yrange,xrange1,yrange1))
        maxi <- 1.2*maxi
        x0 <- maxi*x0
        x0 <- apply(x0,2,scale,scale=FALSE)
        if (pcaxis)
            space <- eigen(crossprod(cent.mat))$vectors
        else
            space <- diag(2)
        x0 <- t(t(x0%*%space)+mean.mat)
        x0 <- tps3d(x0,matrix,tarmatrix)
        
        ## create deformation cube
        zinit <- NULL
        zinit0 <- zinit <- (c(1,2,2+ngrid,1+ngrid))
        for( i in 1:(ngrid-2))
            zinit <- cbind(zinit,(zinit0+i))
        zinit0 <- zinit
        for (i in 1:(ngrid-2))
            zinit <- cbind(zinit,zinit0+(i*ngrid))
    }
    lims <- apply(rbind(matrix,tarmatrix,x0),2,range)
    if (1 %in% show)
        plot(matrix,col=col1,xlim=lims[,1],ylim = lims[,2],asp=1,xlab = "",ylab = "", axes = F,...)
    if(2 %in% show) {
        if (1 %in% show)
            points(tarmatrix,col=col2,...)
        else
            plot(tarmatrix,col=col2,xlim=lims[,1],ylim = lims[,2],asp=1,xlab = "",ylab = "", axes = F,...)
        if (lines) {
            linemesh <- list()
            
            linemesh$vb <- rbind(matrix,tarmatrix)
            linemesh$it <- cbind(1:k,(1:k)+k)
            for (i in 1:nrow(linemesh$it))
                lines(linemesh$vb[linemesh$it[i,],],lwd=lwd)
        }
    }
    if (ngrid > 1) {
        for (i in 1:ncol(zinit)) {
            polygon(x0[zinit[,i],],lwd=lwd)
        }
        if (2 %in% show)
            points(tarmatrix,col=col2,...)
        if (1 %in% show)
            points(matrix,col=col1,...)
    }
}

