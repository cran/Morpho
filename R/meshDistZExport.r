#' @rdname render
#' @export
export <- function(x,...)UseMethod("export")

#' @rdname render
#' @method export meshDist
#' @importFrom Rvcg vcgPlyWrite
#' @export
export.meshDist <- function(x,file="default",imagedim="100x800",titleplot="Distance in mm",...)
{
    tol <- x$params$tol
    colramp <- x$colramp
    widxheight <- as.integer(strsplit(imagedim,split="x")[[1]])
    vcgPlyWrite(x$colMesh,filename=file,writeCol = TRUE)
    png(filename=paste(file,".png",sep=""),width=widxheight[1],height=widxheight[2])
    diffo <- ((colramp[[2]][2]-colramp[[2]][1])/2)
    image(colramp[[1]],colramp[[2]][-1]-diffo,t(colramp[[3]][1,-1])-diffo,col=colramp[[4]],useRaster=TRUE,ylab=titleplot,xlab="",xaxt="n")
    if (!is.null(tol)) {
        if (sum(abs(tol)) != 0) {
            image(colramp[[1]],c(tol[1],tol[2]),t(tol),col="green",useRaster=TRUE,add=TRUE)
        }
    }
    dev.off()
}
