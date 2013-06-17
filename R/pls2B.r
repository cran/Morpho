pls2B <- function(y,x,tol=1e-12, same.config=FALSE, rounds=0, mc.cores=detectCores())
  {
    landmarks <- FALSE
    xorig <- x
    yorig <- y
    win <- FALSE
    if(.Platform$OS.type == "windows")
      win <- TRUE
    else
      registerDoParallel(cores=mc.cores)### register parallel backend
    
    if (length(dim(x)) == 3)
      {
        landmarks <- TRUE
        x <- vecx(x)
      }
    if (length(dim(y)) == 3)
      y <- vecx(y)
    else
      landmarks <- FALSE
      
    xdim <- dim(x)
    ydim <- dim(y)

    if (same.config && !landmarks)
      warning("the option same.config requires landmark array as input")
    
    
    cova <- cov(cbind(x,y))
    svd.cova <- svd(cova[1:xdim[2],c((xdim[2]+1):(xdim[2]+ydim[2]))])

    svs <- svd.cova$d
    svs <- svs/sum(svs)
    svs <- svs[which(svs > 0.001)]

    covas <- svs*100
    l.covas <- length(covas)
    z1 <- x%*%svd.cova$u[,1:l.covas] #pls scores of x
    z2 <-  y%*%svd.cova$v[,1:l.covas] #pls scores of y
    
### calculate correlations between pls scores
    cors <- 0
    for(i in 1:length(covas))
      {cors[i] <- cor(z1[,i],z2[,i])
     }

### Permutation testing
    permupls <- function(i)
      {
        x.sample <- sample(1:xdim[1])
        y.sample <- sample(x.sample)
        if (same.config && landmarks)
          {
           tmparr <- bindArr(xorig[,,x.sample],yorig[,,y.sample],along=1)
           tmpproc <- ProcGPA(tmparr,silent=TRUE)
           x1 <- vecx(tmpproc$rotated[1:dim(xorig)[1],,])
           y1 <- vecx(tmpproc$rotated[1:dim(yorig)[1],,])
         }
        else
          {
            x1 <- x
            y1 <- y
          }
        cova.tmp <- cov(cbind(x1[x.sample,],y1[y.sample,]))
        svd.cova.tmp <- svd(cova.tmp[1:xdim[2],c((xdim[2]+1):(xdim[2]+ydim[2]))])
        svs.tmp <- svd.cova.tmp$d
        return(svs.tmp[1:l.covas])
      }
    p.values <- rep(NA,l.covas)
    if (rounds > 0)
      {
        if (win)
          permuscores <- foreach(i = 1:rounds, .combine = cbind) %do% permupls(i)
        else
          permuscores <- foreach(i = 1:rounds, .combine = cbind) %dopar% permupls(i)
        
        p.val <- function(x,rand.x)
          {
            p.value <- length(which(rand.x >= x))
            
            if (p.value > 0)
              {
                p.value <- p.value/rounds
              }
            else
              {p.value <- 1/rounds}
            gc()
            return(p.value)
          }
        
        for (i in 1:l.covas)
          {
            p.values[i] <- p.val(svd.cova$d[i],permuscores[i,])
          }
      }
### create covariance table
    Cova <- data.frame(svd.cova$d[1:l.covas],covas,cors,p.values)
    colnames(Cova) <- c("singular value","% total covar.","Corr. coefficient", "p-value")
    out <- list(svd=svd.cova,Xscores=z1,Yscores=z2,CoVar=Cova)
    return(out)
  }