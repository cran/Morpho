typprob <- function(x,data,small=FALSE, method=c("chisquare","wilson"), center=NULL, cova=NULL)
{
    method <- substr(method[1],1L,1L)
    if (is.matrix(x))
        nx <- dim(x)[2]
    else
        nx <- length(x)

    ndata <- dim(data)[1]
    if (is.null(center))
        center <- apply(data,2,mean)
    
    if (is.null(cova))
        cova <- cov(data)
    
    dists <- mahalanobis(x, center=center, cov=cova)
    if (method == "w")
        {
            if (small)
                dists <- ndata*log(1+(dists*ndata/(ndata^2-1)))
            t2 <- ndata*dists/(ndata+1)
            f <- t2*(ndata-nx)/(nx*(ndata-1))
            alpha <- pf(f,nx,(ndata-nx),lower.tail=T)
            alpha <- 1-alpha
        }
    else if (method == "c")
        alpha <- pchisq(dists,nx,lower.tail=F)
    else
        stop("please chose valid method")
    
    return(alpha)
}
typprobClass <- function(x,data,groups,small=FALSE,method=c("chisquare","wilson"),outlier=0.01,sep=FALSE)
{
    if (!is.factor(groups))
        {
            groups <- as.factor(groups)
            warning("groups coerced to factors")
        }
    probs <- NULL
    groups <- factor(groups)
    glev <- levels(groups)
    nlev <- length(glev)
    cova <- NULL
    if (sep)
        cova <- covW(data,groups)
    for( i in 1:nlev)
        {
            if (sep)# use separate covariance matrices for scaling
                tmp <- typprob(x,data[groups==glev[i],],small=small,method=method)
            else #used pooled within group covariance
                tmp <- typprob(x,data,small=small,method=method,center=apply(data[groups==glev[i],],2,mean),cova=cova)
            probs <- cbind(probs,tmp)
        }
    colnames(probs) <- as.character(glev)
    classify <- apply(probs,1,function(x){out <- which(x==max(x));return(out)})
    glev <- c(glev,"none")
    outsider <- which(apply(probs,1,max) < outlier)
    classify[outsider] <- nlev+1
    groupaffin <- as.factor(glev[classify])
    
    out <- (list(probs=probs,groupaffin=groupaffin))
    class(out) <- "typprob"
    return(out)
}

covW <- function(data,groups)
    {
        if (!is.factor(groups)) {
            groups <- as.factor(groups)
            warning("groups coerced to factors")
        }
        ndata <- dim(data)[1]
        groups <- factor(groups)
        glev <- levels(groups)
        nlev <- length(glev)
        gsizes <- as.vector(tapply(groups, groups, length))
        covWithin <- 0
        for (i in 1:nlev)
            if (gsizes[i] > 1)
                covWithin <- covWithin + (cov(data[groups==glev[i],]) * (gsizes[i]-1))
            else
                covWithin <- covWithin+diag(1,ncol(data))
        
        covWithin <- covWithin/(ndata - nlev)
        return(covWithin)
    }
