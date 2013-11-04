slider3d <- function(dat.array,SMvector,outlines=NULL,surp=NULL,sur.path="sur",sur.name=NULL, meshlist=NULL, ignore=NULL,sur.type="ply",tol=1e-05,deselect=FALSE,inc.check=TRUE,recursive=TRUE,iterations=0,initproc=TRUE,speed=TRUE,pairedLM=0,weights=NULL,mc.cores = detectCores(), fixRepro=TRUE, ignore.stdout=FALSE)
{
    if(.Platform$OS.type == "windows")
        mc.cores <- 1
    
    if (iterations == 0)
        iterations <- 1e10
    
    if (is.null(outlines) && is.null(surp))	
        stop("nothing to slide")
    
    if (speed) {
        scale <- FALSE
        CSinit <- TRUE
    } else {
        scale <- TRUE
        CSinit <- FALSE
    }
    n <- dim(dat.array)[3]
    k <- dim(dat.array)[1]
    m <- dim(dat.array)[2]
    
    if (pairedLM[1]!=0 && is.vector(pairedLM))# check if there are only 2 symmetric lms
        pairedLM <- t(as.matrix(pairedLM))
    
### update indexing for after ignored landmarks are removed ###	
    if (!is.null(ignore)) {
        li <- length(ignore)
        lm.old <- c(1:k)[-ignore]
        mat.ptr <- matrix(c(1:(k-li),lm.old),k-li,2)
        ptr <- function(xo)	### define pointer function for indexing
            {
                if (length(which(ignore %in% xo))!= 0)
                    xo <- xo[-which(xo %in% ignore)]
                for (i in 1:(k-li))
                    xo[which(xo==mat.ptr[i,2])] <- mat.ptr[i,1]
                return(xo)
            }
        
        if (!is.null(outlines)) ### update outline indices
            outlines <- lapply(outlines,ptr)
        if (!is.null(surp)) 	### update surface indices
            surp <- ptr(surp)
        
        if (!is.null(SMvector)) ### of fixed/sliding definition
            SMvector <- ptr(SMvector)
        
        if (pairedLM[1]!=0){	### update paired landmarks indices
            count <- 0
            del <- NULL
            for (i in 1:dim(pairedLM)[1]) {	
                if (length(which(ignore %in% pairedLM[i,]))!=0) {
                    count <- count+1
                    del[count] <- i
                }
            }
            pairedLM <- pairedLM[-del,]
            if (is.vector(pairedLM))
                pairedLM <- t(as.matrix(pairedLM))
            
            if (dim(pairedLM)[1]==0) {
                pairedLM <- 0
            } else {
                pairedLM <- apply(pairedLM,2,ptr)
                if (is.vector(pairedLM))
                    pairedLM <- t(as.matrix(pairedLM))
            }
        }
        dat.array <- dat.array[-ignore,,]
        k <- dim(dat.array)[1]
    }
    
    vn.array <- dat.array
    data.orig <- dat.array
    if (deselect)
        fixLM <- SMvector
    else if (length(SMvector) < k)
        fixLM <- c(1:k)[-SMvector]
    else
        fixRepro <- TRUE

    if(length(sur.name)==0) {
        sur.name <- dimnames(dat.array)[[3]]
        sur.name <- paste(sur.path,"/",sur.name,".",sur.type,sep="")
    }
    p1 <- 10^12
    
    ini <- rotonto(dat.array[,,1],dat.array[,,2],signref=FALSE) # create mean between first tow configs to avoid singular BE Matrix
    mshape <- (ini$Y+ini$X)/2
    
    cat(paste("Points will be initially projected onto surfaces","\n","-------------------------------------------","\n"))
    ## parallel function in case meshlist != NULL
    parfunmeshlist <- function(i,data) {
        if (!is.list(data))
            out <- closemeshKD(data[,,i],meshlist[[i]])
        else
            out <- closemeshKD(data[[i]],meshlist[[i]])
        return(out)
    }
    
    if (is.null(meshlist)) {
        for (j in 1:n) {
            
            repro <- projRead(dat.array[,,j], sur.name[j], ignore.stdout=ignore.stdout)
            dat.array[,,j] <- t(repro$vb[1:3,])
            vn.array[,,j] <- t(repro$normals[1:3,])
        }
    } else {
        repro <- mclapply(1:n, parfunmeshlist,dat.array,mc.cores=mc.cores)
        for (j in 1:n) {
            reprotmp <- repro[[j]]         
            dat.array[,,j] <- t(reprotmp$vb[1:3,])
            vn.array[,,j] <- t(reprotmp$normals[1:3,])
        }
    }
    
    if (!fixRepro)# use original positions for fix landmarks
        dat.array[fixLM,,] <- data.orig[fixLM,,]
    
    
    cat(paste("\n","-------------------------------------------","\n"),"Projection finished","\n","-------------------------------------------","\n")
    
    if (initproc==TRUE) { # perform proc fit before sliding
        cat("Inital procrustes fit ...")	
        procini <- ProcGPA(dat.array,scale=scale,CSinit=CSinit)
        mshape <- procini$mshape
    }
    dataslide <- dat.array
    
    if (pairedLM[1]!=0) {# create symmetric mean to get rid of assymetry along outlines/surfaces after first relaxation
        Mir <- diag(c(-1,1,1))
        A <- mshape
        Amir <- mshape%*%Mir
        Amir[c(pairedLM),] <- Amir[c(pairedLM[,2:1]),]
        symproc <- rotonto(A,Amir)
        mshape <- (symproc$X+symproc$Y)/2
    }
    cat(paste("Start sliding...","\n","-------------------------------------------","\n"))
    gc(verbose=F)
    ## calculation for a defined max. number of iterations
    count <- 1
    while (p1>tol && count <= iterations) {
        dataslide_old <- dataslide
        mshape_old <- mshape           
        cat(paste("Iteration",count,sep=" "),"..\n")  # reports which Iteration is calculated
        if (recursive==TRUE)    # slided Semilandmarks are used in next iteration step
            dat.array <- dataslide
        
        L <- CreateL(mshape)
        a.list <- as.list(1:n)
        slido <- function(j)          		
            {
                U <- .calcTang_U_s(dat.array[,,j],vn.array[,,j],SMvector=SMvector,outlines=outlines,surface=surp,deselect=deselect,weights=weights)
                dataslido <- calcGamma(U$Gamma0,L$Lsubk3,U$U,dims=m)$Gamatrix
                return(dataslido)
            }
        a.list <- mclapply(a.list,slido,mc.cores=mc.cores)
        
###projection onto surface
        if (is.null(meshlist)) {
            for (j in 1:n) {
                repro <- projRead(a.list[[j]],sur.name[j], ignore.stdout=ignore.stdout)
                dataslide[,,j] <- t(repro$vb[1:3,])
                vn.array[,,j] <- t(repro$normals[1:3,])
            }
        } else {
            repro <- mclapply(1:n, parfunmeshlist,a.list, mc.cores=mc.cores)
            for (j in 1:n) {
                reprotmp <- repro[[j]]         
                dataslide[,,j] <- t(reprotmp$vb[1:3,])
                vn.array[,,j] <- t(reprotmp$normals[1:3,])
            }
        }
        
        if (!fixRepro)# use original positions for fix landmarks
            dataslide[fixLM,,] <- data.orig[fixLM,,]
        
        cat("estimating sample mean shape...")          	
        proc <- ProcGPA(dataslide,scale=scale,CSinit=CSinit)
        mshape <- proc$mshape
        if (pairedLM[1]!=0) {# create symmetric mean to get rid of assymetry along outline after first relaxation
            Mir <- diag(c(-1,1,1))
            A <- mshape
            Amir <- mshape%*%Mir
            Amir[c(pairedLM),] <- Amir[c(pairedLM[,2:1]),]
            symproc <- rotonto(A,Amir)
            mshape <- (A+Amir)/2
        }     
        p1_old <- p1
        testproc <- rotonto(mshape_old,mshape)			   	
        p1 <- sum(diag(crossprod((testproc$X/cSize(testproc$X))-(testproc$Y/cSize(testproc$Y)))))
        
### check for increasing convergence criterion ###		
        if (inc.check) {
            if (p1 > p1_old) {
                dataslide <- dataslide_old
                cat(paste("Distance between means starts increasing: value is ",p1, ".\n Result from last iteration step will be used. \n"))
                p1 <- 0
            } else {
                cat(paste("squared distance between means:",p1,sep=" "),"\n","-------------------------------------------","\n")
                count <- count+1         
            }
        } else {
            cat(paste("squared distance between means:",p1,sep=" "),"\n","-------------------------------------------","\n")
            count <- count+1         
        }
        gc(verbose = FALSE)
    }
    gc(verbose = FALSE)
    return(list(dataslide=dataslide,vn.array=vn.array))
}
