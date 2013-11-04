checkLM <- function(dat.array, path=NULL, prefix="", suffix=".ply", col="white", pt.size=NULL, alpha=0.7, begin=1, render=c("w","s"), point=c("s","p"), add=FALSE, Rdata=FALSE, atlas=NULL, text.lm=FALSE)
    {
        k <- NULL
        marked <- NULL
        j <- 1
        if (!Rdata)
            load <- file2mesh
        outid <- NULL
        point <- point[1]
        ## set point/sphere sizes
        radius <- pt.size
        if (is.null(radius)) {
            if (point == "s")
                radius <- (cSize(dat.array[,,1])/sqrt(nrow(dat.array[,,1])))*(1/30)
            else
                radius <- 10
        }
        size <- radius
        render <- render[1]
        arr <- FALSE
        point <- point[1]
        if (point == "s") {
            rendpoint <- spheres3d
        } else if (point == "p") {
            rendpoint <- points3d
        } else {
            stop("argument \"point\" must be \"s\" for spheres or \"p\" for points")
        }
        dimDat <- dim(dat.array)
        if (length(dimDat) == 3) {
            n <- dim(dat.array)[3]
            name <- dimnames(dat.array)[[3]]
            arr <- TRUE
        } else if (is.list(dat.array)) {
            n <- length(dat.array)
            name <- names(dat.array)
        } else {
            stop("data must be 3-dimensional array or a list")
        }
        i <- begin
        if (render=="w") {
            rend <- wire3d
        } else {
            rend <- shade3d
        }
        if (!add)
            open3d()
        if (!is.null(atlas)) {
            k <- dim(atlas$landmarks)[1]
            #k1 <- dim(atlas$patch)[1]
        }
        while (i <= n) {
            rgl.bringtotop()
            tmp.name <- paste(path,prefix,name[i],suffix,sep="")
            if (arr)
                landmarks <- dat.array[,,i]
            else
                landmarks <- dat.array[[i]]
            if (is.null(atlas)) { 
                outid <- rendpoint(landmarks,radius=radius, size=size)
                if (text.lm)
                    outid <- c(outid, text3d(landmarks, texts=paste(1:dim(landmarks)[1], sep=""), cex=1, adj=c(1,1.5)))
                             
                if (!is.null(path)) {
                    if (!Rdata) {
                        tmpmesh <- file2mesh(tmp.name)
                    } else {
                        input <- load(tmp.name)
                        tmp.name <- gsub(path,"",tmp.name)
                        tmpmesh <- get(input)
                    }
                    
                    outid <- c(outid,rend(tmpmesh,col=col,alpha=alpha))
                    rm(tmpmesh)
                    if (Rdata)
                        rm(list=input)
                    gc()
                }
            } else {
                atlas.tmp <- atlas
                atlas.tmp$mesh <- NULL
                atlas.tmp$landmarks <- landmarks[1:k,]
                atlas.tmp$patch <- landmarks[-c(1:k),]
                
                if (!is.null(path)) {
                    if (!Rdata) {
                        atlas.tmp$mesh <- file2mesh(tmp.name)
                    } else {
                        input <- load(tmp.name)
                        tmp.name <- gsub(path,"",tmp.name)
                        atlas.tmp$mesh <- get(input)
                    }
                }
                outid <- plotAtlas(atlas.tmp, add=FALSE, alpha=alpha, pt.size=radius, render=render, point=point, meshcol=col, legend=FALSE)
                
            }
                
                
                answer <- readline(paste("viewing #",i,"(return=next | m=mark current | s=stop viewing)\n"))
                if (answer == "m") {
                    marked[j] <- i
                    j <- j+1
                } else if (answer == "s") {
                    i <- n+1
                } else
                    i <- i+1
                rgl.pop(id=outid)
            }
            invisible(marked)
        }
