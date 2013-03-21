Semislide<-function(dataframe,SMvector,outlines,tol=1e-05,deselect=FALSE,recursive=TRUE,iterations=0,initproc=FALSE,pairedLM=NULL)

{
  n<-dim(dataframe)[3]
  k<-dim(dataframe)[1]
  m<-dim(dataframe)[2]
  
  p1<-10^12
  if (iterations == 0)
    {
      iterations <- 1e10
    }
  
  ini<-rotonto(dataframe[,,1],dataframe[,,2],reflection=T)
  mshape<-(ini$X+ini$Y)/2
  
  if(initproc==TRUE) # perform proc fit before sliding
    {procini<-ProcGPA(dataframe,scale=TRUE)
     mshape<-procini$mshape
     
   }
  dataslide<-dataframe
  
  if (!is.null(pairedLM))# create symmetric mean to get rid of assymetry along outline after first relaxation
    {
      Mir<-diag(c(-1,1,1))
      A<-mshape
      Amir<-mshape%*%Mir
      Amir[c(pairedLM),]<-Amir[c(pairedLM[,2:1]),]
      symproc<-rotonto(A,Amir)
      mshape<-(symproc$X+symproc$Y)/2
    }
  
  
  count<-1
  while (p1>tol && count <= iterations)
    {
      dataslide_old<-dataslide
      mshape_old<-mshape
      cat(paste("Iteration",count,sep=" "),"..\n")  # reports which Iteration is calculated  
      
      if (recursive==TRUE)      # slided Semilandmarks are used in next iteration step
        { dataframe<-dataslide
        }
      if (m==3)
        {L<-CreateL(mshape)
       }
      else 
        {L<-CreateL2D(mshape)
       } 
      for (j in 1:n)
        {U<-calcTang_U(dataframe[,,j],SMvector=SMvector,outlines=outlines,deselect=deselect)
         dataslide[,,j]<-calcGamma(U$Gamma0,L$Lsubk3,U$U,dims=m)$Gamatrix
       }
      proc<-ProcGPA(dataslide,scale=TRUE)
      mshape<-proc$mshape
      p1_old<-p1   
      p1<-sum(diag(crossprod((mshape_old/cSize(mshape_old))-(mshape/cSize(mshape)))))
                                        #p1<-sum(diag(crossprod(mshape_old-mshape)))/k
      
      ## check for increasing convergence criterion ###		
      if (p1 > p1_old)
        {
          dataslide<-dataslide_old
          cat(paste("Distance between means starts increasing: value is ",p1, ".\n Result from last iteration step will be used. \n"))
          p1<-0
        } 
      else
        {
          cat(paste("squared distance between means:",p1,sep=" "),"\n","-------------------------------------------","\n")
          count<-count+1 
        }          		
    }
  
  return(dataslide)
}
