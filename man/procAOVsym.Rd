\name{procAOVsym}
\alias{procAOVsym}
\title{
  Procrustes ANOVA for structures with object symmetry
}
\description{
  Procrustes ANOVA for structures with object symmetry, currently only
  supporting the factors 'specimen', 'side' and the interaction term.

}
\usage{
procAOVsym(symproc, indnames = NULL)
}
\arguments{
  \item{symproc}{object returned by \code{\link{procSym}}, where
    \code{pairedLM} is specified
}
  \item{indnames}{vector containing specimen identifiers. Only
    necessary, if data does not contain dimnames containing identifiers
}
}
\details{performs a Procrustes ANOVA for configurations with object
  symmetry (as described in Klingenberg et al. 2002). 
}
\value{returns a dataframe containing Sums of Squares for each factor.
  
}
\references{Klingenberg CP, Barluenga M, Meyer A. 2002. Shape analysis of symmetric structures: quantifying variation among individuals and asymmetry. Evolution 56:1909-20.
}
\author{
Stefan Schlager}
\note{In future releases the implementation of support for bilateral symmetry and
  more factors is intended.

}


\seealso{
\code{\link{procSym}}
}
\examples{
data(boneData)
left <- c(4,6,8)
## determine corresponding Landmarks on the right side:
# important: keep same order
right <- c(3,5,7)
pairedLM <- cbind(left,right)
symproc <- procSym(boneLM, pairedLM=pairedLM)
procAOVsym(symproc)
}
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }