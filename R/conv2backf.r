#' invert faces' orientation of triangular mesh
#' 
#' inverts faces' orientation of triangular mesh and recomputes vertex normals
#' 
#' 
#' @param mesh triangular mesh of class \code{mesh3d}
#' @return returns resulting mesh
#' @author Stefan Schlager
#' @seealso \code{\link{updateNormals}}
#' @keywords ~kwd1 ~kwd2
#' @examples
#' 
#' require(rgl)
#' data(nose)
#' \dontrun{
#' shade3d(shortnose.mesh,col=3)
#' }
#' noseinvert <- conv2backf(shortnose.mesh)
#' ## show normals
#' \dontrun{
#' plotNormals(noseinvert,long=0.01)
#' }
#' @export
conv2backf <- function(mesh)
{ 	
	mesh$it <- mesh$it[c(3,2,1),]
        mesh <- updateNormals(mesh)
  	return(mesh)
}
