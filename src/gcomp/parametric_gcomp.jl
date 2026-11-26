"""Formula-based parametric regression standardisation.

GLMs and supported mixed models use the same target selection, intervention,
standardisation, and contrast path. Their prediction and covariance details
remain in their respective backends. This is separate from the numeric-treatment
matrix path used by cross-fitted `run_gcomp`.
"""

include("parametric/fit.jl")
include("parametric/standardise.jl")
include("parametric/contrasts.jl")
include("parametric/bootstrap.jl")
