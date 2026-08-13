# Developer-only numerical QC for the independently implemented Julia
# NNloglik metalearner. SuperLearner is not a Julia runtime dependency.
suppressPackageStartupMessages(library(SuperLearner))

Y <- rep(c(0, 1), 10)
Z <- cbind(
  learner_a = c(
    0.08, 0.82, 0.18, 0.76, 0.88, 0.68, 0.25, 0.84, 0.12, 0.73,
    0.30, 0.20, 0.22, 0.78, 0.40, 0.65, 0.75, 0.80, 0.35, 0.70
  ),
  learner_b = c(
    0.25, 0.65, 0.35, 0.62, 0.58, 0.55, 0.45, 0.60, 0.28, 0.64,
    0.48, 0.42, 0.38, 0.67, 0.47, 0.58, 0.55, 0.70, 0.43, 0.61
  ),
  learner_c = c(
    0.15, 0.72, 0.55, 0.66, 0.65, 0.40, 0.20, 0.76, 0.31, 0.52,
    0.62, 0.35, 0.41, 0.59, 0.33, 0.75, 0.45, 0.57, 0.29, 0.54
  )
)
obsWeights <- c(
  1, 2, 1, 1.5, 3, 1, 2, 1, 0.5, 2,
  1, 2.5, 1, 1, 2, 1.5, 1, 2, 1, 3
)
trim <- 1e-5

method <- method.NNloglik()
fit <- method$computeCoef(
  Z = Z,
  Y = Y,
  libraryNames = colnames(Z),
  verbose = FALSE,
  obsWeights = obsWeights,
  control = list(trimLogit = trim),
  errorsInLibrary = NULL
)
predictions <- drop(method$computePred(
  predY = Z,
  coef = fit$coef,
  control = list(trimLogit = trim)
))
log_loss <- sum(obsWeights * (
  -Y * log(predictions) - (1 - Y) * log1p(-predictions)
)) / sum(obsWeights)

zero_Z <- cbind(learner_a = rep(0.8, 6), learner_b = rep(0.7, 6))
zero_Y <- rep(0, 6)
zero_fit <- method$computeCoef(
  Z = zero_Z,
  Y = zero_Y,
  libraryNames = colnames(zero_Z),
  verbose = FALSE,
  obsWeights = rep(1, length(zero_Y)),
  control = list(trimLogit = trim),
  errorsInLibrary = NULL
)
zero_predictions <- drop(method$computePred(
  predY = zero_Z,
  coef = zero_fit$coef,
  control = list(trimLogit = trim)
))

cat("R version:", R.version.string, "\n")
cat("SuperLearner version:", as.character(packageVersion("SuperLearner")), "\n")
cat("platform:", R.version$platform, "\n")
cat("trim:", format(trim, scientific = TRUE), "\n")
cat("raw coefficients:\n")
dput(unname(fit$optimizer$par))
cat("normalised coefficients:\n")
dput(unname(fit$coef))
cat("ensemble predictions:\n")
dput(unname(predictions))
cat("weighted mean Bernoulli log loss:\n")
dput(unname(log_loss))
cat("all-zero fixture coefficients:\n")
dput(unname(zero_fit$coef))
cat("all-zero fixture predictions:\n")
dput(unname(zero_predictions))
cat("session information:\n")
print(sessionInfo())
