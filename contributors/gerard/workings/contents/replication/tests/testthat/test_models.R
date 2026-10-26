# Unit tests for the R implementation's metric and design matrix.
# Run from the pack root:  Rscript -e 'testthat::test_dir("tests/testthat")'
# The functions are sourced from R/run_all.R without running main().

src <- readLines(file.path("..", "..", "R", "run_all.R"))
src <- src[seq_len(max(grep("^# ---- main", src)) - 1)]
eval(parse(text = src[!grepl("^RUNNER|^pack_root|^stopifnot", src)]))

# Same fixed example as tests/test_models.py.
Y <- c(0, 0, 120, 0, 3500, 0, 800, 0)
PRED <- c(50, 40, 90, 60, 400, 30, 300, 45)

test_that("D2 matches the value scikit-learn gives for the shared example", {
  expect_equal(d2_tweedie(Y, PRED, 1.5), 0.34644972, tolerance = 1e-8)
})

test_that("deviance scaling justifies the exposure weights", {
  y <- c(0, 2, 7.5); mu <- c(1, 1.5, 4); k <- 0.37; p <- 1.5
  expect_equal(tweedie_deviance(k * y, k * mu, p), k^(2 - p) * tweedie_deviance(y, mu, p))
})

test_that("levels sort in C-locale order, matching Python's sorted()", {
  df <- data.frame(f = c("b", "MISSING", "a", "b"), z = c(0.1, 0.2, 0.3, 0.4))
  lv <- factor_levels(df, "f")
  expect_identical(lv$f, c("MISSING", "a", "b"))
  X <- design_matrix(df, lv, "z")
  expect_identical(colnames(X), c("f=a", "f=b", "z"))
  expect_identical(as.numeric(X[, 1]), c(0, 0, 1, 0))
})

test_that("exposure-weighted Tweedie fit recovers the simulated rate", {
  set.seed(0)
  n <- 20000
  x <- matrix(as.numeric(sample(0:1, n, replace = TRUE)), ncol = 1, dimnames = list(NULL, "x"))
  expo <- runif(n, 0.1, 1)
  y <- rpois(n, exp(1 + 0.5 * x[, 1]) * expo) * 10
  m <- fit_glm(x, y / expo, expo^0.5, 1.5, 0)
  b <- coefs_of(m)
  expect_equal(unname(b["(intercept)"]), 1 + log(10), tolerance = 0.05)
  expect_equal(unname(b["x"]), 0.5, tolerance = 0.05)
})
