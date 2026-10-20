# R 4.5. Packages and versions used for the reference run on 2026-10-20.
# Install with:  Rscript R/packages.R
pkgs <- c(DBI = "1.2.3", duckdb = "1.5.6", jsonlite = "2.0.0", digest = "0.6.37", testthat = "3.3.2")
missing <- names(pkgs)[!vapply(names(pkgs), requireNamespace, TRUE, quietly = TRUE)]
if (length(missing)) install.packages(missing, repos = "https://cloud.r-project.org")
for (p in names(pkgs)) {
  v <- as.character(packageVersion(p))
  if (v != pkgs[[p]]) message(sprintf("Note: %s is %s; the reference run used %s.", p, v, pkgs[[p]]))
}
