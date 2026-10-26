# R runner for the contents replication pack: an independent implementation of python/run_all.py.
#
# The data layer is the same SQL (sql/*.sql) run through DuckDB, so the cleaned data is identical.
# The one-way analysis, the metric and the GLMs are re-implemented here (penalised IRLS in base R),
# and reconcile/reconcile.py compares the two sets of outputs.
#
# Usage (from the pack root):  Rscript R/run_all.R

suppressPackageStartupMessages({
  library(DBI)
  library(duckdb)
  library(jsonlite)
  library(digest)
})

RUNNER <- "r"
pack_root <- normalizePath(".")
stopifnot(file.exists(file.path(pack_root, "config.json")))

# ---- plumbing (mirrors python/common.py) -------------------------------------------------------

load_config <- function() {
  cfg <- fromJSON(file.path(pack_root, "config.json"), simplifyVector = TRUE)
  if (!isTRUE(cfg$decisions$D06_missing_as_level$apply))
    stop("D06 apply=false is not implemented: sql/08 always treats missing as a level.")
  for (key in c("D04_exclude_contacts", "D05_exclude_written_premium")) {
    d <- cfg$decisions[[key]]
    if (isTRUE(d$apply) && d$excluded_feature %in% cfg$model$factors)
      stop(sprintf("%s is applied but %s is in model.factors.", key, d$excluded_feature))
  }
  cfg
}

sha256_file <- function(path) digest(file = path, algo = "sha256")

verify_inputs <- function(cfg) {
  lines <- readLines(file.path(pack_root, cfg$paths$expected_inputs))
  lines <- lines[nzchar(trimws(lines)) & !startsWith(lines, "#")]
  parts <- strsplit(lines, "\\s+")
  expected <- setNames(vapply(parts, `[`, "", 1), vapply(parts, `[`, "", 2))
  actual <- list()
  for (t in cfg$source$tables) {
    name <- paste0(t, ".parquet")
    actual[[name]] <- sha256_file(file.path(pack_root, cfg$paths$data_dir, name))
    if (!identical(unname(expected[name]), actual[[name]]))
      stop(sprintf("Input hash mismatch for %s", name))
  }
  actual
}

sql_literal <- function(v) {
  if (is.logical(v)) return(if (v) "true" else "false")
  if (is.numeric(v)) return(format(v, scientific = FALSE))
  if (grepl("^\\d{4}-\\d{2}-\\d{2}$", v)) return(sprintf("DATE '%s'", v))
  sprintf("'%s'", gsub("'", "''", v))
}

open_duckdb <- function(cfg) {
  con <- dbConnect(duckdb(shared_home = FALSE))
  dbExecute(con, sprintf("SET threads = %d", as.integer(cfg$duckdb_threads)))
  for (t in cfg$source$tables) {
    path <- file.path(pack_root, cfg$paths$data_dir, paste0(t, ".parquet"))
    dbExecute(con, sprintf("CREATE OR REPLACE VIEW %s AS SELECT * FROM read_parquet('%s')", t, path))
  }
  vars <- cfg$sql_variables
  for (d in cfg$decisions) if (!is.null(d$sql_variable)) vars[[d$sql_variable]] <- isTRUE(d$apply)
  for (n in names(vars)) dbExecute(con, sprintf("SET VARIABLE %s = %s", n, sql_literal(vars[[n]])))
  con
}

run_sql_files <- function(con, cfg) {
  hashes <- list()
  for (rel in cfg$sql_files) {
    path <- file.path(pack_root, rel)
    hashes[[rel]] <- sha256_file(path)
    dbExecute(con, paste(readLines(path), collapse = "\n"))
  }
  hashes
}

export_sql_evidence <- function(con, out_dir) {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  names <- dbGetQuery(con, "SELECT table_name FROM duckdb_tables() WHERE table_name LIKE 'ev%' ORDER BY 1")$table_name
  vapply(names, function(n) {
    p <- file.path(out_dir, paste0(n, ".csv"))
    dbExecute(con, sprintf("COPY (SELECT * FROM %s ORDER BY ALL) TO '%s' (HEADER, DELIMITER ',')", n, p))
    p
  }, "")
}

write_evidence <- function(df, path) {
  num <- vapply(df, is.numeric, TRUE)
  df[num] <- lapply(df[num], round, 6)
  write.csv(df, path, row.names = FALSE, na = "")
  path
}

git_commit <- function() {
  out <- tryCatch(suppressWarnings(system2("git", c("-C", shQuote(pack_root), "rev-parse", "HEAD"),
                                           stdout = TRUE, stderr = FALSE)), error = function(e) character())
  if (length(out) == 1 && nchar(out) == 40) out else NA
}

# ---- metric, design matrix, one-way (mirrors python/models.py) ----------------------------------

tweedie_deviance <- function(y, mu, p) {
  2 * (y^(2 - p) / ((1 - p) * (2 - p)) - y * mu^(1 - p) / (1 - p) + mu^(2 - p) / (2 - p))
}

d2_tweedie <- function(y, pred, p) {
  1 - sum(tweedie_deviance(y, pred, p)) / sum(tweedie_deviance(y, rep(mean(y), length(y)), p))
}

# Levels sorted in C-locale order (radix), matching Python's sorted(); the first is the reference.
factor_levels <- function(df, factors) {
  setNames(lapply(factors, function(f) sort(unique(df[[f]]), method = "radix")), factors)
}

design_matrix <- function(df, levels, numeric) {
  cols <- list()
  for (f in names(levels)) {
    x <- df[[f]]
    for (lv in levels[[f]][-1]) cols[[paste0(f, "=", lv)]] <- as.numeric(x == lv)
  }
  for (n in numeric) cols[[n]] <- as.numeric(df[[n]])
  m <- do.call(cbind, cols)
  colnames(m) <- names(cols)
  m
}

agg_by <- function(df, f) {
  e <- tapply(df$exposure, df[[f]], sum)
  c <- tapply(df$claim_cost, df[[f]], sum)
  n <- tapply(df$n_claims, df[[f]], sum)
  list(e = e, c = c, n = n)
}

oneway_table <- function(df, f) {
  a <- agg_by(df, f)
  overall <- sum(df$claim_cost) / sum(df$exposure)
  lv <- names(a$e)
  out <- data.frame(factor = f, level = lv, exposure = as.numeric(a$e),
                    exposure_pct = 100 * as.numeric(a$e) / sum(a$e), n_claims = as.numeric(a$n),
                    stringsAsFactors = FALSE)
  out$freq_per_1000 <- 1000 * out$n_claims / out$exposure
  out$severity <- ifelse(out$n_claims > 0, as.numeric(a$c) / out$n_claims, NA)
  out$loss_cost <- as.numeric(a$c) / out$exposure
  out$relativity <- out$loss_cost / overall
  for (part in list(list("rel_2021_22", df[df$yr <= 2022, ]), list("rel_2023_24", df[df$yr >= 2023, ]))) {
    d <- part[[2]]
    p <- agg_by(d, f)
    rate <- (p$c / p$e) / (sum(d$claim_cost) / sum(d$exposure))
    out[[part[[1]]]] <- as.numeric(rate[lv])
  }
  out
}

oneway_d2_gain <- function(fit, val, f, p) {
  base <- sum(fit$claim_cost) / sum(fit$exposure)
  a <- agg_by(fit, f)
  rate <- as.numeric((a$c / a$e)[val[[f]]])
  rate[is.na(rate)] <- base
  d2_tweedie(val$claim_cost, rate * val$exposure, p) - d2_tweedie(val$claim_cost, base * val$exposure, p)
}

# ---- GLMs ----------------------------------------------------------------------------------------
# Penalised IRLS (Fisher scoring) written out in base R, independent of scikit-learn. It minimises
#   (1 / (2 sum w)) sum w d_p(y, mu) + alpha / 2 ||b without intercept||^2,   log link,
# the same objective as scikit-learn's GLMs, where d_p is the Tweedie unit deviance with power p
# (1 = Poisson, 1.5 = Tweedie, 2 = Gamma). The objective is strictly convex, so both reach the same
# optimum. Fisher weight for a log link: mu^(2-p); working response: eta + (y - mu) / mu.
# Exposure enters as in Python: y / exposure with weight exposure^(2-p).
# (glmnet was used first and was too slow to converge at this precision: audit A11.)

fit_glm <- function(X, y, w, p, alpha, tol = 1e-10, max_iter = 500) {
  X1 <- cbind(`(intercept)` = 1, X)
  w <- w / sum(w)
  pen <- c(0, rep(alpha, ncol(X)))
  beta <- c(log(sum(w * y)), rep(0, ncol(X)))
  for (it in seq_len(max_iter)) {
    eta <- drop(X1 %*% beta)
    mu <- exp(eta)
    W <- w * mu^(2 - p)
    z <- eta + (y - mu) / mu
    A <- crossprod(X1, X1 * W)
    diag(A) <- diag(A) + pen
    new <- drop(solve(A, crossprod(X1, W * z)))
    step <- max(abs(new - beta))
    beta <- new
    if (step < tol) break
  }
  if (step >= tol) warning(sprintf("IRLS did not converge: last step %.2e", step))
  structure(list(coef = setNames(beta, colnames(X1)), iterations = it), class = "irls_fit")
}

coefs_of <- function(m) m$coef

predict_rate <- function(m, X) exp(drop(cbind(1, X) %*% m$coef))

# ---- main ----------------------------------------------------------------------------------------

main <- function() {
  t0 <- Sys.time()
  cfg <- load_config()
  rec <- list(run_id = paste(sample(c(0:9, letters[1:6]), 32, replace = TRUE), collapse = ""),
              runner = RUNNER, started_utc = format(Sys.time(), "%Y-%m-%dT%H:%M:%S+00:00", tz = "UTC"),
              platform = Sys.info()[["sysname"]], git_commit = git_commit(),
              config_sha256 = sha256_file(file.path(pack_root, "config.json")),
              code_sha256 = list(`R/run_all.R` = sha256_file(file.path(pack_root, "R", "run_all.R"))),
              decisions = lapply(cfg$decisions, function(d) d[c("apply", "signed_off", "signed_off_by")]))
  out_dir <- file.path(pack_root, cfg$paths$evidence_dir, RUNNER)
  status <- "failed"
  on.exit({
    rec$status <- status
    rec$versions <- list(R = paste(R.version$major, R.version$minor, sep = "."),
                         duckdb = as.character(packageVersion("duckdb")),
                         solver = "penalised IRLS (base R)")
    rec$finished_utc <- format(Sys.time(), "%Y-%m-%dT%H:%M:%S+00:00", tz = "UTC")
    rec$elapsed_s <- round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1)
    cat(toJSON(rec, auto_unbox = TRUE, null = "null", na = "null", digits = NA), "\n",
        file = file.path(pack_root, cfg$paths$run_log), append = TRUE, sep = "")
  })

  rec$inputs_sha256 <- verify_inputs(cfg)
  con <- open_duckdb(cfg)
  rec$sql_sha256 <- run_sql_files(con, cfg)
  outputs <- as.character(export_sql_evidence(con, out_dir))
  feats <- dbGetQuery(con, "SELECT * FROM contents_features WHERE src = 'train'")
  dbDisconnect(con, shutdown = TRUE)

  mc <- cfg$model
  p <- mc$tweedie_power
  lambda <- mc$alpha
  fit <- feats[feats$yr <= mc$fit_year_max, ]
  val <- feats[feats$yr == mc$score_year, ]

  # A07: one-way tables and out-of-time scores.
  tables <- list(); scores <- list()
  for (group in names(cfg$oneway$factors)) for (f in cfg$oneway$factors[[group]]) {
    t <- oneway_table(feats, f)
    tables[[f]] <- t
    scores[[f]] <- data.frame(group = group, factor = f, n_levels = nrow(t),
                              d2_gain = oneway_d2_gain(fit, val, f, p),
                              rel_min = min(t$relativity), rel_max = max(t$relativity))
  }
  tables <- do.call(rbind, tables)
  tables <- tables[order(tables$factor, tables$level, method = "radix"), ]
  scores <- do.call(rbind, scores)
  scores <- scores[order(scores$factor, method = "radix"), ]
  outputs <- c(outputs, write_evidence(tables, file.path(out_dir, "ev_m1_oneway_tables.csv")),
               write_evidence(scores, file.path(out_dir, "ev_m1_oneway_scores.csv")))

  # A09: GLM structure comparison.
  levels <- factor_levels(feats, mc$factors)
  X_fit <- design_matrix(fit, levels, mc$numeric)
  X_val <- design_matrix(val, levels, mc$numeric)
  e_fit <- fit$exposure; e_val <- val$exposure; y_val <- val$claim_cost
  coef_rows <- list(); preds <- list(); totals <- list()

  tweedie_component <- function(name, target) {
    m <- fit_glm(X_fit, fit[[target]] / e_fit, e_fit^(2 - p), p, lambda)
    b <- coefs_of(m)
    coef_rows[[name]] <<- data.frame(component = name, param = names(b), coef = as.numeric(b))
    pred <- predict_rate(m, X_val) * e_val
    totals[[name]] <<- data.frame(component = name, pred_total = sum(pred))
    pred
  }

  base <- sum(fit$claim_cost) / sum(fit$exposure)
  preds[["exposure only"]] <- base * e_val
  preds[["A single Tweedie"]] <- tweedie_component("A total", "claim_cost")
  preds[["B per-peril Tweedie"]] <- Reduce(`+`, lapply(names(mc$perils), function(k)
    tweedie_component(paste("B", k), mc$perils[[k]][1])))
  preds[["C per-size-band Tweedie"]] <- Reduce(`+`, lapply(mc$size_bands, function(b)
    tweedie_component(paste("C", b), b)))
  d_parts <- lapply(names(mc$perils), function(k) {
    cost_col <- mc$perils[[k]][1]; n_col <- mc$perils[[k]][2]
    n <- fit[[n_col]]
    fm <- fit_glm(X_fit, n / e_fit, e_fit, 1, lambda)
    has <- n > 0
    sm <- fit_glm(X_fit[has, ], fit[[cost_col]][has] / n[has], n[has], 2, lambda)
    for (pair in list(list(paste("D freq", k), fm), list(paste("D sev", k), sm))) {
      b <- coefs_of(pair[[2]])
      coef_rows[[pair[[1]]]] <<- data.frame(component = pair[[1]], param = names(b), coef = as.numeric(b))
    }
    pred <- predict_rate(fm, X_val) * e_val * predict_rate(sm, X_val)
    totals[[paste("D", k)]] <<- data.frame(component = paste("D", k), pred_total = sum(pred))
    pred
  })
  preds[["D per-peril freq x sev"]] <- Reduce(`+`, d_parts)
  totals <- do.call(rbind, totals)
  outputs <- c(outputs, write_evidence(totals[order(totals$component, method = "radix"), ],
                                       file.path(out_dir, "ev_m2_component_totals.csv")))

  split <- data.frame(model = names(preds),
                      d2 = vapply(preds, function(v) d2_tweedie(y_val, v, p), 0),
                      pred_over_actual = vapply(preds, function(v) sum(v) / sum(y_val), 0))
  coefs <- do.call(rbind, coef_rows)
  rel <- coefs[startsWith(coefs$component, "C ") &
                 sub("=.*$", "", coefs$param) %in% mc$relativity_params, ]
  rel$relativity <- exp(rel$coef)
  rel <- reshape(rel[, c("param", "component", "relativity")], idvar = "param", timevar = "component",
                 direction = "wide")
  names(rel) <- sub("^relativity\\.", "", names(rel))
  rel <- rel[order(rel$param, method = "radix"), ]
  design <- data.frame(item = c("n_fit_rows", "n_score_rows", "n_params_incl_intercept",
                                paste0("reference_level:", names(levels))),
                       value = c(nrow(fit), nrow(val), ncol(X_fit) + 1,
                                 vapply(levels, `[`, "", 1)))
  outputs <- c(outputs,
               write_evidence(split, file.path(out_dir, "ev_m2_split_test.csv")),
               write_evidence(coefs, file.path(out_dir, "ev_m2_coefficients.csv")),
               write_evidence(rel, file.path(out_dir, "ev_m2_band_relativities.csv")),
               write_evidence(design, file.path(out_dir, "ev_m0_design_summary.csv")))

  rec$outputs_sha256 <- setNames(lapply(outputs, sha256_file),
                                 sub(paste0("^", pack_root, "/"), "", normalizePath(outputs)))
  rec$metrics <- list(split_test_d2 = as.list(setNames(round(split$d2, 6), split$model)))
  print(format(split, digits = 4), row.names = FALSE)
  status <- "success"
}

main()
