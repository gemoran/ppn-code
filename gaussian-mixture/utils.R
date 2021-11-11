library(MASS)
library(LaplacesDemon)
library(ggplot2)

get_empirical_dists = function(x, y, bins = 50) {
  min = min(x, y)
  max = max(x, y)
  
  p = numeric(bins)
  q = numeric(bins)
  
  cuts = seq(min, max, length.out = bins + 1)
  
  for (i in 1:bins) {
    if (i == 1) {
      p[i] = sum(x <= cuts[i])
      q[i] = sum(y <= cuts[i])
    } else {
      if (i < bins) {
        p[i] = sum((x > cuts[i-1]) & (x <= cuts[i]))
        q[i] = sum((y > cuts[i-1]) & (y <= cuts[i]))
      } else {
        if (i == bins) {
          p[i] = sum(x > cuts[i])
          q[i] = sum(y > cuts[i])
        }
      }
    }
  }
  
  rbind(p, q) / length(x)
  
}

GeomSplitViolin <- ggproto("GeomSplitViolin", GeomViolin, 
                           draw_group = function(self, data, ..., draw_quantiles = NULL) {
                             data <- transform(data, xminv = x - violinwidth * (x - xmin), xmaxv = x + violinwidth * (xmax - x))
                             grp <- data[1, "group"]
                             newdata <- plyr::arrange(transform(data, x = if (grp %% 2 == 1) xminv else xmaxv), if (grp %% 2 == 1) y else -y)
                             newdata <- rbind(newdata[1, ], newdata, newdata[nrow(newdata), ], newdata[1, ])
                             newdata[c(1, nrow(newdata) - 1, nrow(newdata)), "x"] <- round(newdata[1, "x"])
                             
                             if (length(draw_quantiles) > 0 & !scales::zero_range(range(data$y))) {
                               stopifnot(all(draw_quantiles >= 0), all(draw_quantiles <=
                                                                         1))
                               quantiles <- ggplot2:::create_quantile_segment_frame(data, draw_quantiles)
                               aesthetics <- data[rep(1, nrow(quantiles)), setdiff(names(data), c("x", "y")), drop = FALSE]
                               aesthetics$alpha <- rep(1, nrow(quantiles))
                               both <- cbind(quantiles, aesthetics)
                               quantile_grob <- GeomPath$draw_panel(both, ...)
                               ggplot2:::ggname("geom_split_violin", grid::grobTree(GeomPolygon$draw_panel(newdata, ...), quantile_grob))
                             }
                             else {
                               ggplot2:::ggname("geom_split_violin", GeomPolygon$draw_panel(newdata, ...))
                             }
                           })

geom_split_violin <- function(mapping = NULL, data = NULL, stat = "ydensity", position = "identity", ..., 
                              draw_quantiles = NULL, trim = TRUE, scale = "area", na.rm = FALSE, 
                              show.legend = NA, inherit.aes = TRUE) {
  layer(data = data, mapping = mapping, stat = stat, geom = GeomSplitViolin, 
        position = position, show.legend = show.legend, inherit.aes = inherit.aes, 
        params = list(trim = trim, scale = scale, draw_quantiles = draw_quantiles, na.rm = na.rm, ...))
}

# functions -------------------------

# gibbs sampling

gamma_draw <- function(Y, mus, sigmas) {
  K <- ncol(mus)
  n <- nrow(Y)
  p <- rep(1/K, K)
  dist <- matrix(0, nrow = K, ncol = n)
  for (k in 1:K) {
    dist[k, ] <- exp(-apply((t(Y) - mus[, k])/sqrt(sigmas[,k]), 2, function(x) sum(x^2)/2)) / sqrt((prod(sigmas[, k])))
  }
  gamma <- dist * p
  gamma[, apply(gamma, 2, sum) == 0] <- 0.5
  if (K > 1) {
    gamma <- t(apply(gamma, 2, function(x) x / sum(x)))
  } else {
    gamma <- matrix(apply(gamma, 2, function(x) x / sum(x)), nrow = n, ncol = K)
  }
  gamma <- apply(gamma, 1, function(x) sample(1:K, 1, replace = T, prob = x))
  gamma <- cbind(1:n, gamma)
  g <- matrix(0, nrow = n, ncol = K)
  g[gamma] <- 1
  g
}

mu_draw <- function(Y, gamma, sigmas) {
  K <- ncol(gamma)
  d <- ncol(Y)
  nk <- apply(gamma, 2, sum)
  mus <- matrix(0, nrow = d, ncol = K)
  
  for (k in 1:K) {
    yk_mean <- apply(Y * gamma[, k], 2, sum)
    mu_var = solve(diag(1/sigmas[, k]) * nk[k] + diag(d)/mu_sd^2)
    mu_mean = mu_var %*% yk_mean / sigmas[, k]
    mus[, k] <- mvrnorm(1, mu = mu_mean, Sigma = mu_var)
  }
  mus
}

sigma_draw <- function(Y, gamma, mus) {
  K <- ncol(gamma) 
  d <- ncol(Y)
  sigmas <- matrix(0, nrow = d, ncol = K)
  for (k in 1:K) {
    temp <- t((t(Y) - mus[, k])^2)
    temp <- temp * gamma[, k]
    temp <- apply(temp, 2, sum)
    alpha_post <- sum(gamma[, k]) / 2 + sigma_alpha
    beta_post <- temp / 2 + sigma_beta
    sigmas[, k] <- mapply(rinvgamma, rep(1, d), rep(alpha_post, d), beta_post)
  }
  sigmas
}

gibbs <- function(Y, K, burn = 100, npost = 500,
                  mus = matrix(rnorm(d * K), nrow = d, ncol = K),
                  sigmas = matrix(1, nrow = d, ncol = K)) {
  
  n <- nrow(Y)
  
  # initialize mus, p
  p <- rep(1/K, K)
  mus <- matrix(rnorm(d * K), nrow = d, ncol = K)
  sigmas <- matrix(1, nrow = d, ncol = K)
  
  gamma_all <- array(0, dim = c(n, K, npost))
  mus_all <- array(0, dim = c(d, K, npost))
  sigmas_all <- array(0, dim = c(d, K, npost))
  
  for (t in 1:(burn + npost)) {
    gamma = gamma_draw(Y, mus, sigmas)
    mus = mu_draw(Y, gamma, sigmas)
    sigmas = sigma_draw(Y, gamma, mus)
    
    if (t > burn) {
      gamma_all[,, t-burn] = gamma
      mus_all[,, t-burn] = mus
      sigmas_all[,,t-burn] = sigmas
    }
  }
  
  return(list(gamma_all = gamma_all, mus_all = mus_all, sigmas_all = sigmas_all))
}

draw_data <- function(fit) {
  
  N <- dim(fit$gamma_all)[1]
  K <- dim(fit$gamma_all)[2]
  R <- dim(fit$gamma_all)[3]
  d <- dim(fit$mus_all)[1]
  
  Y_pred <- array(0, dim = c(N, d, R))
  k_all <- matrix(0, nrow = N, ncol = R)
  
  for (r in 1:R) {
    
    k_all[, r] <- apply(as.matrix(fit$gamma_all[,,r]), 1, function(x) which(x == 1))
    mus <- t(fit$mus_all[, k_all[, r], r])
    sigmas <- t(fit$sigmas_all[, k_all[, r], r])
    Y_pred[,, r] <- mus + matrix(rnorm(N * d), nrow = N, ncol = d) * sqrt(sigmas) 
    
  }
  
  return(list(Y_pred = Y_pred, group = k_all))
}

marginal_ll_function = function(Y, fit) {
  R <- dim(fit$mus_all)[3]
  
  K <- dim(fit$mus_all)[2]
  D <- dim(fit$mus_all)[1]
  
  ll_mean = numeric(R)
  
  n = nrow(Y)
  
  for (r in 1:R) {
    mus = as.matrix(fit$mus_all[,,r])
    sigmas = as.matrix(fit$sigmas_all[,,r])
    gamma = gamma_draw(Y, mus, sigmas)
    temp = 0
    for (k in 1:K) {
      temp <- temp - 0.5 * sum(gamma[,k] * t((t(Y) - mus[, k])^2/sigmas[, k]))
      temp <- temp - 0.5 * sum(gamma[,k]) * sum(log((2 * pi)^D * sigmas[, k]))
    }
    temp = temp / n
    ll_mean[r] = exp(temp)
  }
  
  marginal_ll = 1/R * sum(1/ll_mean)
  marginal_ll = 1/marginal_ll
  
  return(marginal_ll)
}


ll_function <- function(Y, fit) {
  
  B <- 10
  S <- dim(fit$mus_all)[3]
  
  K <- dim(fit$mus_all)[2]
  
  ind <- sample(1:S, B, replace = F)
  
  ll_mean = numeric(B)

  n = nrow(Y)
  
  for (b in 1:B) {
    mus = as.matrix(fit$mus_all[,,ind[b]])
    sigmas = as.matrix(fit$sigmas_all[,,ind[b]])
    gamma = gamma_draw(Y, mus, sigmas)
    temp = 0
    for (k in 1:K) {
      temp <- temp - 0.5 * sum(gamma[,k] * t((t(Y) - mus[, k])^2/sigmas[, k]))
      temp <- temp - 0.5 * sum(gamma[,k]) * sum(log(sigmas[, k]))
    }
    temp = temp / n
    ll_mean[b] = temp
  }
  
  ll_mean = mean(ll_mean)
  
  return(-ll_mean)
}

ll_function_2 = function(Y, fit) {

  K <- dim(fit$mus_all)[2]
  
  ll_mean = 0
  
  mus = matrix(fit$mus_all[,,1], ncol = K)
  sigmas = matrix(fit$sigmas_all[,,1], ncol = K)
  
  n = nrow(Y)
  
  gamma = gamma_draw(Y, mus, sigmas)
  temp = 0
  for (k in 1:K) {
    temp <- temp - 0.5 * sum(gamma[,k] * t((t(Y) - mus[, k])^2/sigmas[, k]))
    temp <- temp - 0.5 * sum(gamma[,k]) * sum(log(sigmas[, k]))
  }
  temp = temp / n
  
  ll_mean = temp
  
  return(-ll_mean)
}


# EM algorithm

gamma_update = function(Y, mus, sigmas) {
  K = ncol(mus)
  n = nrow(Y)
  p = rep(1/K, K)
  dist = matrix(0, nrow = K, ncol = n)
  for (k in 1:K) {
    dist[k, ] = exp(-apply((t(Y) - mus[, k])/sqrt(sigmas[,k]), 2, function(x) sum(x^2)/2)) / sqrt((prod(sigmas[, k])))
  }
  gammas = dist * p
  gammas[, apply(gammas, 2, sum) == 0] = 0.5
  if (K > 1) {
    gammas = t(apply(gammas, 2, function(x) x / sum(x)))
  } else {
    gammas = matrix(apply(gammas, 2, function(x) x / sum(x)), nrow = n, ncol = K)
  }
  gammas
}

mu_update = function(Y, gammas, sigmas) {
  K = ncol(gammas)
  d = ncol(Y)
  nk = apply(gammas, 2, sum)
  mus = matrix(0, nrow = d, ncol = K)
  for (k in 1:K) {
    yk_mean <- apply(Y * gammas[, k], 2, sum)
    mu_var = solve(diag(1/sigmas[, k]) * nk[k] + diag(d)/mu_sd^2)
    mus[, k] = mu_var %*% yk_mean / sigmas[, k]
  }
  mus
}

sigma_update <- function(Y, gammas, mus) {
  K <- ncol(gammas) 
  d <- ncol(Y)
  sigmas <- matrix(0, nrow = d, ncol = K)
  for (k in 1:K) {
    temp <- t((t(Y) - mus[, k])^2)
    temp <- temp * gammas[, k]
    temp <- apply(temp, 2, sum)
    alpha_post <- sum(gammas[, k]) / 2 + sigma_alpha
    beta_post <- temp / 2 + sigma_beta
    sigmas[, k] <- beta_post / (alpha_post + 1)
  }
  sigmas
}

fit_em = function(Y, K) {
  d = ncol(Y)
  mus = matrix(rnorm(d * K, sd = 1), nrow = d, ncol = K)
  sigmas = matrix(1, nrow = d, ncol = K)
  
  delta_norm = 1
  mus_prev = mus
  n_step = 500
  
  for (i in 1:n_step) {
    gammas = gamma_update(Y, mus, sigmas)
    mus = mu_update(Y, gammas, sigmas)
    sigmas = sigma_update(Y, gammas, mus)
    
    delta_norm = sum((mus - mus_prev)^2)
    mus_prev = mus
    if (delta_norm < 10e-5) break
  }

  return(list(gammas = gammas, mus = mus, sigmas = sigmas))
}




draw_pp_data = function(N, R, fit) {
  
  gammas = fit$gammas
  mus = fit$mus
  sigmas = fit$sigmas
  
  K = ncol(mus)
  d = nrow(mus)
  
  Y_pred = array(0, dim = c(N, d, R))
  group = matrix(0, nrow = N, ncol = R)
  gamma_post = apply(gammas, 2, mean)
  
  for (r in 1:R) {
    group[,r] = sample(1:K, size = N, replace = T, prob = gamma_post)
    mus_draw = t(mus[, group[,r]])
    sigmas_draw = t(sigmas[, group[,r]])
    Y_pred[,, r] <- mus_draw + matrix(rnorm(N * d), nrow = N, ncol = d) * sqrt(sigmas_draw) 
  }
  
  return(list(Y_pred = Y_pred, group = group))
}

ll_em_function <- function(Y, fit) {
  
  mus = fit$mus
  sigmas = fit$sigmas

  K = ncol(mus)

  gammas = gamma_update(Y, mus, sigmas)
  
  temp = 0
  for (k in 1:K) {
    temp <- temp - 0.5 * sum(gammas[,k] * t((t(Y) - mus[, k])^2/sigmas[, k]))
    temp <- temp - 0.5 * sum(gammas[,k]) * sum(log(sigmas[, k]))
  }
  
  temp = temp / nrow(Y)
  
  -temp
}
