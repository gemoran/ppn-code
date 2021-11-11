library(MASS)
library(reshape2)
library(ggplot2)
library(LaplacesDemon)
library(grid)
library(gridExtra)
library(gtable)
library(xtable)
library(ggpubr)
library(RColorBrewer)
library(kSamples)
library(dplyr)
library(tidyverse)
library(ggtext)
library(philentropy)

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

theta_reweight = function(theta) {
  
  if (ncol(theta) == 1) {
    theta[1:4, ] = apply(matrix(theta[1:4, ]), 2, function(x) x / sum(x))
    theta[5:7, ] = apply(matrix(theta[5:7, ]), 2, function(x) x / sum(x))
    theta[8:10, ] = apply(matrix(theta[8:10, ]), 2, function(x) x / sum(x))
  } else {
    theta[1:4, apply(theta[1:4, ], 2, sum) == 0] = 0.5
    theta[5:7, apply(theta[5:7, ], 2, sum) == 0] = 0.5
    theta[8:10, apply(theta[8:10, ], 2, sum) == 0] = 0.5
    
    theta[1:4, ] = apply(theta[1:4, ], 2, function(x) x / sum(x))
    theta[5:7, ] = apply(theta[5:7, ], 2, function(x) x / sum(x))
    theta[8:10, ] = apply(theta[8:10, ], 2, function(x) x / sum(x))
  }

  theta 
}

E_step = function(tokens, theta, p) {
  n = nrow(tokens)
  n_class = length(p)
  z = matrix(0, nrow = n, ncol = n_class)
  if(any(is.na(theta))) print("theta is na here")
  for (i in 1:n) {
    for (k in 1:n_class) {
      z[i, k] = exp(sum(log(theta[, k]^tokens[i, ] + 1e-7))) * p[k]
    }
  }
  if (any(is.na(z))) {
    print("z is na here")
  }
  z[apply(z, 1, sum) == 0, ] = 0.5

  if (n_class == 1) {
    z = matrix(z)
  }
  z_sum = apply(z, 1, sum)    # n x 1
  z = z / z_sum 
  z
}

M_step = function(tokens, z, alpha_p, alpha_theta) {
  n = nrow(tokens)
  n_cat = ncol(tokens)
  n_class = ncol(z)
  
  p = apply(z, 2, mean) + alpha_p - 1
  p = p/(sum(p) + 1e-7)
  
  theta = matrix(0, nrow = n_cat, ncol = n_class)
  
  for (j in 1:n_cat) {
    for (l in 1:n_class) {
      theta[j, l] = sum(tokens[, j] * z[, l] + alpha_theta[j] - 1)
    }
  }
  
  theta = theta_reweight(theta)
  
  return(list(theta = theta, p = p))
}


EM_algorithm = function(tokens, n_class, alpha_p, alpha_theta) {
  n_cat = ncol(tokens)
  n = nrow(tokens)
  
  # initialize theta and p
  p = as.vector(rdirichlet(1, rep(50, n_class)))
  theta = matrix(rbeta(n_cat * n_class, 50, 50), nrow = n_cat, ncol = n_class)
  theta = theta_reweight(theta)

  iter = 5000
  diff = 1
  theta_old = theta
  p_old = p
  z_old = matrix(0, nrow = n, ncol = n_class)
  for (i in 1:iter) {
    z = E_step(tokens, theta, p)
    if (any(is.na(z))) {
      print("z is na")
      print(p)
    }
    
    M_out = M_step(tokens, z, alpha_p, alpha_theta)
    theta = M_out$theta
    if (any(is.na(theta))) print("theta is na")
    
    p = M_out$p
    if (any(is.na(p))) print("p is na")
    
    diff = sum((theta - theta_old)^2) + sum((p - p_old)^2) + sum((z - z_old)^2)
    
    if (is.na(diff)) print("diff is na")
    if (diff < 1e-7) {
      break
    }
    theta_old = theta
    p_old = p
    z_old = z
  }
  
  cat("EM algorithm took ", i, " iterations",  "\n")
  
  return(list(z = z, theta = theta, p = p))
}

draw_pp_data = function(n, n_rep, fit) {
  
  z = fit$z
  theta = fit$theta
  p = fit$p
  
  n_cat = nrow(theta)
  n_class = ncol(theta)
  
  tokens_pred = array(0, dim = c(n, n_cat, n_rep))
  group = matrix(0, nrow = n, ncol = n_rep)

  for (r in 1:n_rep) {
    group[,r] = sample(1:n_class, size = n, replace = T, prob = p)
    theta_draw = theta[, group[,r]]    # n_cat x n 
    theta_draw = data.frame(theta_draw)
    tokens_pred[, 1:4, r] = t(mapply(rmultinom, prob = theta_draw[1:4, ], MoreArgs = list(n = 1, size = 1)))
    tokens_pred[, 5:7, r] = t(mapply(rmultinom, prob = theta_draw[5:7, ], MoreArgs = list(n = 1, size = 1)))
    tokens_pred[, 8:10, r] = t(mapply(rmultinom, prob = theta_draw[8:10, ], MoreArgs = list(n = 1, size = 1)))
  }
  
  return(list(tokens_pred = tokens_pred, group = group))
}

ll_function = function(tokens, fit) {
  
  z = fit$z
  theta = fit$theta
  p = fit$p
  
  n = nrow(tokens)
  n_cat = ncol(tokens)
  n_class = length(p)
  
  z = E_step(tokens, theta, p)
  
  ll = 0
  for (i in 1:n) {
    for (j in 1:n_cat) {
      for (l in 1:n_class) {
        ll = ll + tokens[i, j] * z[i, l] * log(theta[j, l] + 1e-8)
      }
    }
  }
  
  return(-ll/n)
}

chi_squared = function(tokens, fit) {
  
  # 2 * sum(y * log (y / E[y | theta]))
  n = nrow(tokens)
  n_cat = ncol(tokens)
  
  theta = fit$theta
  p = fit$p
  
  z = E_step(tokens, theta, p)
  
  EY = matrix(0, nrow = n, ncol = n_cat)
  for (i in 1:n) {
    temp = z[i, ] * t(theta)
    EY[i, ] = apply(temp, 2, sum)
  }
  
  out = 2 * mean(tokens * (log(tokens/(EY + 1e-7) + 1e-7)))
  
  out
}

chi_squared_gibbs = function(tokens, fit) {
  
  # 2 * sum(y * log (y / E[y | theta]))
  n = nrow(tokens)
  n_cat = ncol(tokens)
  
  B <- 20
  S <- dim(fit$z_draws)[3]
  ind <- sample(1:S, B, replace = F)
  
  z = fit$z_draws
  theta = fit$theta_draws
  p = fit$p_draws
  
  EY = matrix(0, nrow = n, ncol = n_cat)
  
  for (b in 1:B) {
    z_current = gibbs_z(tokens, theta[,,ind[b]], p[,ind[b]])
    
    for (i in 1:n) {
      temp = z_current[i,] * t(theta[,,ind[b]])
      EY[i, ] = EY[i, ] + apply(temp, 2, sum)
    }
    
  }
  
  EY = EY / B

  out = 2 * mean(tokens * log(tokens /(EY + 1e-7) + 1e-7))
  
  out
}


draw_z = function(tokens, theta, p) {
  n_class = length(p)
  n = nrow(tokens)
  probs = E_step(tokens, theta, p)
  z = apply(probs, 1, function(x) sample(1:n_class, 1, replace = T, prob = x))
  z = cbind(1:n, z)
  z_out = matrix(0, nrow = n, ncol = n_class)
  z_out[z] = 1
  z_out
}

gibbs_p = function(z, alpha_p) {
  z_sum = apply(z, 2, sum) 
  p = as.vector(rdirichlet(1, z_sum + alpha_p - 1))
  p
}

gibbs_z = function(tokens, theta, p) {
  n = nrow(tokens)
  n_class = length(p)
  
  if (n_class == 1) {
    theta = as.numeric(theta)
    theta = matrix(theta, ncol = 1)
  }
  
  z_prob = matrix(0, nrow = n, ncol = n_class)
  if(any(is.na(theta))) print("theta is na here")
  for (i in 1:n) {
    for (k in 1:n_class) {
      z_prob[i, k] = exp(sum(log(theta[, k]^tokens[i, ] + 1e-7))) * p[k]
    }
  }
  if (any(is.na(z_prob))) {
    print("z is na here")
  }
  z_prob[apply(z_prob, 1, sum) == 0, ] = 1
  if (n_class == 1) {
    z_prob = matrix(z_prob)
  }
  z_sum = apply(z_prob, 1, sum)    # n x 1
  z_prob = z_prob / z_sum 
  z = matrix(0, nrow = n, ncol = n_class)
  for (i in 1:n) {
    z[i, ] = rmultinom(1, 1, z_prob[i, ])
  }
  z
}

gibbs_theta = function(tokens, z, p, alpha_theta) {
  # alpha_theta a vector of length n_cat
  n = nrow(tokens)
  n_cat = ncol(tokens)
  n_class = ncol(z)
  
  theta_post = matrix(0, nrow = n_cat, ncol = n_class)
  
  for (j in 1:n_cat) {
    for (k in 1:n_class) {
      theta_post[j, k] = sum(tokens[, j] * z[, k])
    }
  }
  
  theta_post = theta_post + alpha_theta - 1
  theta = matrix(0, nrow = n_cat, ncol = n_class)
  
  if (ncol(theta) == 1) {
    theta[1:4, 1] = rdirichlet(1, theta_post[1:4, 1])
    theta[5:7, 1] = rdirichlet(1, theta_post[5:7, 1])
    theta[8:10, 1] = rdirichlet(1, theta_post[8:10, 1])
  } else {
    theta_post[1:4, apply(theta_post[1:4, ], 2, sum) == 0] = 1
    theta_post[5:7, apply(theta_post[5:7, ], 2, sum) == 0] = 1
    theta_post[8:10, apply(theta_post[8:10, ], 2, sum) == 0] = 1
    
    for (k in 1:n_class) {
      theta[1:4, k] = rdirichlet(1, theta_post[1:4, k])
      theta[5:7, k] = rdirichlet(1, theta_post[5:7, k])
      theta[8:10, k] = rdirichlet(1, theta_post[8:10, k])
    }

  }

  return(theta)
}

gibbs_sampling = function(tokens, n_class, alpha_p, alpha_theta, n_rep){
  n = nrow(tokens)
  n_cat = ncol(tokens)
  
  burn = 100
  iter = n_rep + burn
  
  z_draws = array(0, dim = c(n, n_class, n_rep))
  theta_draws = array(0, dim = c(n_cat, n_class, n_rep))
  p_draws = matrix(0, nrow = n_class, ncol = n_rep)
  
  # initialize theta and p
  p = as.vector(rdirichlet(1, rep(1, n_class)))
  theta = matrix(rbeta(n_cat * n_class, 1, 1), nrow = n_cat, ncol = n_class)
  theta = theta_reweight(theta)
  
  for (i in 1:iter) {
    z = gibbs_z(tokens, theta, p)
    p = gibbs_p(z, alpha_p)
    theta = gibbs_theta(tokens, z, p, alpha_theta)
    
    if (i > burn) {
      z_draws[,,i-burn] = z
      theta_draws[,,i-burn] = theta
      p_draws[, i-burn ] = p
    }
  }
  
  return(list(z_draws = z_draws, theta_draws=theta_draws, p_draws=p_draws))
}

marginal_ll_function <- function(tokens, fit) {
  
  R <- dim(fit$z_draws)[3]
  
  n_class <- dim(fit$z_draws)[2]
  n_cat = dim(fit$theta_draws)[2]
  
  ll_mean = numeric(R)
  
  n = nrow(tokens)
  
  for (r in 1:R) {
    temp = 0
    if (n_class == 1) {
      theta = matrix(fit$theta_draws[,,r])
    } else {
      theta = fit$theta_draws[,,r]
    }
    p = fit$p_draws[,r]
    
    z = gibbs_z(tokens, theta, p)
    
    for (i in 1:n) {
      for (j in 1:n_cat) {
        for (l in 1:n_class) {
          temp = temp + tokens[i, j] * z[i, l] * log(theta[j, l] + 1e-8)
        }
      }
    }
    ll_mean[r] = exp(temp/n)
  }
  
  marginal_ll = 1/R * sum(1/ll_mean)
  marginal_ll = 1/marginal_ll
  
  return(marginal_ll)
}

ll_gibbs_function <- function(tokens, fit) {
  
  B <- 100
  S <- dim(fit$z_draws)[3]
  
  n_class <- dim(fit$z_draws)[2]
  n_cat = dim(fit$theta_draws)[2]
  
  ind <- sample(1:S, B, replace = F)
  
  ll_mean = numeric(B)
  
  n = nrow(tokens)
  
  for (b in 1:B) {
    temp = 0
    if (n_class == 1) {
      theta = matrix(fit$theta_draws[,,ind[b]])
    } else {
      theta = fit$theta_draws[,,ind[b]]
    }
    p = fit$p_draws[,ind[b]]
    
    z = gibbs_z(tokens, theta, p)
    
    for (i in 1:n) {
      for (j in 1:n_cat) {
        for (l in 1:n_class) {
          temp = temp + tokens[i, j] * z[i, l] * log(theta[j, l] + 1e-8)
        }
      }
    }
    ll_mean[b] = temp
  }
  
  ll_mean = mean(ll_mean)
  
  return(-ll_mean/n)
}

gibbs_pp_data = function(n, n_rep, fit) {
  
  z = fit$z_draws
  theta = fit$theta_draws
  p = fit$p_draws
  
  n_class = ncol(z)
  
  n_cat = nrow(theta)
  n_class = ncol(theta)
  
  tokens_pred = array(0, dim = c(n, n_cat, n_rep))
  group = matrix(0, nrow = n, ncol = n_rep)
  
  for (r in 1:n_rep) {
    if (n_class == 1) {
      theta_draw = theta[,,r]
    } else {
      group[,r] = apply(z[,,r], 1, function(x) which(x!=0))
      theta_draw = theta[, group[,r], r]    # n_cat x n 
      theta_draw = data.frame(theta_draw)
    }

    if (n_class == 1) {
      tokens_pred[, 1:4, r] = t(rmultinom(n, 1, prob = theta_draw[1:4]))
      tokens_pred[, 5:7, r] = t(rmultinom(n, 1, prob = theta_draw[5:7]))
      tokens_pred[, 8:10, r] = t(rmultinom(n, 1, prob = theta_draw[8:10]))
    } else {
      tokens_pred[, 1:4, r] = t(mapply(rmultinom, prob = theta_draw[1:4, ], MoreArgs = list(n = 1, size = 1)))
      tokens_pred[, 5:7, r] = t(mapply(rmultinom, prob = theta_draw[5:7, ], MoreArgs = list(n = 1, size = 1)))
      tokens_pred[, 8:10, r] = t(mapply(rmultinom, prob = theta_draw[8:10, ], MoreArgs = list(n = 1, size = 1)))
      
    }
    
   }
  
  return(list(tokens_pred = tokens_pred, group = group))
}
