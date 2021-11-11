library(MASS)
library(tidyverse)
library(ggtext)
library(ggpubr)
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

set.seed(1234)

n = 2000
p = 10
R = 500
n_models = 2

X = matrix(rnorm(n * p), nrow = n, ncol = p)

theta = 2.5
beta = rep(0, p)

y_in = theta + X %*% beta + rnorm(n)
y_val = theta + X %*% beta + rnorm(n)

tXX = t(X) %*% X
tXX_inv = chol2inv(chol(tXX))

beta_in = as.vector(tXX_inv %*% (t(X) %*% y_in))
beta_val = as.vector(tXX_inv %*% (t(X) %*% y_val))

theta_in_A = as.vector(mean(y_in))
theta_in_B = as.vector(mean(y_in) - apply(X, 2, mean) %*% beta_in)
theta_val_A = as.vector(mean(y_val))
theta_val_B = as.vector(mean(y_val) - apply(X, 2, mean) %*% beta_val)

ll_pp_data = array(0, dim = c(n_models, n_models, R))

d_obs_A = numeric(R)
d_obs_B = numeric(R)

partial_pc_A = numeric(R)
partial_pc_B = numeric(R)

for (r in 1:R) {
  
  theta_A_rep = rnorm(1, mean = theta_in_A, sd = sqrt(1/n))
  y_rep_A = rnorm(n, mean = theta_in_A, sd = 1)
  
  theta_B_rep = rnorm(1, mean = theta_in_B, sd = sqrt(1/n))
  beta_B_rep = mvrnorm(1, mu = beta_in, Sigma = tXX_inv)
  
  y_rep_B = theta_B_rep + X %*% beta_B_rep + rnorm(n)
  
  ll_pp_data[1, 1, r] = mean((y_rep_A - theta_val_A)^2)
  ll_pp_data[1, 2, r] = mean((y_rep_A - theta_val_B - X %*% beta_val)^2)
  ll_pp_data[2, 2, r] = mean((y_rep_B - theta_val_B - X %*% beta_val)^2)
  
  d_obs_A[r] = mean((y_in - theta_val_A)^2)
  d_obs_B[r] = mean((y_in - theta_val_B - X %*% beta_val)^2)
  
  partial_pc_A[r] = as.numeric(d_obs_A[r] > ll_pp_data[1, 1, r]) 
  partial_pc_B[r] = as.numeric(d_obs_B[r] > ll_pp_data[2, 2, r]) 
  
}

mat = get_empirical_dists(ll_pp_data[1, 2,], ll_pp_data[2,2,])

temp1 = KL(x = mat)
temp2 = KL(x = mat[c(2, 1), ])
kl_dist = (temp1 + temp2)/2


gg <- vector("list", n_models * n_models)

ll_partial = numeric(n_models)
ll_partial[1] = mean(d_obs_A)
ll_partial[2] = mean(d_obs_B)

for (k in 1:n_models) {
  for (l in 1:n_models) {
    
    ind <- (l - 1) * n_models + k
    dat <- data.frame(generating = ll_pp_data[k,l,], candidate = ll_pp_data[l,l,])
    
    if (k < l) {
      gg[[ind]] <- ggplot() + 
        geom_density(data = dat, mapping = aes(x = generating, y = ..density..), fill = "red", alpha = 0.3) +
        geom_density(data = dat, mapping = aes(x = candidate, y = ..density..), fill = "blue", alpha = 0.3) +
        theme(axis.title=element_text(size=10), plot.title=element_text(size=12)) +
        labs(x = " ", y = " ", title = paste("PPN(d<sub>B</sub>, <span style='color:#FF0000;' > A</span>, <span style='color:#0000FF;' > B</span>)", sep = "")) +
        theme(plot.title = element_markdown()) +
        scale_x_continuous(breaks = scales::pretty_breaks(n = 3)) +
        scale_y_continuous(breaks = scales::pretty_breaks(n = 3)) 
      
    }
    
    if (k == l) {
      if (k == 1) {
        gg[[ind]] <- ggplot() + 
          geom_density(data = dat, mapping = aes(x = generating, y = ..density..), fill = "blue", alpha = 0.3) +
          theme(axis.title=element_text(size=10), plot.title=element_text(size=12)) +
          geom_vline(xintercept = ll_partial[k], color = "red", linetype = 1) +
          labs(x = " ", y = " ", title = expression(paste("PPC(", d[A], ", A)"))) +
          scale_x_continuous(breaks = scales::pretty_breaks(n = 3)) +
          scale_y_continuous(breaks = scales::pretty_breaks(n = 3)) 
      }
      if (k == 2) {
        gg[[ind]] <- ggplot() + 
          geom_density(data = dat, mapping = aes(x = generating, y = ..density..), fill = "blue", alpha = 0.3) +
          theme(axis.title=element_text(size=10), plot.title=element_text(size=12)) +
          geom_vline(xintercept = ll_partial[k], color = "red", linetype = 1) +
          labs(x = " ", y = " ", title = expression(paste("PPC(", d[B], ", B)"))) +
          scale_x_continuous(breaks = scales::pretty_breaks(n = 3)) +
          scale_y_continuous(breaks = scales::pretty_breaks(n = 3)) 
      }
    }
  }
}

fig1 <- ggarrange(plotlist = gg,
                  nrow = n_models, ncol = n_models,
                  align = "v")

pdf(file = paste('img/linear_n_', n, '_p_', p, '.pdf', sep = ''), width = 5, height = 4)

fig1
  
dev.off()




