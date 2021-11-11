source("utils.R")
library(dplyr)
library(reshape2)
library(ggplot2)
library(grid)
library(gridExtra)
library(gtable)
library(xtable)
library(ggpubr)
library(RColorBrewer)
library(ggtext)
library(philentropy)

set.seed(12345)
# example --------------------------------------------

# parameters
K_true <- 3 # true no. mixture components
d <- 2 # dimension of observed data
n_train <- 500 # no. training observations
n_test <- n_train # no. test observations
n_val = n_train
R <- 200 # no. datasets drawn from posterior predictive

# prior hyperparameters
# mu \sim N(0, mu_sd)
# sigma_mu \sim inv-gamma(sigma_alpha, sigma_beta) 
mu_sd <- 10 
sigma_alpha <- 1
sigma_beta <- 1

# generate data
mus_true <- matrix(c(-5, 5, 0, 0, 10, 5), nrow = d, ncol = K_true)
p_true <- rep(1/K_true, K_true)
sigmas_true <- matrix(c(1, 1, 2, 1, 2, 4), nrow = d, ncol = K_true)

Y_train <- matrix(0, nrow = n_train, ncol = d)
group_train <- sample(1:K_true, n_train, prob = p_true, replace = T)

for (i in 1:n_train) {
  Y_train[i, ] <- mvrnorm(n = 1, mu = mus_true[, group_train[i]], Sigma = diag(sigmas_true[, group_train[[i]]]))
}

Y_test <- matrix(0, nrow = n_test, ncol = d)
group_test <- sample(1:K_true, n_test, prob = p_true, replace = T)

for (i in 1:n_test) {
  Y_test[i, ] <- mvrnorm(n = 1, mu = mus_true[, group_test[i]], Sigma = diag(sigmas_true[, group_test[[i]]]))
}

Y_val <- matrix(0, nrow = n_val, ncol = d)
group_val <- sample(1:K_true, n_val, prob = p_true, replace = T)

for (i in 1:n_val) {
  Y_val[i, ] <- mvrnorm(n = 1, mu = mus_true[, group_val[i]], Sigma = diag(sigmas_true[, group_val[[i]]]))
}

# fit models
models <- c(1, 2, 3, 4)
n_models <- length(models)

save_result = F

for (i in 1:2) {
  if (i == 1) {
    use_em = T
  } else {
    use_em = F
  }
  
  fit <- vector("list", n_models)
  fit_test <- vector("list", n_models)
  
  # fit models
  for (k in 1:n_models) {
    if (use_em) {
      fit[[k]] <- fit_em(Y_train, models[k])
      fit_test[[k]] = fit_em(Y_test, models[k])
    } else {
      fit[[k]] <- gibbs(Y_train, models[k], burn = 100, npost = R)
      fit_test[[k]] <- gibbs(Y_test, models[k], burn = 100, npost = R)
    }
  }
  
  # posterior predictive model selection
  
  ll_pp_data <- array(0, dim = c(n_models, n_models, R))
  ll_partial_data = matrix(0, nrow = n_models, ncol = R)
  partial_pc <- numeric(n_models)
  ll_partial <- numeric(n_models)
  marginal_ll <- numeric(n_models)
  
  for (k in 1:n_models) {
    
    # from posterior predictive of model k, draw R datasets of size n_test
    if (use_em) {
      pp_data <- draw_pp_data(n_train, R, fit[[k]])$Y_pred
    } else {
      pp_data <- draw_data(fit[[k]])$Y_pred
    }
    
    for (l in 1:n_models) {
      
      # calculate log likelihood of data from model k (under model l)
      for (r in 1:R) {
        
        if (use_em) {
          ll_pp_data[k, l, r] <- ll_em_function(pp_data[,,r], fit_test[[l]])
        } else {
          ll_pp_data[k, l, r] <- ll_function(pp_data[,,r], fit[[l]])
        }
        
      }
      cat("k = ", k, ", l = ", l, "\n")
      
    }
  }
  
  # calculate ppc and pop_pc
  for (k in 1:n_models) {
    if (use_em) {
      ll_partial[k] = ll_em_function(Y_val, fit_test[[k]])
    } else {
      ll_partial[k] = ll_function(Y_val, fit_test[[k]])
      marginal_ll[k] = marginal_ll_function(Y_train, fit[[k]])
    }
    
    partial_pc[k] =  min(sum(ll_pp_data[k, k, ] < ll_partial[k])/R, sum(ll_pp_data[k, k, ] > ll_partial[k])/R)
  }
  
  bayes_factors = crossprod(t(marginal_ll), 1/marginal_ll)
  
  kl_dist <- matrix(NA, nrow = n_models, ncol = n_models)
  
  for (k in 1:n_models)  {
    for (l in 1:n_models) {
      if (k != l) {
        mat = get_empirical_dists(ll_pp_data[k, l,], ll_pp_data[l,l,])
        
        temp1 = KL(x = mat)
        temp2 = KL(x = mat[c(2, 1), ])
        kl_dist[l, k] <- (temp1 + temp2)/2
      }
    }
  }
  
  # plot super-diagonal of kl dist
  kl_diag = numeric(n_models - 1)
  for (k in 1:(n_models - 1)) {
    kl_diag[k] = kl_dist[k + 1, k]
  }
  
  dat = data.frame(x = c(2:n_models), y = kl_diag)
  dat$x = as.factor(dat$x)
  levels(dat$x) = c("M1/M2", "M2/M3", "M3/M4", "M4/M5")

  if (use_em) {
    filename = "img/kl_line_EM.pdf"
  } else {
    filename = "img/kl_line_gibbs.pdf"
  }
  
  if (save_result) {
    if (use_em) {
      write.table(kl_dist, file = "out/kl_dist_EM.txt", sep = " & ", row.names = F)
    } else {
      write.table(kl_dist, file = "out/kl_dist_gibbs.txt", sep = " & ", row.names = F)
      print(xtable(bayes_factors), file = "out/bayes_factors.txt")
    }
    
    pdf(file = filename, width = 4, height = 2)
    
    print(ggplot(dat = dat, aes(x = x, y = y, group = 1)) + 
            geom_line() + 
            xlab("Models compared") +
            ylab("Symmetrized KL"))
    
    dev.off()
  } else {
    print(ggplot(dat = dat, aes(x = x, y = y, group = 1)) + 
            geom_line() + 
            xlab("Models compared") +
            ylab("Symmetrized KL"))
  }

  
  
  
  # plot log likelihoods of posterior predictive data 
  
  gg <- vector("list", n_models * n_models)
  
  for (l in 1:n_models) {
    for (k in 1:n_models) {
      ind = (l - 1) * n_models + k

      dat <- data.frame(generating = ll_pp_data[k,l,], candidate = ll_pp_data[l,l,])
      
      if (k < l) {
        gg[[ind]] <- ggplot() + 
          geom_density(data = dat, mapping = aes(x = generating, y = ..density..), fill = "red", alpha = 0.3) +
          geom_density(data = dat, mapping = aes(x = candidate, y = ..density..), fill = "blue", alpha = 0.3) +
          theme(axis.title=element_text(size=10), plot.title=element_text(size=12)) +
          labs(x = " ", y = " ", title = paste("PPN(d<sub>K=",l,"</sub>, <span style='color:#FF0000;' > K = ", k, "</span>, <span style='color:#0000FF;' > K = ", l, "</span>)", sep = "")) +
          theme(plot.title = element_markdown()) +
          scale_x_continuous(breaks = scales::pretty_breaks(n = 3)) +
          scale_y_continuous(breaks = scales::pretty_breaks(n = 3)) 
          
      }
      
      if (k == l) {
        gg[[ind]] <- ggplot() + 
          geom_density(data = dat, mapping = aes(x = generating, y = ..density..), fill = "blue", alpha = 0.3) +
          theme(axis.title=element_text(size=10), plot.title=element_text(size=12)) +
          geom_vline(xintercept = ll_partial[k], color = "red", linetype = 1) +
          labs(x = " ", y = " ", title = substitute(paste("PPC(", d[paste("K = ", k)], ", K = ", k, ")"), list(k = k))) +
          scale_x_continuous(breaks = scales::pretty_breaks(n = 3)) +
          scale_y_continuous(breaks = scales::pretty_breaks(n = 3)) 
      }
      
    }
  }

  
  fig1 <- ggarrange(plotlist = gg,
            nrow = n_models, ncol = n_models,
            align = "v")
  
  # plot discrepancies
  if (use_em) {
    filename = "img/ppn_EM.pdf"
  } else {
    filename = "img/ppn_gibbs.pdf"
    
  }
  
  if (save_result) {
    pdf(file = filename, width = 9, height = 9)
    
    print(fig1)
    
    dev.off()
  } else {
    print(fig1)
  }

  
  
  # plot example data
  
  colnames(Y_test) <- c("x", "y")
  dat_train <- as.data.frame(Y_test)
  dat_train$group_train <- as.character(group_test)
  
  g <- ggplot() + geom_point(data = dat_train, aes(x = x, y = y)) +
    theme_bw() + theme(legend.position = "none") + 
    labs(title = " ")
  
  dat_train_subset = dat_train[sample(1:n_train, size = 100), ]
  
  gg <- vector("list", n_models)
  cols <- brewer.pal(4, "Dark2")
  for (k in 1:n_models) {
    
    if (use_em) {
      dat <- draw_pp_data(n_train, 2, fit[[k]])
    } else {
      dat <- draw_data(fit[[k]])
    }
    
    dat <- cbind(dat$Y_pred[,,1], dat$group[,1])
    
    dat_plot <- as.data.frame(dat)
    colnames(dat_plot) <- c("x", "y", "group")
    dat_plot$group <- as.character(dat_plot$group)
    
    # gg[[k]] <- g + geom_point(data = dat_plot, aes(x = x, y = y), alpha = 0.8) +
    #   labs(title = paste("K =", models[k])) 
    
    if (F) {
      gg[[k]] = ggplot() + stat_density_2d(data = dat_plot, aes(x = x, y = y, lty = group), color = "grey") +
        theme_bw() + labs(title = paste("K =", models[k])) +
        #   geom_point(data = dat_train, aes(x = x, y = y), alpha = 0.6) +
        theme(legend.position = "none") 
    }

    gg[[k]] = ggplot() + geom_point(data = dat_plot, aes(x = x, y = y)) +
      theme_bw() + labs(title = paste("K =", models[k])) +
      theme(legend.position = "none") +
      scale_x_continuous(breaks = scales::pretty_breaks(n = 3)) +
      scale_y_continuous(breaks = scales::pretty_breaks(n = 3)) 
    
  }
  
  if (use_em) {
    
    filename = "img/pp_draws_EM.pdf"
    filename2 = "img/data_EM.pdf"
  } else {
    filename = "img/pp_draws_gibbs.pdf"
    filename2 = "img/data_gibbs.pdf"
    
  }
  
  if (save_result) {
    pdf(file = filename, width = 8, height = 2.25)
    
    fig2 <- ggarrange(plotlist = gg, nrow = 1)
    print(annotate_figure(fig2, top = "Posterior Predictive Data"))
    
    dev.off()
    
    pdf(file = filename2, width = 2, height = 2.25)
    
    print(annotate_figure(g, top = "Observed Data"))
    
    dev.off() 
  } else {
    fig2 <- ggarrange(plotlist = gg, nrow = 1)
    print(fig2)
    
    print(g)
  }

}
