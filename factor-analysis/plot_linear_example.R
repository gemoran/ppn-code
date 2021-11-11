library(reticulate)
library(reshape2)
library(ggplot2)
library(MASS)
library(grid)
library(gridExtra)
library(gtable)
library(ggpubr)
library(RColorBrewer)
library(dplyr)
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

n_models = 3
R = 500

np = import("numpy")

ll_pp_data = np$load("out/linear_pp_data.npy")

ll_partial = np$load("out/linear_ll_partial.npy")

partial_pc = numeric(n_models)

for (m in 1:n_models) {
  partial_pc[m] = min(sum(ll_pp_data[m, m, ] > ll_partial[m])/R, sum(ll_pp_data[m, m, ] < ll_partial[m])/R)
}


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

kl_dist = round(kl_dist, digits = 2)

write.table(kl_dist, file = "out/linear_kl_dist.txt", sep = " & ", row.names = F)

models <- c("PPCA", "VAE", "SKIP")

gg <- vector("list", n_models * n_models)

for (l in 1:n_models) {
  
  for (k in 1:n_models) {
    ind <- (l - 1) * n_models + k
    dat <- data.frame(generating = ll_pp_data[k,l,], candidate = ll_pp_data[l,l,])
    
    if (k != l) {
      gg[[ind]] <- ggplot() + geom_density(data = dat, mapping = aes(x = generating, y = ..density..), fill = "red", alpha = 0.3) +
        geom_density(data = dat, mapping = aes(x = candidate, y = ..density..), fill = "blue", alpha = 0.3)  +
        theme(axis.title=element_text(size=8), plot.title=element_text(size=11)) +
        labs(x = " ", y = " ", title = paste("PPN(d<sub>",models[l],"</sub>, <span style='color:#FF0000;' >", models[k], "</span>, <span style='color:#0000FF;' >", models[l], "</span>)", sep = "")) +
        theme(plot.title = element_markdown())  +
        scale_x_continuous(breaks = scales::pretty_breaks(n = 3)) +
        scale_y_continuous(breaks = scales::pretty_breaks(n = 3)) 
    }
    
    if (k == l) {
      gg[[ind]] <- ggplot() + geom_density(data = dat, mapping = aes(x = generating, y = ..density..), fill = "blue", alpha = 0.3) +
        theme(axis.title=element_text(size=8), plot.title=element_text(size=11)) + 
        geom_vline(xintercept = ll_partial[k], color = "red", linetype = 1) +
        labs(x = " ", y = " ", title = substitute(paste("PPC(", d[k], ", ", k, ")"), list(k = models[k])))  +
        scale_x_continuous(breaks = scales::pretty_breaks(n = 3)) +
        scale_y_continuous(breaks = scales::pretty_breaks(n = 3)) 
      
    }
  }
}


fig1 <- ggarrange(plotlist = gg,
                  nrow = n_models, ncol = n_models,
                  align = "v")


# plot discrepancies
fig_title = "img/linear_ppn.pdf"
pdf(fig_title, width = 7, height = 7)

print(fig1)

dev.off()

