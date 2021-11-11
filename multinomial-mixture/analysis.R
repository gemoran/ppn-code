source("utils.R")

set.seed(12345)

# read in data
dat = read.csv("dat/gelman_96_data.csv", header = T)

dat_new = data.frame(motor = rep(dat$motor, each = 3), 
                     cry = rep(dat$cry, each = 3),
                     fear = rep(c(1, 2, 3), 3 * 4))

dat_new$value = as.vector(t(dat[,-c(1,2)]))

n = sum(dat_new$value)

n_cat = length(unique(dat_new$motor)) + length(unique(dat_new$cry)) + length(unique(dat_new$fear))

tokens = NULL
for (i in 1:nrow(dat_new)) {
  add_tokens = numeric(n_cat)
  add_tokens[dat_new[i, 1]] = 1
  add_tokens[dat_new[i, 2] + 4] = 1
  add_tokens[dat_new[i, 3] + 7] = 1
  add_tokens = matrix(rep(add_tokens, dat_new[i, 4]), ncol = n_cat, byrow = T)
  tokens = rbind(tokens, add_tokens)
}

tokens_1 = tokens

dat = read.csv("dat/cohort2.csv", header = T)

dat_new = data.frame(motor = rep(dat$motor, each = 3), 
                     cry = rep(dat$cry, each = 3),
                     fear = rep(c(1, 2, 3), 3 * 4))

dat_new$value = as.vector(t(dat[,-c(1,2)]))

n = sum(dat_new$value)

n_cat = length(unique(dat_new$motor)) + length(unique(dat_new$cry)) + length(unique(dat_new$fear))

tokens = NULL
for (i in 1:nrow(dat_new)) {
  add_tokens = numeric(n_cat)
  add_tokens[dat_new[i, 1]] = 1
  add_tokens[dat_new[i, 2] + 4] = 1
  add_tokens[dat_new[i, 3] + 7] = 1
  add_tokens = matrix(rep(add_tokens, dat_new[i, 4]), ncol = n_cat, byrow = T)
  tokens = rbind(tokens, add_tokens)
}

tokens_2 = tokens

#-------------------------------------------
# split data into 3

tokens_all = rbind(tokens_1, tokens_2)

n_all = nrow(tokens_all)
n_train = ceiling(n_all/3)
n_test = n_train
n_val = n_all - (n_train + n_test)

ind = sample(1:n_all, n_all, replace = F)

tokens_train = tokens_all[ind[1:n_train], ]
tokens_test = tokens_all[ind[(n_train+1):(n_train+n_test)],]
tokens_val = tokens_all[ind[(n_train+n_test+1):n_all],]

# draws from posterior predictive
n_rep = 500
models = c(1, 2, 3, 4)
n_models = length(models)

fit = vector("list", n_models)
fit_test = vector("list", n_models)

alpha_theta = rep(2, n_cat)

use_em = F

# fit models K=1,2,3,4
for (k in 1:n_models) {
  alpha_p = rep(2, models[k])
  if (use_em) {
    fit[[k]] = EM_algorithm(tokens_train, models[k], alpha_p, alpha_theta)
    fit_test[[k]] = EM_algorithm(tokens_test, models[k], alpha_p, alpha_theta)
  } else {
    fit[[k]] = gibbs_sampling(tokens_train, models[k], alpha_p, alpha_theta, n_rep)
    fit_test[[k]] = gibbs_sampling(tokens_test, models[k],alpha_p, alpha_theta, n_rep)
  }
}

# run PPN ----------------------------------

ll_pp_data = array(0, dim = c(n_models, n_models, n_rep))

partial_pc = numeric(n_models)

ll_holdout = numeric(n_models)
ll_partial = numeric(n_models)

for (k in 1:n_models) {
  
  # from posterior predictive of model k, draw R datasets of size n_test
  if (use_em) {
    pp_data = draw_pp_data(n_train, n_rep, fit[[k]])$tokens_pred
  } else {
    pp_data = gibbs_pp_data(n_train, n_rep, fit[[k]])$tokens_pred
    
  }
  
  for (l in 1:n_models) {
    
    # calculate log likelihood of data from model k (under model l)
    for (r in 1:n_rep) {
      if (use_em) {
        ll_pp_data[k, l, r] = chi_squared(pp_data[,,r], fit_test[[l]])
      } else {
        ll_pp_data[k, l, r] = chi_squared_gibbs(pp_data[,,r], fit_test[[l]])
      }

    }
    cat("k = ", k, ", l = ", l, "\n")
    
  }
}

marginal_ll = numeric(n_models)

# calculate partial ppc 
for (k in 1:n_models) {
  
  if (use_em) {
    ll_partial[k] <- chi_squared(tokens_val, fit_test[[k]])
  } else {
    ll_partial[k] <- chi_squared_gibbs(tokens_val, fit_test[[k]])
    
    marginal_ll[k] = marginal_ll_function(tokens_train, fit[[k]])
  }
  
  partial_pc[k] <- min(sum(ll_pp_data[k, k, ] < ll_partial[k])/n_rep, sum(ll_pp_data[k, k, ] > ll_partial[k])/n_rep)
  
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
levels(dat$x) = c("M1/M2", "M2/M3", "M3/M4")

print(xtable(kl_dist), file = "out/kl_dist.txt")
print(xtable(bayes_factors), file = "out/bayes_factors.txt")

filename = "img/kl_line.pdf"

pdf(file = filename, width = 4, height = 2)

print(ggplot(dat = dat, aes(x = x, y = y, group = 1)) + 
        geom_line() + 
        xlab("Models compared") +
        ylab("Symmetrized KL"))

dev.off()

# plot log likelihoods of posterior predictive data 

gg <- vector("list", n_models * n_models)

for (l in 1:n_models) {
  
  for (k in 1:n_models) {
    ind <- (l - 1) * n_models + k
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

filename = "img/ppn.pdf"


pdf(file = filename, width = 9, height = 9)

print(fig1)

dev.off()


g_motor = vector("list", n_models)
g_crying = vector("list", n_models)
g_fear = vector("list", n_models)

motor_dat = as.data.frame(tokens_train[,1:4])
motor_dat = pivot_longer(motor_dat, cols = 1:ncol(motor_dat))
motor_dat$name = as.factor(motor_dat$name)
levels(motor_dat$name) = as.character(1:4)

g = ggplot(motor_dat) + 
  geom_bar(aes(x = name, y = value), stat = "summary", fun = "mean") +
  labs(x = "", y = "proportion", title = "") +
  coord_cartesian(ylim = c(0, 0.75))

pdf(file = "img/motor_observed.pdf", width = 2, height = 2.5, onefile = F)

print(annotate_figure(g, top = "Observed Data", left = "Motor"))

dev.off()

crying_dat = as.data.frame(tokens_train[,5:7])
crying_dat = pivot_longer(crying_dat, cols = 1:ncol(crying_dat))
crying_dat$name = as.factor(crying_dat$name)
levels(crying_dat$name) = as.character(1:length(unique(crying_dat$name)))

g =     ggplot(crying_dat) + 
  geom_bar(aes(x = name, y = value), stat = "summary", fun = "mean") +
  labs(x = " ", y = "proportion", title = " ") +
  coord_cartesian(ylim = c(0, 0.75))

pdf(file = "img/crying_observed.pdf", width = 2, height = 2)

print(
  annotate_figure(g, left = "Crying")
)

dev.off()

fear_dat = as.data.frame(tokens_train[,8:10])
fear_dat = pivot_longer(fear_dat, cols = 1:ncol(fear_dat))
fear_dat$name = as.factor(fear_dat$name)
levels(fear_dat$name) = as.character(1:length(unique(fear_dat$name)))

g = ggplot(fear_dat) + 
  geom_bar(aes(x = name, y = value), stat = "summary", fun = "mean") +
  labs(x = " ", y = "proportion", title = " ") +
  coord_cartesian(ylim = c(0, 0.75))

pdf(file = "img/fear_observed.pdf", width = 2, height = 2)

print(
  annotate_figure(g, left = "Fear")
)

dev.off()

for (k in 1:n_models) {
  if (use_em) {
    pp_data = draw_pp_data(n_train, 20, fit[[k]])$tokens_pred
  } else {
    pp_data = gibbs_pp_data(n_train, 20, fit[[k]])$tokens_pred
  }
  
  motor_dat = melt(apply(pp_data[,1:4,], c(2, 3), mean))
  names(motor_dat) = c('name', 'rep', 'value')
  motor_dat$name = as.factor(motor_dat$name)
  levels(motor_dat$name) = as.character(1:4)
  
  g_motor[[k]] =  ggplot(motor_dat, aes(x = name, y = value)) + 
    geom_bar(stat = "summary", fun = "mean") +
    stat_summary(fun.data = mean_sdl, geom = 'errorbar', width = .2) +
    labs(x = " ", y = " ", title = paste("K =", k)) +
    coord_cartesian(ylim = c(0, 0.75))
  
  crying_dat = melt(apply(pp_data[,5:7,], c(2, 3), mean))
  names(crying_dat) = c('name', 'rep', 'value')
  crying_dat$name = as.factor(crying_dat$name)
  levels(crying_dat$name) = as.character(1:length(unique(crying_dat$name)))
  
  g_crying[[k]] = ggplot(crying_dat, aes(x = name, y = value)) + 
    geom_bar(stat = "summary", fun = "mean") +
    stat_summary(fun.data = mean_sdl, geom = 'errorbar', width = .2) +
    labs(x = " ", y = " ", title = "") +
    coord_cartesian(ylim = c(0, 0.75))
  
  fear_dat = melt(apply(pp_data[,8:10,], c(2, 3), mean))
  names(fear_dat) = c('name', 'rep', 'value')
  fear_dat$name = as.factor(fear_dat$name)
  levels(fear_dat$name) = as.character(1:length(unique(fear_dat$name)))
  
  g_fear[[k]] = ggplot(fear_dat, aes(x = name, y = value)) + 
    geom_bar(stat = "summary", fun = "mean") +
    stat_summary(fun.data = mean_sdl, geom = 'errorbar', width = .2) +
    labs(x = " ", y = " ", title = "") +
    coord_cartesian(ylim = c(0, 0.75))
}

gg = ggarrange(plotlist = g_motor,
               nrow = 1)

pdf(file = "img/motor_pp.pdf", width = 8, height = 2.5)

print(
  annotate_figure(gg, top = "Posterior Predictive Data")
)

dev.off()

gg = ggarrange(plotlist = g_crying,
               nrow = 1)

pdf(file = "img/crying_pp.pdf", width = 8, height = 2)

print(gg)

dev.off()

gg = ggarrange(plotlist = g_fear,
               nrow = 1)

pdf(file = "img/fear_pp.pdf", width = 8, height = 2)

print(gg)

dev.off()








