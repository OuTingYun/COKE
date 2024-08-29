is_discrete <- function(x) {
  grepl("D", x, fixed = TRUE)
}

is_continuous <- function(x) {
  grepl("C", x, fixed = TRUE)
}

get_discrete_edge_nums <- function(A) {
  c(sum(A[sapply(rownames(A), is_discrete), ]), # number of edges pointing to a discrete node
    sum(A[, sapply(rownames(A), is_discrete)]), # number of edges pointing from a discrete node
    sum(A[sapply(rownames(A), is_discrete), sapply(rownames(A), is_discrete)])) # number of edges pointing from a discrete node and to a discrete node
}

get_continuous_edge_nums <- function(A) {
  c(sum(A[sapply(rownames(A), is_continuous), ]), # number of edges pointing to a continuous node
    sum(A[, sapply(rownames(A), is_continuous)]), # number of edges pointing from a continuous node
    sum(A[sapply(rownames(A), is_continuous), sapply(rownames(A), is_continuous)])) # number of edges pointing from a continuous node and to a discrete node
}

evaluate_scores <- function(A, A_learn) {
  c(ap_score(A, A_learn), 
    discrete_ap_score(A, A_learn),
    continuous_ap_score(A, A_learn), 
    ar_score(A, A_learn), 
    discrete_ar_score(A, A_learn), 
    continuous_ar_score(A, A_learn))
}

discrete_ap_score <- function(A, A.learn) {
  # correctly predicted adjacency
  cpa <- get_discrete_edge_nums(A * A.learn)  # when both entries are 1
  # all predicted adjacent
  apa <- get_discrete_edge_nums(A.learn)
  # adjacency precision
  ap <- cpa / apa
  
  ap
}


discrete_ar_score <- function(A, A.learn) {
  # correctly predicted adjacent
  cpa <- get_discrete_edge_nums(A * A.learn)  # when both entries are 1
  # all true adjacent
  ata <- get_discrete_edge_nums(A)
  # adjacency precision
  ar <- cpa / ata
  
  ar
}


continuous_ap_score <- function(A, A.learn) {
  # correctly predicted adjacency
  cpa <- get_continuous_edge_nums(A * A.learn)  # when both entries are 1
  # all predicted adjacent
  apa <- get_continuous_edge_nums(A.learn)
  # adjacency precision
  ap <- cpa / apa
  
  ap
}


continuous_ar_score <- function(A, A.learn) {
  # correctly predicted adjacent
  cpa <- get_continuous_edge_nums(A * A.learn)  # when both entries are 1
  # all true adjacent
  ata <- get_continuous_edge_nums(A)
  # adjacency precision
  ar <- cpa / ata
  
  ar
}

ap_score <- function(A, A.learn) {
  # correctly predicted adjacent
  cpa <- sum(A * A.learn)  # when both entries are 1
  # all predicted adjacent
  apa <- sum(A.learn)
  # adjacency precision
  ap <- cpa / apa
  
  ap
}

## AR: adjacency recall
ar_score <- function(A, A.learn) {
  # correctly predicted adjacent
  cpa <- sum(A * A.learn)  # when both entries are 1
  # all true adjacent
  ata <- sum(A)
  # adjacency precision
  ar <- cpa / ata
  
  ar
}

## Structural Hamming Distance (SHD)
# reference: https://github.com/cran/pcalg/blob/afe34d671144292350173b6f534c0eeb7fd0bb90/R/pcalg.R#L1086
shd_score <- function(a1, a2) {
  a1 <- as.matrix(a1)
  a2 <- as.matrix(a2)
  shd <- 0
  s1 <- a1 + t(a1)
  s2 <- a2 + t(a2)
  s1[s1 == 2] <- 1
  s2[s2 == 2] <- 1
  ds <- s1 - s2
  ind <- which(ds > 0)
  a1[ind] <- 0
  shd <- shd + length(ind)/2
  ## Add missing edges to g1
  ind <- which(ds < 0)
  a1[ind] <- a2[ind]
  shd <- shd + length(ind)/2
  ## Compare Orientation
  d <- abs(a1-a2)
  ## return
  shd <- shd + sum((d + t(d)) > 0)/2
  shd
}

##### The following functions are to overshadow the ones in the CausalLearn package #####
# gen_*_constraints() uses CausalLearn::extract_seq if not put in this file.
extract_seq <- function(name) {
  as.numeric(gsub("G([0-9]+)[.](\\w+)", "\\1", name))
}

gen_blacklist <- function(df) {
  cnames <- colnames(df)
  if (is.unsorted(sapply(cnames, extract_seq))) {
    stop("The column names of the data frame should be sorted by seq in an increasing order.")
  }
  bl <- rbind(gen_group_contraints(cnames), 
              gen_temporal_constraints(cnames), 
              gen_distance_constraints(cnames))
  bl
}

gen_group_contraints <- function(cnames) {
  seqs <- sort(unique(sapply(cnames, extract_seq)))
  bl <- c()
  for (seq in seqs) {
    bl <- rbind(bl, bnlearn::set2blacklist(cnames[sapply(cnames, extract_seq) == seq]))
  }
  bl
}

gen_temporal_constraints <- function(cnames) {
  bnlearn::ordering2blacklist(cnames)
}

#' In the original package, generating distance constraints is incorporated inside 
#'  the generation of domain constraints. For simulation's purpose, I extract it out here.
#'  For the sake of coding convenience, I set the max_step_distance to be 30 by default.
gen_distance_constraints <- function(cnames) {
  bl <- c()
  for (name in cnames) {
    seq <- extract_seq(name)
    if (seq > 30 & any(sapply(cnames, extract_seq) <= seq - 30)) {
      bl <- rbind(bl, bnlearn::tiers2blacklist(list(name, cnames[sapply(cnames, extract_seq) <= seq - 30])))
    }
  }
  bl
}

boot_bnl_fit <- function(df, method='hc-bic', ncores=25, nboot=100, additional_blacklist = NULL) {
  stopifnot(is_bnlearn_method(method))
  bl <- gen_blacklist(df)
  if (!is.null(additional_blacklist)) {
   bl <- rbind(bl, additional_blacklist) 
  }
  df <- df_discretize(df)
  doParallel::registerDoParallel(cores = ncores)
  i <- 0 # TODO need to examine whether adding it will make the bootstrap fail.
  arcs <- foreach(i = 1:nboot, .combine = "rbind", .packages = c("bnlearn")) %dopar% {
    set.seed(2021 + i)
    indices <- sample(nrow(df), replace = TRUE)
    bnl_structure(df[indices, ], method, bl)$arcs
  }
  data.frame(dplyr::summarise(dplyr::group_by_all(as.data.frame(arcs)),
                              strength = dplyr::n() / nboot))
}

bnl_fit <- function(df, method='hc-bic') {
  stopifnot(is_bnlearn_method(method))
  df <- df_discretize(df)
  bl <- gen_blacklist(df)
  structure <- bnl_structure(df, method, bl)
  bnlearn::arc.strength(structure, df, criterion = "bde")
}

#' Simulate data for hierarchical modeling.
#' 
#' In the adjacency setup, we assume that variables inside the same block are 
#'  closer to each other in comparison with the ones outside the block. Therefore, 
#'  we set different probabilities for inside and outside the block when drawing 
#'  parents for each node.
#'  
#' Note: we don't use pr.cont in the simulation here, as we are considering a 
#'  setting that is similar to the real analysis. Instead, we use a different 
#'  parameter num_discrete_per_step. See details in the param section.
#' 
#' @param n sample size
#' @param p number of variables 600
#' @param k number of steps 60
#' @param b number of blocks 10
#' @param num_discrete_per_step when there are fewer than num_discrete_per_step 
#'  variables in the step, then all of them are set as continuous; when there 
#'  are more than or equal to num_discrete_per_step variables in the step, then 
#'  num_discrete_per_step of them are set as binary and the rest are set as 
#'  continuous. This is to match the setting in the real analysis that there are 
#'  10 measurement steps.
#' @param degree maximum number of parents for each node 4
#' @param seed random seed for reproducibility
#' @param rand.mean whether to randomly generate mean 
#' @param fixed.beta
#' @param imb controls imbalance ratio, should be greater than or equal to 1. The greater 
#'  imb is, the more imbalanced the data is.
#' @param block_tightness when drawing the parents for each node, with 
#'  block_tightness probability, the parent will be from the same block as the 
#'  node, and with 1 - block_tightness probability, the parent will be from 
#'  other blocks.
#' @param max_step_distance the maximum step distance, if two steps are farther 
#'  away, then no links are allowed in between.
#' @param missing_rate the probability that a step is missing. If a step is 
#'  missing, then all variables in this steps are missing.
data.simulate <- function(n, p, k, b, num_discrete_per_step = 5, degree = 4, 
                          seed = 99, rand.mean = FALSE, fixed.beta = 0, 
                          imb = 1, block_tightness = 0.7, 
                          max_step_distance = 30, missing_rate = 0.0) {
  ### Step 1: adjacency matrix
  ## 1.1 set up the initial discrete/continuous state
  a <- vector(length = p) # 0 means discrete, and 1 means continuous
  
  ## 1.2 set up steps
  # randomly generate the border points of steps
  set.seed(seed + 1)
  ib <- sort(c(0, sample(1:(p-1), k-1), p))  # (ib[l]+1):ib[l+1] corresponds to group l
  vnames <- c() # variable names
  steps <- list()  # grouping information
  for(l in 1:k) {
    # set group index
    ig <- (ib[l] + 1):ib[l+1]
    # append to steps
    steps[[l]] <- c(ig) # steps contains the indices of variables
    # assign discrete/continuous status
    num_vars <- length(ig)
    if (num_vars < num_discrete_per_step) {
      a[ig] <- 1 # assign all to be continuous
    } else if (num_vars == num_discrete_per_step) {
      a[ig] <- 0 # assign all to be discrete
    } else {
      a[ig[1]:(ig[1] + num_discrete_per_step - 1)] <- 0 # assign the first 5 to be discrete
      a[(ig[1] + num_discrete_per_step):(ig[num_vars])] <- 1 # assign the rest to be continuous
    }
    # get names for steps, C for continuous, D for discrete, C1, C2, ..., D1, D2, ...
    gnames <- vector(length = num_vars)  # initialize group names
    ag <- a[ig]  # get the continuous/discrete status
    gnames[which(ag==1)] <- paste("S", l-1, "_", "C", 1:sum(ag), sep="")  # name the continuous nodes
    gnames[which(ag==0)] <- paste("S", l-1, "_", "D", 1:sum(1-ag), sep="")  # name the discrete nodes
    # append to vnames
    vnames <- c(vnames, gnames)
  }
  
  ## 1.3 set up blocks
  # randomly generate the border points of blocks
  set.seed(seed + 2)
  ib <- sort(c(0, sample(1:(k-1), b-1), k))  # (ib[l]+1):ib[l+1] corresponds to block l
  blocks <- list()
  for (l in 1:b) {
    # set up block index
    ig <- (ib[l] + 1):ib[l+1]
    blocks[[l]] <- c(ig) # the ith block contains the indices of steps
  }
  
  ## 1.4 set up adjacency structure
  # A should be a lower triangular matrix with diagonal being 0 to satisfy the temporal constraint
  A <- matrix(0, nrow = p, ncol = p)
  for (l in 1:b) {
    start_var_index <- steps[[blocks[[l]][1]]][1] # the start variable index in the block
    end_vars <- steps[[blocks[[l]][length(blocks[[l]])]]]
    end_var_index <- end_vars[length(end_vars)] # the ending variables index in the block
    for (i in start_var_index:end_var_index) {
      # degree constraints
      d <- min(degree, i - 1)
      # randomly generate the number of parents uniformly from 0 to d
      set.seed(seed + i)
      np <- floor(runif(1) * (d + 1))
      # generate the indices of parents: 1 - block_tightness prob from outside blocks, block_tightness prob from the same block
      out_block_parents <- sample(1:(start_var_index - 1), min(np, start_var_index - 1))
      if (i == start_var_index) {
        ip <- out_block_parents # indices of parents
      } else {
        in_block_parents <- sample(start_var_index:(i - 1), min(np, i - start_var_index))
        block_choice <- rbinom(np, 1, block_tightness)
        ip <- c(na.omit(c(out_block_parents[which(block_choice == 0)], in_block_parents[which(block_choice == 1)])))
      }
      # set the parent nodes
      A[i, ip] <- 1
    }
  }

  
  ## 1.5 add grouping constraints (no within-group causal relations) and distance constraints (two nodes that are more than max_step_distance steps far away cannot be linked)
  for (l in 1:k) {
    ig <- steps[[l]]
    # set adjacency matrix elements that are within a group to be 0
    #A[ig, ig] <- 0   # no within-group causal relations
    
    if (l + max_step_distance <= k) {
      A[steps[[l + max_step_distance]]:p, ig] <- 0 # two nodes that are more than max_step_distance steps far away cannot be linked  
    }
  }

  
  ### Step 2: design matrix
  ## 2.1 initialize the design matrix
  X <- matrix(0, nrow = n, ncol = p)
  
  ## 2.2 fill in the design matrix 
  for (i in 1:p){
    # if it is a continuous variable
    if (a[i] == 1) {
      set.seed(seed + i)
      if (sum(A[i, ]) == 0) { # if no parents, then random Gaussian N(0, 1)
        if (rand.mean) {
          X[, i] <- rnorm(n, rnorm(1, 0, 2), 1)  # add some variation to mean
        } else {
          X[, i] <- rnorm(n, 0, 1)
        }
      } else {  # if with parents, then conditional Gaussian N(X.beta, 1)
        # randomly generate coefficients
        if (rand.mean) {
          beta <- rnorm(p, rnorm(p, 0, 1), 1)  # different means for beta
        } else if (fixed.beta) {
          beta <- rep(fixed.beta, p)
        } else {
          beta <- rnorm(p, 0, 1)
        }
        # set beta for non-parents to be 0
        beta <- A[i, ] * beta  
        # conditional Gaussian
        set.seed(seed + 2 * i)
        X[, i] <- rnorm(n, X%*%beta, 1)
      }
    } else { # if it is a discrete variable
      if (sum(A[i, ]) == 0) { # if no parents, then Bernoulli with random prob
        # randomly generate the probability and Bernoulli variables
        std <- 0
        s <- seed + 2 * i
        while(!std) {   # ensure informative columns
          s <- s + 1
          set.seed(s)
          X[, i] <- rbinom(n, 1, runif(1) / imb)
          std <- sd(X[, i])
        }
      } else { # if with parents, then Bernoulli with logit prob
        # randomly generate coefficients
        if (rand.mean) {
          beta <- rnorm(p, rnorm(p, 0, 1), 1)  # different means for beta
        } else if (fixed.beta) {
          beta <- rep(fixed.beta, p)
        } else {
          beta <- rnorm(p, 0, 1)
        }
        # set beta for non-parents to be 0
        beta <- A[i, ] * beta
        # calculate prob under logistic assumption
        pr <- plogis(X%*%beta)
        # conditional Bernoulli
        std <- 0
        s <- seed + 2 * i
        while(!std) {   # ensure informative columns
          s <- s + 1
          set.seed(s)
          X[, i] <- rbinom(n, 1, pr / imb)
          std <- sd(X[, i])
        }
      }
    }
  }
  
  ### Step 3: add missingness to the data:
  missingness <- matrix(data = rbinom(n * k, 1, missing_rate), nrow = n, ncol = k)
  missingness <- t(apply(missingness, 1, function(x) rep(x, times = sapply(steps, length))))
  X[which(missingness == 1)] <- NA
  
  ### Step 4: wrap up data frame, adjacency matrix, arcs
  df <- as.data.frame(X)
  colnames(df) <- vnames
  
  colnames(A) <- vnames
  rownames(A) <- vnames
  
  # get arcs structure in the network
  adj.ind <- which(A == 1, arr.ind = T)
  net.arcs <- as.data.frame(cbind(vnames[adj.ind[, 2]], vnames[adj.ind[, 1]]))
  colnames(net.arcs) <- c('from', 'to')
  
  # return df and grouping info
  list(df = df, adj = A, arcs = net.arcs, steps = steps, blocks = blocks, 
       params = list(n = n, p = p, k = k, b = b, 
                     num_discrete_per_step = num_discrete_per_step, 
                     degree = degree, seed = seed, 
                     missing_rate = missing_rate))
}



#' Perform block-level data processing
#' 
#' @param sim.data simulated data generated by data.simulate()
#' @param npc number of principal components
block_processing <- function(sim.data, npc = 4) {
  df <- sim.data$df
  blocks <- sim.data$blocks
  num_discrete_per_step <- sim.data$params$num_discrete_per_step
  varnames <- colnames(df)
  
  block_df <- c()
  block_df_varnames <- c()
  for (l in 1:length(blocks)) {
    block_varnames <- varnames[sapply(varnames, extract_seq) %in% blocks[[l]]]
    block_continuous_varnames <- block_varnames[!sapply(block_varnames, is_discrete)]
    block_discrete_varnames <- block_varnames[sapply(block_varnames, is_discrete)]
    if (length(block_continuous_varnames) > 0) {
      # apply PCA to continuous data
      npc_adj <- min(npc, length(block_continuous_varnames))
      X_cont <- df[, block_continuous_varnames]
      # impute the df by means for each col, the block-level is coarse, so the bias 
      #   introduced by imputation should be smaller.
      if (length(block_continuous_varnames) > 1) {
        X_cont <- apply(X_cont, 2, function(x) {x[is.na(x)] <- mean(x, na.rm = TRUE); x})
      } else {
        X_cont[is.na(X_cont)] <- mean(X_cont, na.rm = TRUE)
      }
      
      X_cont <- prcomp(X_cont, center = TRUE, scale. = TRUE)$x[, 1:npc_adj]
      block_df <- cbind(block_df, X_cont)
      block_df_varnames <- c(block_df_varnames, paste("G", l, ".C", 1:npc_adj, sep=""))
    }
    if (length(block_discrete_varnames) > 0) {
      # max aggregation on discrete data
      X_disc <- c()
      for (i in 1:num_discrete_per_step) {
        disc_names <- block_discrete_varnames[sapply(block_discrete_varnames, 
                                                     function(x) endsWith(x, paste("D", i, sep = "")))]
        X_disc_sub <- df[, disc_names]
        X_disc_sub[is.na(X_disc_sub)] <- 0 # NA means no abnormality occurs, so set as 0
        if (length(disc_names) > 1) {
          X_disc <- cbind(X_disc, apply(X_disc_sub, 1, max))
        } else {
          X_disc <- cbind(X_disc, X_disc_sub)
        }
        
      }
      block_df <- cbind(block_df, X_disc)
      block_df_varnames <- c(block_df_varnames, paste("G", l, ".D", 1:num_discrete_per_step, sep=""))
    }
  }
  block_df <- as.data.frame(block_df)
  colnames(block_df) <- block_df_varnames
  
  # remove non-informative columns
  block_df <- block_df[, which(apply(block_df, 2, sd) != 0)]
  
  block_df
}


#' Get block-level mappings from the simulated data
#' 
get_true_block_level_mappings <- function(sim.data) {
  reverse_blocks <- c()
  for (i in 1:length(sim.data$blocks)) {
    for (j in sim.data$blocks[[i]]) {
      reverse_blocks[j] <- i  
    }
  }
  
  true_block_level_mappings <- sim.data$arcs %>% 
    mutate(from = reverse_blocks[extract_seq(from)], to = reverse_blocks[extract_seq(to)]) %>% 
    distinct_all() %>% 
    filter(from != to) %>% 
    select(from, to)
}  


#' Get block-level distance constraints from the blocks information.
#' 
#' This is to take into consideration of the step-level distance constraints.
get_block_distance_constraints <- function(block_df, blocks, max_step_distance = 30) {
  cnames <- colnames(block_df)
  bl <- c()
  for (i in 1:(length(blocks) - 1)) {
    steps_in_block <- blocks[[i]]
    last_step_in_block <- steps_in_block[length(steps_in_block)]
    for (j in (i + 1):length(blocks)) {
      if (blocks[[j]][1] - last_step_in_block >= max_step_distance) {
        for (name in cnames[sapply(cnames, extract_seq) == i]) {
          bl <- rbind(bl, bnlearn::tiers2blacklist(list(cnames[sapply(cnames, extract_seq) >= j], name)))
        }
        break
      }
    }
  }
  bl
}

#================ Configuration settings =================
n <- 10000
p <- 50
k <- 20
#================ Configuration settings =================

b <- 1
max_step_distance <- k+1
missing_rate <- 0.0 # 0.1, 0.05, 0.02
seed <- 10
degree<-4
block_tightness = 0.7

sim.data <- data.simulate(n = n, p = p, k = k, b = b, imb = 0, num_discrete_per_step=p+1,
                              degree = degree,

                              block_tightness = block_tightness, 
                              max_step_distance = max_step_distance,
                              missing_rate = missing_rate, 
                              seed = seed)

dir_name <- "data"
dir.create(dir_name)

path_name = paste(dir_name,"/",sep="")

write.csv(sim.data$df, file = paste(path_name,"X.csv",sep=""),row.names = FALSE)
write.csv(sim.data$arcs, file = paste(path_name,"arcs.csv",sep=""),row.names = FALSE)
write.csv(sim.data$adj, file = paste(path_name,"A.csv",sep=""))
