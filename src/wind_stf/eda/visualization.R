library(ggplot2) # tidyverse data visualization package
library(plotly)
library(xts)
library(TSstudio)
library(data.table)
library(lubridate)
library(dplyr)
library(gridExtra)

# Ensure current working dir is data/08_reporting
tryCatch(
  {
  setwd("./data/08_reporting")
  },
  error = function( err ){
    print("Already in the expected working dir. :)")}
)

golden_ratio <- (1+sqrt(5))/2

load("../02_intermediate/eda.vars.RData")

### What is a typical value for yearly WPG-kW? What districts present approx this value?
# A: median = 79243199 [kW]; DEB22 79549998 [kW]
GetTotalCommissionings <- function(.nuts_id, .power.installed.xts=power.installed.xts){
  return( length( unique(.power.installed.xts[, .nuts_id]) ) )
}

GetChangeInInstalledPower <- function(){
  capacity.initial <- data.table( power.installed.xts[start(power.installed.xts),] )
  capacity.end <- power.installed.xts[end(power.installed.xts),]

  capacity.DT <- data.table(
    nuts.id = names(capacity.initial),
    capacity.initial = as.numeric(capacity.initial),
    capacity.end = as.numeric(capacity.end)
  )

  capacity.DT[, 'delta'] <- 100 * ( capacity.DT[, 'capacity.end'] - capacity.DT[, 'capacity.initial'] ) / capacity.DT[, 'capacity.initial']
  return(capacity.DT)
}

GetDeltaInstalledPower <- function (.nuts.id){
  return(capacity.DT[nuts.id==.nuts.id, 'delta'])
}

capacity.DT <- GetChangeInInstalledPower()

PlotDensityYearlyWPGkWh <- function(.nuts_id, .power.installed.xts=power.installed.xts){
  power.generated.yearly.DT <- data.table(
    wpg.yearly = power.generated.yearly,
    nuts.id = names(power.generated.yearly),
    total.commissionings = sapply(names(power.generated.yearly), FUN=GetTotalCommissionings),
    # capacity.delta = sapply(names(power.generated.yearly), FUN=GetDeltaInstalledPower)
  )

  power.generated.yearly.DT[, 'capacity.delta'] = as.numeric ( sapply(names(power.generated.yearly), FUN=GetDeltaInstalledPower) )
   p.wpg.yearly <- ggplot(data=power.generated.yearly.DT, aes(x=wpg.yearly)) +
    geom_density() +
    geom_rug(alpha=0.8, aes(color=capacity.delta), show.legend = TRUE) +
    #geom_vline(xintercept = median(p.wpg.yearly), colour="green", linetype = "dashed") +
    #geom_text(aes(x=median(p.wpg.yearly), label="median\n", y=0.05), colour="black", alpha=0.7, angle=90, text=element_text(size=11)) +
    #xlab("2015 Average yearly power generation by district [kW]") +
    #ylab("Estimated Density [-]") +
    scale_x_log10()

  ggplotly(p.wpg.yearly, text = c('nuts.id', 'capacity.delta'))


  if (!plotly){
    p.wpg.yearly
  }
  else {
      plot_ly(data.table(val=power.generated.yearly),
        # color=ts.pairs$min.avg.cf,
        type = 'density',
        text = index,
        alpha=0.2,)
  }
}
# PlotDensityYearlyWPGkWh()

### How does a typical WPG-kWh time series look like?
PlotTypicalWPGkwh <- function() {
  power.generated.daily.dt <- data.table(power.generated.xts * 1E-06) %>%
    mutate(day = as.Date(index(power.generated.xts), format = "%Y-%m-%d")) %>%
    aggregate(. ~ day, data = ., FUN = sum) %>%
    data.table(.)

  power.capacity.daily.dt <- data.table(power.installed.xts * 1E-06) %>%
    mutate(day = as.Date(index(power.installed.xts), format = "%Y-%m-%d")) %>%
    aggregate(. ~ day, data = ., FUN = sum) %>%
    data.table(.)

  setkey(power.generated.daily.dt, day)
  setkey(power.capacity.daily.dt, day)
  merged.deb22.dt <- power.generated.daily.dt[power.capacity.daily.dt, nomatch = 0]

  p.wpg.daily.kwh.xts <- ggplot(power.generated.daily.dt, aes(x = day)) +
    geom_line(aes(y = merged.deb22.dt$i.DE71D, color = 'capacity'), size = 0.4, alpha = 1.0) +
    geom_line(aes(y = merged.deb22.dt$DE71D, color = 'generation'), size = 0.4, alpha = 1.0) +
    scale_color_manual(name = NULL,
                       values = c('capacity' = '#737373', 'generation' = '#F98C0AFF')) +
    xlab("Date") +
    ylab("Daily Generation [GWh]") +
    # scale_y_continuous(name, breaks, labels, limits, trans)
    scale_y_continuous(breaks = c(0, 0.4, 0.8, 1.2), labels = c(0, 0.4, 0.8, 1.2), limits=c(0, 1.3)) +
    # scale_y_log10() +
    theme(axis.title.x = element_text(size = rel(0.75)),
          axis.title.y = element_text(size = rel(0.75)),
          legend.position = c(0.10, 0.80),
          legend.background = element_rect(fill = alpha('white', 0.0), color = 'transparent', size = 0.75))#,
          #legend.key = element_rect(fill = 'white', colour = 'white', size=0.5))

  p.wpg.daily.kwh.density <- ggplot(power.generated.daily.dt, aes(y = DE71D)) +
    geom_density(aes(x =  ..scaled..)) +
    geom_rug(alpha = 0.3) +
    geom_hline(yintercept = median(power.generated.daily.dt$DE71D), color = "#F98C0AFF", alpha = 0.6) +
    geom_text(aes(y = median(power.generated.daily.dt$DE71D), label = "median\n", x = 0.25), color = "#F98C0AFF", alpha = 0.7, angle = 0, text = element_text(size = rel(0.75))) +
    # scale_y_log10() +
    xlab("Estimated Density") +
    scale_y_continuous(breaks = c(0, 0.4, 0.8, 1.2), labels = c(0, 0.4, 0.8, 1.2), limits=c(0, 1.3)) +
    scale_x_continuous(breaks=c(0.0, 0.5, 1.0)) +
    theme(legend.position = "none",
          axis.text.y = element_blank(),
          axis.ticks.y = element_blank(),
          axis.title.y = element_blank(),
          axis.title.x = element_text(size = rel(0.75))
    )
  p.wpg.daily.kw <- grid.arrange(p.wpg.daily.kwh.xts, p.wpg.daily.kwh.density, ncol = 2, nrow = 1, widths = c(5, 1))

  ggsave(
    filename = paste0('../08_reporting/wpg-daily-typical-ts_',
                      format(Sys.time(), "%Y%m%d_%H%M%S"),
                      '.png'),
    plot = p.wpg.daily.kw,
    scale = golden_ratio,
    width = 210 / golden_ratio,
    height = (210 / golden_ratio) / (3 * golden_ratio),
    units = 'mm',
    dpi = 300,
    limitsize = TRUE,
  )
}
# PlotTypicalWPGkwh()

### How does a typical WPG-CF time series look like?
PlotTypicalWPGcf <- function() {
    capacity.factors.dt <- data.table(capacity.factors) %>%
    mutate(day = as.Date(index(capacity.factors), format = "%Y-%m-%d")) %>%
    aggregate(. ~ day, data = ., FUN = mean) %>%
    data.table(.)

  p.wpg.daily.cf <- ggplot(capacity.factors.dt, aes(x = day)) +
    geom_line(aes(y = capacity.factors.dt$DE71D), size = 0.4, color = '#F98C0AFF', alpha = 1.0) +
    ylim(0.00, 1.00) +
    xlab("Date") +
    ylab("Daily Capacity Factor [-]") +
    theme(axis.title.x = element_text(size = rel(0.75)),
          axis.title.y = element_text(size = rel(0.75)),)

   p.wpg.daily.cf.density <- ggplot(capacity.factors.dt, aes(y = DE71D)) +
    geom_density(aes(x =  ..scaled..)) +
    ylim(0.00, 1.00) +
    geom_rug(alpha = 0.3) +
    # geom_hline(aes(yintercept = median( as.numeric( capacity.factors.dt[, 'DE71D']) ), color = "median"), alpha = 0.6) +
    # geom_text(aes(y = median(capacity.factors.dt$DEB22), label = "median\n", x = 0.2), color = "#F98C0AFF", alpha = 0.7, angle = 0, hjust = 0, text = element_text(size = 9)) +
    #geom_hline(aes(yintercept = mean( as.numeric( capacity.factors.dt[, 'DE71D'] ) ), color = "mean"), alpha = 0.6) +
    # geom_text(aes(y = mean(capacity.factors.dt$DEB22), label = "mean\n", x = 0.2), color = "#BB3754FF", alpha = 0.7, angle = 0, hjust = 0, text = element_text(size = 9)) +
    scale_color_manual(name = NULL, values = c(median = "#F98C0AFF", mean = "#BB3754FF")) +
    xlab("Estimated Density") +
    scale_x_continuous(breaks=c(0.0, 0.5, 1.0)) +
    theme(axis.text.y = element_blank(),
          axis.ticks.y = element_blank(),
          axis.title.y = element_blank(),
          axis.title.x = element_text(size = rel(0.75)),
          legend.position = c(0.6, 0.8),
          legend.background = element_rect(fill = alpha('white', 0.0), color = 'transparent', size = 0.75),
          legend.key = element_rect(fill = 'transparent', colour = 'transparent'))
  p.wpg.daily.cf <- grid.arrange(p.wpg.daily.cf, p.wpg.daily.cf.density, ncol = 2, nrow = 1, widths = c(5, 1))

  ggsave(
    filename = paste0('../08_reporting/wpg-cf-daily-typical-ts_',
                      format(Sys.time(), "%Y%m%d_%H%M%S"),
                      '.png'),
    plot = p.wpg.daily.cf,
    scale = golden_ratio,
    width = 210 / golden_ratio,
    height = (210 / golden_ratio) / (3 * golden_ratio),
    units = 'mm',
    dpi = 300,
    limitsize = TRUE,
  )
}
PlotTypicalWPGcf()

### How do ALL WPG-CF time series look like?
PlotAllWpgCf <- function() {
  capacity.factors.DT <- data.table(capacity.factors) %>%
    mutate(day = as.Date(index(capacity.factors), format = "%Y-%m-%d")) %>%
    aggregate(. ~ day, data = ., FUN = mean) %>%
    data.table(.)

  capacity.factors.DT.long <- melt(capacity.factors.DT, id = "day")  # convert to long format
  lines.qty <- dim(capacity.factors.DT)[2] -1

  curves.colors <- rep('#848c86', lines.qty)
  curves.colors[122] <- '#F98C0AFF'  # highlight DE71D

  curves.alphas<- rep(0.01, lines.qty)
  curves.alphas[122] <- 1.0  # highlight DE71D

  p.wpg.daily.cf.all <- ggplot(data=capacity.factors.DT.long, aes(x=day, y=value, color=variable, alpha=variable)) +
    geom_line(show.legend = FALSE) +
    xlab("Date") +
    ylab("Daily Capacity Factor [-]") +
    ylim(0.00, 1.00) +
    scale_color_manual(values=curves.colors) +
    scale_alpha_manual(values=curves.alphas) +
    theme(axis.title.x = element_text(size = rel(0.75)),
          axis.title.y = element_text(size = rel(0.75)),)

  ggsave(
    filename = paste0('../08_reporting/wpg-cf-daily-all-ts_',
                      format(Sys.time(), "%Y%m%d_%H%M%S"),
                      '.png'),
    plot = p.wpg.daily.cf.all,
    scale = golden_ratio,
    width = 210 / golden_ratio,
    height = (210 / golden_ratio) / (3 * golden_ratio),
    units = 'mm',
    dpi = 300,
    limitsize = TRUE,
  )
}
# PlotAllWpgCf()

### How are hourly, distrital time series temporarily correlated?
# We use a pearson cross-correlation function to assess the temporal correlation between the hourly time series.
# This requires the choice of a maximum lag.
# Assuming a nearly critical condition in which an air mass travels in a straight line between the producing districts
# most distant from one another (max(ts.pairs$distances)=800km) with an average streamline velocity the minimum
# found across the land, assumed to be 7 km/h. We then use this modeled critical wind trajectory to
# determine a reasonable upper bound for the maximum lag considered for evaluating the cross-correlation function: 800/7 ~= 114h.
# Figure \ref{fig:single-pair-ccf} shows the cross-correlation for the most distant pair of producing districts.
# Correlations tend to fall below ??? as the lag surpasses ??? hours.

# Furthermore, in order to attain an informative overview of cross-correlations between districts time series,
# we restrict ourselves to those pairs presenting no more than 400 km of Euclidean distance,
# which tend to present spearman correlations of at least 0.5 (fig. \ref{fig:spatial-analysis).
# This results in about 75% of all producing district pairs remaining for the temporal analysis.
# When assessing the correlation coefficients at different lags, we define as a positive lag the condition
# where the northernmost district in the pair is lagged in relation to the southernmost one in the pair.
# This follows the consideration that, in Germany, wind flows from SSW (240°) tend to predominate both in
# frequency and speed and, in consequence, also in power \cite{windatlas}.

PlotAllCorrelogramsSuperposed <- function(){
  #ts.pairs.near <- FilterOutDistantPairs(ts.pairs)
  #ts.pairs.near.orderedSN <- OrderAllPairsByLatitude(ts.pairs.near)

  #save(ts.pairs.near.orderedSN,
  #     file = "../02_intermediate/ts.pairs.near.orderedSN.RData")
  load("../02_intermediate/ts.pairs.near.orderedSN.RData")

  #ccfs.DT <- GetAllCCFs(ts.pairs.near.orderedSN)
  #save(ccfs.DT,
  #     file = "../02_intermediate/ts.pairs.ccfs.DT.orderedSN.RData")
  load("../02_intermediate/ts.pairs.ccfs.DT.orderedSN.RData")


  ccfs.DT$lag <- seq(-72, 72, 1)
  # ccfs.DT$dist.normalized <- MinMaxNormalizeVector(ts.pairs.near.orderedSN[, 'distances'])

  ccfs.long.DT <- melt(ccfs.DT, id = "lag")  # convert to long format
  ccfs.mini.long.DT <- ccfs.long.DT[1:(145 * 60),]
  # ccfs.mini.long.DT <- melt(data=ccfs.DT, id.vars=c("lag", 'dist.normalized'), vars=c("DE145-DE114", "DE145-DE146", "DE145-DE132", "DE145-DE12A", "DE145-DE133"))  # convert to long format
  # ccfs.mini.long.DT$dist.normalized <- MinMaxNormalizeVector(ts.pairs.near.orderedSN[, 'distances'])

  lines.qty <- dim(ccfs.long.DT)[1] / 145
  #p.ccfs <- ggplot(data=ccfs.long.DT,
  #       aes(x=lag, y=value, color=variable)) +
  #    geom_line(show.legend = FALSE, alpha=0.001) +
  #    xlab("Lag [h]") +
  #    ylab("Pearson CCF [-]") +
  #    scale_x_continuous(breaks = c(-72, -48, - 24, 0, 24, 48, 72)) +
  #    scale_color_manual(values=rep('#F98C0AFF', lines.qty))
  #p.ccfs
  #
  #ggsave(
  #  filename = paste0('../08_reporting/ccf-all_',
  #                  format(Sys.time(), "%Y%m%d_%H%M%S"),
  #                  '.png'),
  #  plot = p.ccfs,
  #  scale = golden_ratio,
  #  width = 210/golden_ratio,
  #  height = (210/golden_ratio)/golden_ratio,
  #  units = 'mm',
  #  dpi = 300,
  #  limitsize = TRUE,
  #)

  p.ccf.sample <- ggplot(data = ccfs.long.DT[variable == 'DE145-DEB22']) +
    geom_segment(aes(xend = lag, x = lag, yend = 0, y = value), show.legend = FALSE, alpha = 1, color = '#F98C0AFF') +
    xlab("Lag [h]") +
    ylab("Pearson CCF [-]") +
    scale_x_continuous(breaks = c(-72, -48, -24, 0, 24, 48, 72)) +
    scale_y_continuous(breaks = c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) +
    ylim(0, 1)
  p.ccf.sample

  ggsave(
    filename = paste0('../08_reporting/ccf-sample_',
                      format(Sys.time(), "%Y%m%d_%H%M%S"),
                      '.png'),
    plot = p.ccf.sample,
    scale = golden_ratio,
    width = 210 / golden_ratio,
    height = (210 / golden_ratio) / golden_ratio,
    units = 'mm',
    dpi = 300,
    limitsize = TRUE,
  )

  #ggplot(data=ccfs.DT)

  #ccf.object <- ccf(rank(as.ts(capacity.factors[, 'DED43'])),
  #                  rank(as.ts(capacity.factors[, 'DED53'])),
  #                  lag = 6,
  #                  plot = TRUE)
  #
  #plot(x=seq(-6,6), y=ccf.object$acf)
  #
  #ts.pairs$ccf =  mapply(ccf, capacity.factors, MoreArgs = list(lag.max = 72))
  #  ccf(rank(as.ts(capacity.factors[,'DEA56'])), rank(as.ts(capacity.factors[,'DEA59'])), lag.max = 72)
}

FilterOutDistantPairs <- function(ts.pairs){
  ts.pairs.near <- ts.pairs[distances < 400E+3, ]
  print(paste0(100*dim(ts.pairs.near)[1]/dim(ts.pairs)[1], '% of original pair entries remain.'))
  return(ts.pairs.near)
}

OrderPairByLatitude <- function(ids_pair){
  id1 <- ids_pair[1]
  id2 <- ids_pair[2]

  # has.switched <- FALSE
  ordered.id_pair <- c(id1, id2)
  if (turbines.centroids.dt[NUTS_ID==id1, lat] > turbines.centroids.dt[NUTS_ID==id2, lat]){
     ordered.id_pair <- c(id2, id1)
  }
  return(ordered.id_pair)
}

OrderAllPairsByLatitude <- function(ts.pairs){
  ordered.id_pairs <- apply(ts.pairs[, c('id1', 'id2')], MARGIN=1, FUN=OrderPairByLatitude)
  ts.pairs.ordered <- ts.pairs
  ts.pairs.ordered$id1 <- ordered.id_pairs[1, ]
  ts.pairs.ordered$id2 <- ordered.id_pairs[2, ]
  return(ts.pairs.ordered)
}

GetCCF <- function(id1, id2){
  ccf.object <- ccf(rank(as.ts(capacity.factors[, id1])),
                    rank(as.ts(capacity.factors[, id2])),
                    lag.max = 72,
                    plot = FALSE)
  return(data.table(ccf.object$acf))
}

GetAllCCFs <- function(ts.pairs){
  ccfs.DT <- mapply(FUN=GetCCF, ts.pairs$id1, ts.pairs$id2) %>%
    do.call(rbind, .) %>%
    t(.) %>%
    data.table(.)
  names(ccfs.DT) <- ts.pairs$pairs.id
  return(ccfs.DT)
}

MinMaxNormalizeVector <- function(x){
  x.normalized <- (x-min(x))/(max(x)-min(x))
  return(as.vector(x.normalized))
}
distn <- data.table(
  dist.normalized= MinMaxNormalizeVector(ts.pairs.near.orderedSN[, 'distances']),
  pair.id = ts.pairs.near.orderedSN[, 'pairs.id'])

GetNormalizedDistance <- function (.pairs.id){
  return (distn[pair.id.pairs.id==.pairs.id, 'dist.normalized.distances'])
}


### How dependent are the TS from different districs: correlation btw. pair of TS vs centroid distance
PlotCorrelationsVsDistances <- function (){
  #plot_ly(x=ts.pairs$distances,
  #        y=ts.pairs$spearman.corr,
  #        color=ts.pairs$min.avg.cf,
  #        type = 'scatter',
  #        text = ts.pairs$pairs.id,
  #        alpha=0.2,)
  p <- ggplot(data=ts.pairs, aes(x=distances, y=spearman.corr)) +
    geom_point(alpha = 0.1) +
    xlab("Euclidean Distance [km]") +
    ylab("Spearman Correlation [-]")
  p
  ggsave( '../08_reporting/correlation-spearman-vs-distance.png',
    plot = p,
    scale = golden_ratio,
    width = 210/golden_ratio,
    height = (210/golden_ratio)/golden_ratio,
    units = 'mm',
    dpi = 300,
    limitsize = TRUE,
  )

  p <- ggplot(data=ts.pairs, aes(x=distances, y=pearson.corr)) +
    geom_point(alpha = 0.1) +
    xlab("Euclidean Distance [km]") +
    ylab("Pearson Correlation [-]")
  p
  ggsave( '../08_reporting/correlation-pearson-vs-distance.png',
    plot = p,
    scale = golden_ratio,
    width = 210/golden_ratio,
    height = (210/golden_ratio)/golden_ratio,
    units = 'mm',
    dpi = 300,
    limitsize = TRUE,
  )
   # ggplotly(p)
}

# iea.dt <- read.csv("data/08_reporting/presentation/onshore-wind-power-generation-in-the-sustainable-development-scenario-2000-2030.csv", sep=";")

#PlotCorrelationsVsDistances()
