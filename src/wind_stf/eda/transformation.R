library(beepr)

library(hdf5r)    # for handling hdf5 files
library(data.table)

library(sf)
library(raster)
library(dplyr)
# library(spData)
# library(spDataLarge)
library(tidyr)
library(eurostat)
library(tmap)    # for static and interactive maps
# library(leaflet) # for interactive maps
# library(mapview) # for interactive maps

#library(shiny)   # for web applications
library(tmaptools)

library(TSstudio)
library(lubridate)
library(xts)

library(units)

# Defining constants ===========================================
# Uber Color Schemes
# colorscheme_uber <- data.table(
#  terrain = c("waters","land","highway","roads","parks","airport"),
#   dark = c("#12232f", "#09101d", "#304a5c", "#223949", "#041922", "#131926"),
#   light = c("#dbe2e6", "#ebf0f0", "#f5fafa", "#f0 f5f5", "#e6eae9", "#d9dede"),
# )
# rownames(colorscheme_uber) <- colorscheme_uber$terrain

golden_ratio <- (1+sqrt(5))/2


# Loading data ===========================================
# Ensure current working dir is data/01_raw
tryCatch(
  {
  setwd("./data/01_raw")
  },
  error = function( err ){
    print("Already in the expected working dir. :)")}
)

LoadGeodata <- function(){
  # Get geospatial data: first try from local files, then download from GISCO if necessary
  tryCatch(
    {
      # attempt to read file locally
      # use assign so you can access the variable outside of the function
      assign("geodata", st_read("./geospatial/geodata.shp"), envir = .GlobalEnv)
      print("Loaded geodata from local storage.")
    },
    error = function( err ){
      print("Could not read data from current directory, attempting download...")
      tryCatch(
        {
          assign("geodata",
                 get_eurostat_geospatial(
                   output_class = "sf",
                   resolution = "1",
                   nuts_level = 3,
                   year = 2013),
                 envir = .GlobalEnv)
          st_write(geodata, "./geospatial/geodata.shp")
          print("Loaded geodata from GISCO server. Saved it locally.")
        },
        error = function( err ){
          print("Could not download geodata from GISCO. Aborting execution.")
        }
      )
    }
  )

  return(list('de' = geodata[which(geodata$CNTR_CODE == 'DE'), ],
              'eu' = geodata[geodata$CNTR_CODE %in% c("DK", "PL", "CZ", "SK", "CH", "AT", "FR", "BE", "LU", "NL"), ]))  # DE neighboring countries ))
}
geodata <- LoadGeodata()

# Load metadata: wind turbines spatial data frame
turbines.metadata <- st_as_sf(
  read.csv("metadata/wind_turbine_data.csv", sep=";"),
  coords = c("lon", "lat"),
  crs=st_crs(geodata$de))

turbines.metadata$dt <- as.Date.character(turbines.metadata$dt, tryFormats = c("%d.%m.%Y"))  # commissioning date column to standard datetime format
turbines.metadata.st <- read.csv("metadata/wind_turbine_data.csv", sep=";")

GetPowerGeneratedTS <- function(){
  power.generated <- read.csv("./power-generation/wpinfeed_inkW_nuts3_2015_utc.csv")  # TODO: drop districts not in geodata$de$NUTS_ID
  power.generated$X <- ymd_hms(power.generated$X, tz="UTC")
  rownames(power.generated) <- power.generated$X
  power.generated <- select(power.generated, -X)

  power.generated.xts <- xts(power.generated, order.by = ymd_hms(rownames(power.generated)))
  return(power.generated.xts)
}

# Preprocessing  ===========================================
districts.blacklist <- c(c('DE409', 'DE40C', 'DE403'), # districts TS which represent outliers in correlogram
                         c('DE24C', 'DE266', 'DEA2C')) # districts with zero installed capacity at 2015-01-01
turbines.metadata <- turbines.metadata[ !( turbines.metadata$NUTS_ID %in% districts.blacklist), ]
turbines.metadata.st <- turbines.metadata.st[ !( turbines.metadata.st$NUTS_ID %in% districts.blacklist), ]

# Transform a matrix upper diagonal into a vector
TransformMatrixIntoVector <- function(A){
  bool.mask <- upper.tri(A, diag = FALSE)
  V <- A[bool.mask]
  return(V)
}

GetPowerInstalled <- function(metadata=turbines.metadata){
  power.installed.deltas <- aggregate(power ~ dt + NUTS_ID, data=metadata, sum)  # installed power aggregated for sameday, same district commissionings

  power.installed <- power.installed.deltas %>%
    arrange(NUTS_ID, dt) %>%
    group_by(NUTS_ID) %>%
    mutate(power.tot = cumsum(power))

  return(power.installed)
}

GetPowerInstalledTimeSeries <- function(.power.installed=power.installed, colnames){
  blank.xts <- xts(x=NULL,
                   seq(from=min(.power.installed$dt) + hours(1),
                       to=as.Date("2015-12-31") + hours(23),
                       by="hour"))

  .power.installed <- select(.power.installed, -power)
  power.installed.xts <- read.zoo(.power.installed, split="NUTS_ID") %>%
    merge.xts(blank.xts,
              .,
              fill=na.locf) %>%
    na.fill(., fill=0) %>%
    .["20150101/20151231"]

  power.installed.xts <- power.installed.xts[, colnames]  # order columns the same way of power.generated.xts

  # tests for GetPowerInstalledTimeSeries
  # assertive.base::are_identical(sum(new.commissionings$power), max(capacity.installed$power))
  # assertive.properties::is_monotonic_increasing(capacity.installed$power)
  # sum(is.na(power.installed.xts)) == 0
  # sum(names(power.installed.xts) == names(power.generated.xts))

  return(power.installed.xts)
}

GetAllCapacityFactorsTimeSeries <- function(power.generated.xts, power.installed.xts){
  cf <- power.generated.xts/power.installed.xts
  cf <- na.fill(cf, fill=0)  # handle NAs resulting from divisions by zero (no power.installed)

  # unit test for GetCapacityFactor
  # GetColumnsContainingNA(cf, view=TRUE)

  return(cf)
}

GetColumnsContainingNA <- function(df, view=FALSE){
  if(sum(is.na(cf)) != 0) {
    columns.with.na <- df[,which(sapply(df, function(x) sum(is.na(x))) > 0)]
    columns.with.na.names <- names(columns.with.na)
    if(view) View(columns.with.na)
    return(columns.with.na.names)
  }
  else return("No NA found")
}

# checking for data loss during transformation
sum(is.na(turbines.metadata))
"DE146" %in% unique(turbines.metadata$NUTS_ID)

turbines.metadata <- turbines.metadata[which(turbines.metadata$dt < "2015-12-31"),]          # only consider turbines commissioned before 2015-12-31
turbines.metadata$NUTS_ID <- trimws(turbines.metadata$NUTS_ID)

# Load power generation data
# TODO: drop observations after 2015
# file.names <- dir("./power-generation/")
# file.paths <- paste("./power-generation/", file.names, sep="")
# power.generated <- do.call(rbind, lapply(file.paths,read.csv))

GetTurbinesCentroids <- function(){
  turbines.metadata.st <- read.csv("metadata/wind_turbine_data.csv", sep=";")
  turbines.metadata.st  <- turbines.metadata.st[ !(turbines.metadata.st$NUTS_ID %in% districts.blacklist), ]

  turbines.metadata.st$dt <- as.Date.character(turbines.metadata.st$dt, tryFormats = c("%d.%m.%Y"))  # commissioning date column to standard datetime format
  turbines.metadata.st <- turbines.metadata.st[which(turbines.metadata.st$dt < "2015-12-31"),]
  turbines.centroids <- aggregate(. ~ NUTS_ID, data= turbines.metadata.st[,c("lat", "lon", "NUTS_ID")], mean)
  rownames(turbines.centroids) <- turbines.centroids$NUTS_ID
  turbines.centroids2 <- turbines.centroids[ names(capacity.factors), ] # order centroids table the same way as capacity.factors, and corr.cf tables
  turbines.centroids <- st_as_sf(turbines.centroids,
                                 coords = c("lon", "lat"),
                                 crs=st_crs(geodata$de))
  return(turbines.centroids)
}

GetTurbinesCentroidsDT <- function(){
  turbines.metadata.st <- read.csv("metadata/wind_turbine_data.csv", sep=";")
  turbines.metadata.st  <- turbines.metadata.st[ !(turbines.metadata.st$NUTS_ID %in% districts.blacklist), ]

  turbines.metadata.st$dt <- as.Date.character(turbines.metadata.st$dt, tryFormats = c("%d.%m.%Y"))  # commissioning date column to standard datetime format
  turbines.metadata.st <- turbines.metadata.st[which(turbines.metadata.st$dt < "2015-12-31"),]
  turbines.centroids <- aggregate(. ~ NUTS_ID, data= turbines.metadata.st[,c("lat", "lon", "NUTS_ID")], mean)
  rownames(turbines.centroids) <- turbines.centroids$NUTS_ID
  turbines.centroids.dt <- turbines.centroids[ names(capacity.factors), ] %>% # order centroids table the same way as capacity.factors, and corr.cf tables
              data.table(.)
  return(turbines.centroids.dt)
}

GetTurbinesCentroidsDistances <- function(turbines.centroids){
  distances.centroids <- st_distance(turbines.centroids)
  rownames(distances.centroids) <- turbines.centroids$NUTS_ID
  colnames(distances.centroids) <- turbines.centroids$NUTS_ID
  return(distances.centroids * 1E-03 )
}

# EDA  ===========================================
# TODO: why districts in power.generated not all in geodata$de districts with power.installed>0? Hypotheses: (1) geodata$de, power.generated with different NUTS3 definition
#colnames(power.generated) %in% geodata$de[which(geodata$de$power.installed>0),]$NUTS_ID
DropBlackMailedDistricts <- function(power.generated.xts){
  return ( power.generated.xts[ , !(names(power.generated.xts) %in% districts.blacklist)] )
}

power.generated.xts <- GetPowerGeneratedTS()
power.generated.xts <- DropBlackMailedDistricts(power.generated.xts)
power.installed.xts <- GetPowerInstalled(turbines.metadata) %>%
  GetPowerInstalledTimeSeries(., colnames=names(power.generated.xts))

capacity.factors <- GetAllCapacityFactorsTimeSeries(power.generated.xts, power.installed.xts)

# Get matrix: correlation matrix for generated power
corr.cf <- cor(capacity.factors, method = "spearman")
corr.cf.pearson <- cor(capacity.factors, method = "pearson")
# Get centroid of turbines locations by district
turbines.centroids <- GetTurbinesCentroids()

# Euclidean distances between pairs of centroids
distances.centroids <- GetTurbinesCentroidsDistances(turbines.centroids)

# TS pairs: a dataframe characterising pairs of time series
district.pairs <- combn( unique( turbines.metadata$NUTS_ID ), 2 )
district.pairs.id <- paste(district.pairs[1,], district.pairs[2,], sep="-")

GetDistrictPowerInstalled <- function(id){
  return( power.installed.xts[ end(power.installed.xts), id ] )
}

# some districts were not generating any wind power for most of the period. Better not to consider them.
GetAllDistrictAvgCf <- function(){
  cf.mean <- apply( capacity.factors, MARGIN=2, mean)
  return( cf.mean )
}

GetDistrictAvgCf <- function(id){
  return( cf.mean[ id ] )
}
# min.power.installed <- apply(district.pairs, MARGIN=c(1,2), FUN=GetDistrictPowerInstalled) %>%  # get power installed for every cell in district.pairs
#                        apply( . , MARGIN=2, FUN=min)                                            # get minimum power installed in district pair pair

cf.mean <- GetAllDistrictAvgCf()
min.avg.cf <- apply(district.pairs, MARGIN=c(1,2), FUN=GetDistrictAvgCf) %>%
              apply( . , MARGIN=2, FUN=min)

# starting.dates <- GetAllDistrictFirstCommissioning()
# latest.production.start <- apply(district.pairs, MARGIN=c(1,2), FUN=GetDistrictAvgCf) %>%
#                            apply( . , MARGIN=2, FUN=min)

ts.pairs <- data.table(pairs.id=district.pairs.id,
                       id1=district.pairs[1,],
                       id2=district.pairs[2,],
                       distances= as.double( TransformMatrixIntoVector(distances.centroids) ),
                       spearman.corr=TransformMatrixIntoVector(corr.cf),
                       pearson.corr = TransformMatrixIntoVector(corr.cf.pearson),
                       # least.power=min.power.installed,
                       min.avg.cf=min.avg.cf)

power.generated.yearly <- apply(power.generated.xts, MARGIN = 2, sum)


turbines.centroids.dt <- GetTurbinesCentroidsDT()


#save(geodata,
#     turbines.centroids.dt,
#     district.pairs,
#     ts.pairs,
#     power.installed.xts,
#     power.generated.xts,
#     capacity.factors,
#     power.generated.yearly,
#     file = "../02_intermediate/eda.vars.RData")

# ===============================
# Plotting
# ===============================
# NUTS3
nuts3map <-  tm_shape(geodata$de, xlim=c(10.458-5,10.458+5)) + tm_fill(col="#223949") + tm_borders(col="#304A5C") +
  tm_shape(geodata$eu, xlim=c(10.458-6,10.458+6)) + tm_fill(col="#09101D") +
  tm_layout(bg.color = "#12232f", inner.margins = c(0, .02, .02, .02))
nuts3map
#tmap_save(nuts3map,
#          filename = paste0('../08_reporting/districs-map_',
#                            format(Sys.time(), "%Y%m%d_%H%M%S"),
#                            '.png'),
#          dpi = 600,
#          outer.margins = c(0, 0, 0, 0))


# Turbine locations: map
color.water <- "#64747b"
color.eu <- "#D9DEDE"
color.de <- "#F5FAFA"

#turbines.metadata.st['power.normalized'] <- ( turbines.metadata.st['power'] - min(turbines.metadata.st['power']) ) / ( max( turbines.metadata.st['power'] ) - min(turbines.metadata.st['power']) )
#turbines.metadata['power.normalized'] <- turbines.metadata.st['power.normalized']
turbines.metadata.st['power'] <- turbines.metadata.st['power'] * 1E-03
turbines.metadata['power'] <- turbines.metadata.st['power']

turbines.metadata['dt.year'] <- apply(data.table(turbines.metadata$dt), MARGIN=1, FUN=lubridate::year) %>%
                                sapply(., FUN=as.integer)

turbines.metadata <- turbines.metadata[order(turbines.metadata$dt.year),]

turbines.locations <-
  tm_shape(geodata$de, xlim=c(10.458-6,10.458+6)) +
  tm_fill(col=color.de) +

  tm_shape(geodata$eu, xlim=c(10.458-6,10.458+6))+
  tm_fill(col=color.eu) +

  tm_shape(turbines.metadata) +
  tm_bubbles(alpha=0.25, border.alpha = 0,
             col="dt.year",
             style="quantile",
             palette = viridisLite::inferno(n=5, direction = -1),
             title.col="Commisioning \n Year", legend.col.show = TRUE, legend.col.is.portrait = TRUE,
             size="power",
             perceptual=TRUE,
             title.size="Rated Power [MW]", legend.size.show = TRUE
  ) +
  tm_layout(bg.color = color.water,
            inner.margins = c(0, .02, .02, .02),
            legend.format=list(fun=function(x) formatC(x, digits=0, format="d")),
            legend.position = c("right", "bottom"))
  # tm_scale_bar(breaks = c(0, 50, 100), text.size = 0.7, position = c("left", "bottom"))
turbines.locations

tmap_save(turbines.locations,
          filename = paste0('../08_reporting/map_turbines_locations_',
                  format(Sys.time(), "%Y%m%d_%H%M%S"),
                  '.png'),
          dpi=600,
          outer.margins = c(0,0,0,0))

# Turbine counts: NUTS3 aggregation map
turbines.nuts3aggregation <- st_contains(geodata_de, turbines.metadata)
geodata_de$turbine.counts <- unlist(lapply(turbines.nuts3aggregation, length))
geodata_de$turbines.ids <- lapply(turbines.nuts3aggregation, unlist)

turbine.counts.map <-  tm_shape(geodata_de, xlim=c(10.458-5,10.458+5)) + tm_borders(col="#223949") + tm_fill(col="turbine.counts", breaks=c(0, 1, 10, 30, 100, 300, 762) , palette = get_brewer_pal("Blues", n = 6, contrast = c(0, 1))) +
  tm_shape(geodata_eu, xlim=c(10.458-6,10.458+6)) + tm_fill(col="#09101D") +
  tm_layout(bg.color = "#12232f", inner.margins = c(0, .02, .02, .02), legend.text.color = "#f5fafa", legend.title.color="#f5fafa", legend.position = c("left", "top"))
turbine.counts.map
#tmap_save(turbine.counts.map,
#          filename = paste0('../08_reporting/map_turbine_counts_',
#                            format(Sys.time(), "%Y%m%d_%H%M%S"),
#                            '.png'),
#          dpi = 600,
#          outer.margins = c(0, 0, 0, 0))