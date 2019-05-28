# Title: C2T4 Discover Associations Between Products

# Last update: 2019.01.26

# File: ProductAssociation.R
# Project name: Market Basket Analysis (MBA)


###############
# Project Notes
###############

# Summarize project:
# Identify purchasing patterns that will provide insight into Electronidex's clientele.

# Summarize top model and/or filtered dataset
# The top model was model_name used with ds_name.


###############
# Housekeeping
###############

# Clear objects if necessary
rm(list = ls())

# get working directory
getwd()
?getwd  # get help
# set working directory
setwd("")
dir()


################
# Load packages
################

install.packages("arules")
install.packages("caTools")
install.packages("prabclus")
install.packages("DEoptimR")
install.packages("arulesViz")
install.packages("DiceDesign")
install.packages("trimcluster")
library(caret)
library(corrplot)
library(C50)
library(doParallel)
library(mlbench)
library(readr)
library(parallel)
library(plyr)
#library(knitr)
library(arules)
library(caTools)
library(prabclus)
library(DiceOptim)
library(DiceDesign)
library(trimcluster)
library(arulesViz)


#####################
# Parallel processing
#####################

# NOTE: Be sure to use the correct package for your operating system. 
# Remember that all packages should loaded in the 'Load packages' section.

#--- for Win ---#
detectCores()  # detect number of cores
cl <- makeCluster(2)  # select number of cores
registerDoParallel(cl) # register cluster
getDoParWorkers()  # confirm number of cores being used by RStudio
# Stop Cluster. After performing your tasks, make sure to stop your cluster. 
stopCluster(cl)


###############
# Import data
##############

#--- Load raw datasets ---#

## Load Train/Existing data (Dataset 1)
electTrans <- read.transactions("ElectronidexTransactions2017.csv",
                                sep=",",
                                rm.duplicates = TRUE,
                                format = "basket")


################
# Evaluate data
################

#--- Dataset 1 ---#
summary(electTrans)
#transactions as itemMatrix in sparse format with
#9833 rows (elements/itemsets/transactions) and
#125 columns (items) and a density of 0.03506172 

#most frequent items:
#  iMac                HP Laptop CYBERPOWER Gamer Desktop            Apple Earpods        Apple MacBook Air 
#2519                     1909                     1809                     1715                     1530 
#(Other) 
#33622 

#element (itemset/transaction) length distribution:
#  sizes
#0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16   17   18   19   20   21   22   23   25   26   27 
#2 2163 1647 1294 1021  856  646  540  439  353  247  171  119   77   72   56   41   26   20   10   10   10    5    3    1    1    3 
#29   30 
#1    1 

#Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#0.000   2.000   3.000   4.383   6.000  30.000 

#includes extended item information - examples:
#                            labels
#1 1TB Portable External Hard Drive
#2 2TB Portable External Hard Drive
#3                   3-Button Mouse

# View the 10 transactions
inspect(electTrans[1:10])
length(electTrans) # Number of transactions.
size(electTrans) # Number of items per transaction
LIST(electTrans[1:10]) # Lists the transactions by conversion (LIST must be capitalized)
itemLabels(electTrans)# To see the item labels

# Check frequcny of item by index
itemFrequency(electTrans[,1])
#1TB Portable External Hard Drive 
#0.002745297 

# plot

# BARCHART Show items that have support >= 10%, and top 5
itemFrequencyPlot(electTrans, support=0.1)
itemFrequencyPlot(electTrans, topN=5)

# Show top 5 most frequently occuring items, with graph labels
itemFrequencyPlot(electTrans, 
                  topN = 5, 
                  main="Top 5 Most Frequently Occuring Items in Transactions", 
                  ylab="Item Frequency (support)", 
                  xlab="Items")

# Check support of all items
itemFrequency(electTrans, type="relative")

image(electTrans)
image(sample(electTrans, 1000))



##############
# Build model
##############

# Default values: support = 0.1, confidence = 0.8
m1 <- apriori(electTrans)
m1
#set of 0 rules

# Support = 0.05, confidence = 0.5
m2 <- apriori(electTrans, 
              parameter = list(support=0.05, confidence=0.5, minlen = 2))
m2
#set of 0 rules


# Support = 0.05, confidence = 0.25
m3 <- apriori(electTrans, 
              parameter = list(support=0.01, 
                               confidence=0.25, 
                               minlen = 2))
m3
#set of 203 rules

summary(m3)
#set of 203 rules

#rule length distribution (lhs + rhs):sizes
#2   3 
#102 101 

#Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#2.000   2.000   2.000   2.498   3.000   3.000 

#summary of quality measures:
#  support          confidence          lift      
#Min.   :0.01007   Min.   :0.2500   Min.   :1.155  
#1st Qu.:0.01108   1st Qu.:0.3060   1st Qu.:1.592  
#Median :0.01474   Median :0.3607   Median :1.850  
#Mean   :0.01806   Mean   :0.3721   Mean   :1.910  
#3rd Qu.:0.02100   3rd Qu.:0.4275   3rd Qu.:2.184  
#Max.   :0.07555   Max.   :0.6023   Max.   :3.360  
#       count      
#Min.   : 99.0  
#1st Qu.:109.0  
#Median :145.0  
#Mean   :177.7  
#3rd Qu.:206.5  
#Max.   :743.0  

#mining info:
#  data ntransactions support confidence
#electTrans          9835    0.01       0.25

inspect(m3[1:2])
#     lhs                                            rhs            support confidence     lift count
#[1] {Otium Wireless Sports Bluetooth Headphone} => {HP Laptop} 0.01006609  0.3897638 2.008029    99
#[2] {Logitech Keyboard}                         => {iMac}      0.01067616  0.3860294 1.507185   105

# HP Laptop is 2x more likely to be purchased with Otium Wireless Sports Bluetooth Headphone
# than it is to be purchased by itself

# Similarly, iMac is 1.5x more likely to be purchased with Logitech Keyboard
# than it is to be purchased by itself

inspect(sort(m3, by="lift")[1:10])
#     lhs                                    rhs                 support    confidence lift     count
#[1]  {Acer Aspire,HP Laptop}             => {ViewSonic Monitor} 0.01077783 0.3706294  3.359576 106  
#[2]  {Acer Aspire,ViewSonic Monitor}     => {HP Laptop}         0.01077783 0.6022727  3.102856 106  
#[3]  {Dell Desktop,HP Laptop}            => {ViewSonic Monitor} 0.01525165 0.3393665  3.076193 150  
#[4]  {ASUS Chromebook}                   => {ViewSonic Monitor} 0.01748856 0.3333333  3.021505 172  
#[5]  {Acer Desktop,HP Laptop}            => {Dell Desktop}      0.01240468 0.3973941  2.965380 122  
#[6]  {Dell Desktop,ViewSonic Monitor}    => {HP Laptop}         0.01525165 0.5747126  2.960869 150  
#[7]  {Acer Aspire,HP Laptop}             => {Dell Desktop}      0.01108287 0.3811189  2.843933 109  
#[8]  {HP Laptop,iMac}                    => {ViewSonic Monitor} 0.02369090 0.3135935  2.842574 233  
#[9]  {HP Laptop,Lenovo Desktop Computer} => {ViewSonic Monitor} 0.01403152 0.3039648  2.755293 138  
#[10] {Dell Desktop,HP Laptop}            => {Acer Desktop}      0.01240468 0.2760181  2.709220 122    

m3.sorted <- sort(m3, by="lift")

plot(m3, jitter = 1)
plotly_arules(m3, method = "scatterplot", measure = c("support", "confidence"), 
              shading = "lift", max = 1000)

is.redundant(m3.sorted)

# Find redundant rules
subset.matrix <- is.subset(m3.sorted, m3.sorted, sparse = FALSE)
subset.matrix[lower.tri(subset.matrix, diag=T)] <- NA
redundant <- colSums(subset.matrix, na.rm=T) >= 1
count(redundant)

# Remove redundant rules
m3.pruned <- m3[!redundant]
inspect(sort(m3.pruned, by="lift")[1:5])

m3.pruned.sorted <- sort(m3.pruned, by="lift")

plot(m3.pruned.sorted, jitter = 1)
plotly_arules(m3.pruned.sorted, method = "scatterplot", measure = c("support", "confidence"), 
              shading = "lift", max = 1000)

plot(m3.pruned.sorted[1:10], measure="lift", shading="support", method="graph", control=list(type="items"))

plot(m3.pruned.sorted, method="two-key plot")

plot(m3.pruned.sorted,
     measure=c("support", "lift"),
     shading = "confidence")
