library(gamlr)
library(AUC)
library(xgboost)
library(caret)
library(methods)
library(data.table)
library(magrittr)
library(Ckmeans.1d.dp)

trainRaw<-read.csv("train.csv")
testRaw<-read.csv("test.csv")

############################
#
#  CLEANING
#
############################

y<-trainRaw$TARGET

all<-rbind(trainRaw[,-371], testRaw)
all$ID<-0
## remove columns where all values are zero
dim(all)
all<-all[,sapply(all, function(v) var(v, na.rm=TRUE)!=0)]
dim(all)

## there aren't any columns with missing values
sapply(all, function(x) any(is.na(x)))


############################
#
#    FEATURE EXPLORATION
#
############################

## there aren't many folks with a TARGET of 1
hist(trainRaw$TARGET)

############################
#
#   Sparse Matrix Construction
#
############################


train<-all[1:nrow(trainRaw),]
train<-sparse.model.matrix(~.-1, data=train)

test<-all[(nrow(train)+1):nrow(all),]
test<-sparse.model.matrix(~.-1, data=test)

############################
#
#   XGBOOST
#
############################

param <- list("objective" = "binary:logistic",
              "eval_metric" = "logloss")

cv.nround <- 5
cv.nfold <- 3

bst.cv = xgb.cv(param=param, data = train, label = y, 
                nfold = cv.nfold, nrounds = cv.nround)

nround = 50
bst = xgboost(param=param, data = train, label = y, nrounds=nround)

model <- xgb.dump(bst, with.stats = T)
model[1:10]

# Get the feature real names
names <- dimnames(train)[[2]]

# Compute feature importance matrix
importance_matrix <- xgb.importance(names, model = bst)

# Nice graph
xgb.plot.importance(importance_matrix[1:10,])

############################
#
#   PREDICTION
#
############################


pred = predict(bst, test)
submit <- data.frame(ID = testRaw$ID, TARGET = pred)
colnames(submit)<-c("ID","TARGET")
write.csv(submit, file = "santander_xgboost_submit.csv", row.names = FALSE)

