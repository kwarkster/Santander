library(gamlr)
library(AUC)
library(xgboost)
library(caret)

train<-read.csv("train.csv")
test<-read.csv("test.csv")

dim(train)
summary(train)

############################
#
#  CLEANING
#
############################

x.all<-rbind(train[,-371], test)

## remove columns where all values are zero
dim(x.all)
x.all<-x.all[,sapply(x.all, function(v) var(v, na.rm=TRUE)!=0)]
dim(x.all)

## there aren't any columns with missing values
sapply(x.all, function(x) any(is.na(x)))


############################
#
#    FEATURE EXPLORATION
#
############################

## there aren't many folks with a TARGET of 1
hist(train$TARGET)

############################
#
#   MODELING W/ LASSO 
#   REGULARIZATION
#
############################


x.train<-x.all[1:nrow(train),]
x.train<-sparse.model.matrix(~., data=x.train)

x.test<-x.all[(nrow(train)+1):nrow(x.all),]
x.test<-sparse.model.matrix(~., data=x.test)

y<-train$TARGET

#BIC
reg<-gamlr(x.train, y, family="binomial")
summary(reg)
plot(reg)
Baicc<-coef(reg, select=which.min(AICc(reg)))[-1,]
log(reg$lambda[which.min(AICc(reg))]) #ideal log lambda value is -8.14, seg100
exp(-8.14) #0.00029 penalty

Baicc[order(-Baicc)[1:10]]
NonzeroB <-Baicc[Baicc!=0] #betas with nonzero coefficients, 
				   #added to model under ideal model 
				   #according to AICc criteria.
length(NonzeroB)
Baicc[which.min(Baicc)] #ind_var13
Baicc[which.max(Baicc)] #ind_var30_0

#CV Min
cv_reg<-cv.gamlr(x.train, y, family="binomial", nfold=15)

cv_Beta<-coef(cv_reg, select="min")[-1,]
cv_NonzeroB<-cv_Beta[cv_Beta!=0] 
length(cv_NonzeroB)
cv_NonzeroB[which.min(cv_NonzeroB)] 
cv_NonzeroB[which.max(cv_NonzeroB)] 
cv_NonzeroB[order(-cv_NonzeroB)[1:10]] 
cv_NonzeroB[order(cv_NonzeroB)[1:10]] 

exp(log(cv_reg$lambda.min))


cv_reg$seg.min #seg100 - OOS R^2 =.12

#CV 1se#
cv1se_Beta<-coef(cv_reg, select="1se")[-1,]
cv1se_NonzeroB<-cv_Beta[cv_Beta!=0] 
length(cv1se_NonzeroB)
cv1se_NonzeroB[which.min(cv_NonzeroB)] 
cv1se_NonzeroB[order(-cv_NonzeroB)[1:10]]
log(cv_reg$lambda.1se)
exp(-8.010307)

cv_reg$seg.1se #seg57 OOS R^2 =.11


############################
#
#   PREDICTION
#
############################

preds<-predict(cv_reg, x.test, select="min", type="response")

submit <- data.frame(ID = test$ID, TARGET = preds)
colnames(submit)<-c("ID","TARGET")
write.csv(submit, file = "santander_submit.csv", row.names = FALSE)

