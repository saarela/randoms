## Non-essential cleaning:
rm(list=ls())   ## remove old variables from work space
graphics.off()  ## close existing figures
## Load libraries for the methods used:
library(MASS)   ## for LDA
library(e1071)  ## as the name suggests, this is needed for SVM (also naive Bayes)
library(class)  ## for k nearest neighbor
library(mnormt) ## for multivariate normal density and random numbers

##--------------------------------------------------
## Functions
meshgrid <- function(x1,x2) {
    n1 <- length(x1)
    n2 <- length(x2)
    x1 <- matrix(rep(x1,n2),nrow=n2,byrow=TRUE)
    x2 <- matrix(rep(x2,n1),nrow=n2,byrow=FALSE)
    list("x1"=x1,"x2"=x2)
}

plot_background <- function(d,npoints,title,xlab,ylab) {
    ncols <- length(unique(d))
    opacity <- .2
    cols <- c(rgb(0,0,1,opacity),
              rgb(1,0,0,opacity),
              rgb(0,1,0,opacity),
              rgb(1,0,1,opacity),
              rgb(0,1,1,opacity),
              rgb(1,1,0,opacity))  
    
    d <- matrix(as.numeric(d),ncol=npoints)
    image(x,x,t(d),
          col=cols[1:ncols],
          xaxs="r",yaxs="r",
          xlim=c(-3,3),ylim=c(-3,3),bty="n",axes=FALSE,
          main=title,col.main=gray(.2))

    axis(1,at=-3:3,labels=c("-3","","","0","","","3"),tck=-.02)
    axis(2,at=-3:3,labels=c("-3","","","0","","","3"),tck=-.02)

    mtext(xlab,side=1,line=2,at=0,cex=.8)
    mtext(ylab,side=2,line=2,at=0,cex=.8)

}

plot_contours <- function(mu,s) {

    opacity <- 1
    cols <- c(rgb(0,0,1,opacity),
              rgb(1,0,0,opacity),
              rgb(0,.8,0,opacity),
              rgb(.8,0,.8,opacity),
              rgb(0,.8,.8,opacity),
              rgb(.8,.8,0,opacity))

    ##levels <- dnorm(c(1),0,1)
    ##levels <- levels/max(levels)

    
    ## Distribution means
    for (ii in 1:length(mu)) {
        for (jj in -1:1) {
            d <- mu[ii]+jj*s[ii]
            x <- seq(-d,d,length=100)
            y <- sqrt(d^2-x^2)
            if (jj==0)
                linewidth <- 2
            else
                linewidth <- 1
            lines(x,y,col=cols[ii],lwd=linewidth)
            lines(x,-y,col=cols[ii],lwd=linewidth)
        }
        
        ##points(mu[ii,1],mu[ii,2],pch=20,col=cols[ii])

         ## Contours
         #dtmp <- matrix(d[,ii],ncol=npoints)
         #contour(x=x,y=x,t(dtmp),
         #        col=cols[ii],levels = levels*max(dtmp),
         #        drawlabels = FALSE,add=TRUE)
     }
}

plot_classification <- function(dat,clhat,title,xlab,ylab){

    opacity <- 1
    cols <- c(rgb(0,0,1,opacity),
              rgb(1,0,0,opacity),
              rgb(0,.8,0,opacity),
              rgb(.8,0,.8,opacity),
              rgb(0,.8,.8,opacity),
              rgb(.8,.8,0,opacity))

    g <- unique(dat$class)
    
    plot(NA,NA,xlim=c(-3,3),ylim=c(-3,3),
         main=title,,col.main=gray(.2),
         bty="n",type="n",tck=-.02,axes=FALSE)

    axis(1,at=-3:3,labels=c("-3","","","0","","","3"),tck=-.02)
    axis(2,at=-3:3,labels=c("-3","","","0","","","3"),tck=-.02)

    mtext(xlab,side=1,line=2,at=0,cex=.8)
    mtext(ylab,side=2,line=2,at=0,cex=.8)

    for (ii in 1:length(g)) {
        points(dat$x1[dat$class==(ii-1)],
               dat$x2[dat$class==(ii-1)],
               pch=21,col=cols[ii],cex=1.2,lwd=.5)
    }
    for (ii in 1:length(g)) {
        points(dat$x1[clhat==(ii-1)],
               dat$x2[clhat==(ii-1)],
               pch=4,col=cols[ii],lwd=.5)
    }
}

class_opt <- function() {

}

##--------------------------------------------------
## Total number of data points
n <- 2000

## Proportion of data used as training data
prop_train <- .8

## Randomly set class labels
nclass <- 2
cl <- sample(0:(nclass-1),n,replace=T)

## Number of points in the grid.  These values are used for
## illustrating the different decision regions in the two-dimensional
## space were using.
npoints <- 201

## max value of the axes
xmax <- 3

## Set class means.  The space is two-dimensional but in this example
## the distributions are radial, so the mean is a scalar (distance
## from origin).  One vector holds all the means.
mu <- c(.5,2)

## Standard deviations of the distributions, again in a vector
sigma <- c(.5,.5)

## Make random data vectors.  First initialize the matrix:
x <- matrix(NA,ncol=2,nrow=n)
r <- matrix(NA,ncol=1,nrow=n)
theta <- matrix(NA,ncol=1,nrow=n)
## Then loop through the classes and make random draws from the normal
## distribution.  The distributions are radial.
for (ii in 1:nclass) {
    theta[cl==ii-1] <- runif(sum(cl==ii-1),0,2*pi)
    r[cl==ii-1] <- rnorm(sum(cl==ii-1),mu[ii],sigma[ii])
}

x[,1] <- r*cos(theta)
x[,2] <- r*sin(theta)


## Make a data frame.  The columns hold the class labels and the
## simulated values on the two dimensions:
df <- data.frame(factor(cl),x)
colnames(df) <- c("class","x1","x2")

## Randomly select training and test sets.  We're only using those two
## sets here, not a validation set, although in practise that would be
## used for example to choose the K in K-nearest-neighbor, to choose
## kernel type for SVM etc.
idxtrain <- sample(1:n,round(prop_train*n),replace=FALSE)
dat.train <- df[idxtrain,]
dat.test <- df[-idxtrain,]  ## the minus says "all but these indices"

## Make another data frame with finely (and regularly) spaced values
## on a grid.  This is used for the background color in the plots to
## visualize the decision regions.
x <- seq(-xmax,xmax,length=npoints)
X <- meshgrid(x,x)
x1 <- matrix(X$x1,nrow=npoints^2)
x2 <- matrix(X$x2,nrow=npoints^2)
dat.bkgr <- data.frame("x1"=x1,"x2"=x2)

##--------------------------------------------------

## Initialize figure.  You can set the plot type to "svg" but that
## leads to a huge file size because of the background coloring (which
## is included as an image).  SVG is vector graphics so the quality is
## better of course.  Without the background color the file size is
## manageable.
plt.type <- "png" ## "svg" "png" or ""

if (plt.type=="svg") {
    svg(file="classification_nonlin.svg",
        width=4, height=12, pointsize=10, onefile=FALSE)
} else if (plt.type=="png") {
    png(filename = "classification_nonlin.png",
        width=4, height=12, units="in", pointsize=12,
        bg="white", res=300,
        type="cairo",antialias="gray")
}

par(mar=c(3,3,1,1))

l <- layout(matrix(1:12,ncol=2,byrow=TRUE))
layout.show(l)

## Each row of the figure will show results from one methods.  Each
## row has two panels: Left panel with the shaded background showing
## the decision regions and the contour plots of the real
## distributions.  Right panel with test data and the
## correctly/incorrectly classified points.

##--------------------------------------------------
## Optimal classification (parameters known).

## Bivariate normal densities, used as background colors to illustrate
## decision regions.  Think that we are computing the likelihoods for
## each point on a grid.
d <- matrix(NA,ncol=length(mu),nrow=dim(dat.bkgr)[1])
for (ii in 1:nclass) {
    r <- apply(dat.bkgr[,c("x1","x2")],1,function(x) sqrt(sum(x^2)))
    ## r <- sqrt(dat.bkgr$x1^2+dat.bkgr$x2^2) 
    d[,ii] <- dnorm(r,mu[ii],sigma[ii])
}

## First, get column number of highest likelihood for each row, then
## subtract one to get class number (this basically gives a map that
## shows the regions for the three decision that we can plot).
areas.opt <- apply(d,1,function(x) which.max(x)) - 1

## Now compute likelihoods for test data
l <- matrix(NA,ncol=length(mu),nrow=dim(dat.test)[1])
for (ii in 1:nclass) {
    r <- apply(dat.test[,c("x1","x2")],1,function(x) sqrt(sum(x^2)))
    ##theta <- atan(dat.test$x2,dat.test$x1)
    l[,ii] <- dnorm(r,mu[ii],sigma[ii])
}

## Assign inferred class labeled based on max likelihood (the three
## classes have equal prior probabilities, so likelihood is ok).
clhat.opt <- apply(l,1,function(x) which.max(x)) - 1

## Compute a confusion matrix; True versus predicted classes
conf.opt <- tbl <- table("Pred"=clhat.opt,"True"=dat.test$class)
## Proportion of correctly classified observations:
pc.opt <- sum(diag(conf.opt))/sum(conf.opt)
## OR: pc.opt <- mean(clhat.opt==dat.test$class)

print(conf.opt)
cat("OPT prop. correct: ",pc.opt,"\n",sep="")

## Background coloring as image, see the plotting functions defined
## earlier.
plot_background(areas.opt,npoints,"Optimal","",expression(x[2]))

## On top of the image, contour plots of the (real) distributions
plot_contours(mu,sigma)

## Plot the classification results.  This plots all the data first as
## circles, colored according to their true class.  On top of that,
## the same data are plotted as crosses with colors based on the
## classification we did above.
plot_classification(dat.test,clhat.opt,sprintf("Accuracy: %4.2f",pc.opt),"","")

##--------------------------------------------------
## Linear discriminant analysis

## Train
m.lda <- lda(class~x1+x2,data=dat.train)

## Get predicted classes for test data
clhat.lda <- predict(m.lda,newdata=dat.test,type="response")$class

## Compute the predicted classes for a grid of points; these are used
## for background coloring of the figure
areas.lda <- predict(m.lda,newdata=data.frame("x1"=x1,"x2"=x2),
                     type="response")$class

## Confusion matrix
conf.lda <- tbl <- table("Pred"=clhat.lda,"True"=dat.test$class)
## Proportion correct
pc.lda <- sum(diag(conf.lda))/sum(conf.lda)
## OR: pc.lda <- mean(clhat.lda==dat.test$class)

print(conf.lda)
cat("LDA prop. correct: ",pc.lda,"\n",sep="")

## Background coloring as image
plot_background(areas.lda,npoints,"LDA","",expression(x[2]))

## Contour plots of the (true) distributions
plot_contours(mu,sigma)

## Plot data
plot_classification(dat.test,clhat.lda,sprintf("Accuracy: %4.2f",pc.lda),"","")

#cf.lda <- coef(m.lda)
#lines(3*c(0,cf.lda[,"LD1"][1]),3*c(0,cf.lda[,"LD1"][2]),pch=20,col="black")
#lines(3*c(0,cf.lda[,"LD2"][1]),3*c(0,cf.lda[,"LD2"][2]),pch=20,col="black")

##--------------------------------------------------
## Support vector machine - linear kernel

## Train
m.svmlin <- svm(class~x1+x2,data=dat.train,kernel="linear")

## Get predicted classes for test data
clhat.svmlin <- predict(m.svmlin,newdata=dat.test)

## Compute the predicted classes for a grid of points; these are used
## for background coloring of the figure
areas.svmlin <- predict(m.svmlin,newdata=data.frame("x1"=x1,"x2"=x2),
                     type="response")

## Confusion matrix
conf.svmlin <- tbl <- table("Pred"=clhat.svmlin,"True"=dat.test$class)
## Proportion correct
pc.svmlin <- sum(diag(conf.svmlin))/sum(conf.svmlin)
## OR: pc.svm <- mean(clhat.svm==dat.test$class)

print(conf.svmlin)
cat("SVM, linear kernel prop. correct: ",pc.svmlin,"\n",sep="")

## Background coloring as image
plot_background(areas.svmlin,npoints,"SVM, linear kernel","",expression(x[2]))

## Contour plots of the (true) distributions
plot_contours(mu,sigma)

## Plot test data
plot_classification(dat.test,clhat.svmlin,sprintf("Accuracy: %4.2f",pc.svmlin),"","")

##--------------------------------------------------
## Support vector machine 

## Train
m.svm <- svm(class~x1+x2,data=dat.train)##,kernel="linear")

## Get predicted classes for test data
clhat.svm <- predict(m.svm,newdata=dat.test)

## Compute the predicted classes for a grid of points; these are used
## for background coloring of the figure
areas.svm <- predict(m.svm,newdata=data.frame("x1"=x1,"x2"=x2),
                     type="response")

## Confusion matrix
conf.svm <- tbl <- table("Pred"=clhat.svm,"True"=dat.test$class)
## Proportion correct
pc.svm <- sum(diag(conf.svm))/sum(conf.svm)
## OR: pc.svm <- mean(clhat.svm==dat.test$class)

print(conf.svm)
cat("SVM prop. correct: ",pc.svm,"\n",sep="")

## Background coloring as image
plot_background(areas.svm,npoints,"SVM","",expression(x[2]))

## Contour plots of the (true) distributions
plot_contours(mu,sigma)

## Plot test data
plot_classification(dat.test,clhat.svm,sprintf("Accuracy: %4.2f",pc.svm),"","")

##--------------------------------------------------
## K nearest neighbor

## Set K.  For each data point in the test set, the classification
## algorithm chooses the K closest points in the training set and
## checks the classes where they belong.  The predicted class for the
## test point is decided on a majority vote.  Very democratic.
K <- 50

clhat.knn <- knn(dat.train[,c("x1","x2")],
                dat.test[,c("x1","x2")],
                dat.train[,"class"],
                k=K)

## Compute the predicted classes for a grid of points; these are used
## for background coloring of the figure
areas.knn <- knn(dat.train[,c("x1","x2")],
                 data.frame("x1"=x1,"x2"=x2),
                 dat.train[,"class"],
                 k=K)

## Confusion matrix
conf.knn <- tbl <- table("Pred"=clhat.knn,"True"=dat.test$class)
## Proportion correct
pc.knn <- sum(diag(conf.knn))/sum(conf.knn)
## OR: pc.knn <- mean(clhat.knn==dat.test$class)

print(conf.knn)
cat("KNN prop. correct: ",pc.knn,"\n",sep="")

## Background coloring as image
plot_background(areas.knn,npoints,
                paste0("KNN (K=",K,")"),
                "",expression(x[2]))

## Contour plots of the (true) distributions
plot_contours(mu,sigma)

## Plot test data
plot_classification(dat.test,clhat.knn,
                    sprintf("Accuracy: %4.2f",pc.knn),
                    "","")

##--------------------------------------------------

## Naive Bayes

## Train
m.nvb <- naiveBayes(class~x1+x2,data=dat.train)

## Get predicted classes for test data
clhat.nvb <- predict(m.nvb,newdata=dat.test)

## Compute the predicted classes for a grid of points; these are used
## for background coloring of the figure
areas.nvb <- predict(m.nvb,newdata=data.frame("x1"=x1,"x2"=x2))

## Confusion matrix
conf.nvb <- tbl <- table("Pred"=clhat.nvb,"True"=dat.test$class)
## Proportion correct
pc.nvb <- sum(diag(conf.nvb))/sum(conf.nvb)
## OR: pc.nvb <- mean(clhat.nvb==dat.test$class)

print(conf.nvb)
cat("NVB prop. correct: ",pc.nvb,"\n",sep="")

## Background coloring as image
plot_background(areas.nvb,npoints,"Naive Bayes",expression(x[1]),expression(x[2]))

## Contour plots of the (true) distributions
plot_contours(mu,sigma)

## Plot test data
plot_classification(dat.test,clhat.nvb,sprintf("Accuracy: %4.2f",pc.nvb),expression(x[1]),"")

##--------------------------------------------------

if (plt.type!="")
    dev.off()
