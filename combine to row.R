

data=read.csv("C:/Users/shpry/PycharmProjects/cal/quadratic_regression_predictions.csv")
names(data)=c("x", "y", "z")
data$x2=data$x^2
data$diff=(c(1,diff(data[,1]))-1)!=0
data$col=1+cumsum(data$diff)
plot(data$diff)
plot(data[,1], 5000-data[,2], cex=.2, col=data$col)

table(diff(data[,1])>2)

data=data[data$col>330&data$col<400,]
plot(data[,1], 5000-data[,3], pch=16, col=data$col)

lm=lm(data[,2]~data$x+data$x2+as.factor(data$col))
sm=summary(lm)

plot(sm[[4]][-c(1:3),1])

data$pred=predict(lm, data$x)