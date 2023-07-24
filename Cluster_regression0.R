
data=read.csv("C:/Users/ryang/PycharmProjects/Scan/data.csv")
data$y=data$umean
data=data[data$col<350&data$col>200,]
names(data)=c("x", "y")
data$y=5000-data$y
data$x=data$x-median(data$x)
data$x2=data$x^2

nrow(data[data$x==0,])
 plot(data$x, data$y)

i=1; kluster=11; sdd=rep(1000, kluster); pii=rep(1, kluster); pii0=pii*9
n.n=nrow(data); inc=rep(NA,n.n);
pvar=matrix(rep(0, 3^2), 3); diag(pvar)=100
y=data$y
x=cbind(int=1,data[,which(names(data)%in%c("x", "x2"))])

cbeta=.5*(1:kluster)+mean(y)
cbeta=seq(min(y), max(y), length.out=kluster)
cbeta=quantile(data$y, seq(0,1,length.out=kluster))
pBeta=Beta=t(cbind(cbeta, seq(0, -.0001, length.out=kluster), seq(0, -.00000001, length.out=kluster)))

T=matrix(rep(NA,n.n*kluster),ncol=kluster)

for (i in 1:100)
{
  for (j in 1:kluster)
  {if (i!=1&j==2) {sdd[j]=sdd[j]/sqrt(2)}
    T[,j]=pii[j]*(dnorm((y-as.matrix(x)%*%as.matrix(Beta[,j])),0,sdd[j]))
  }
  
  T[rowSums(T)!=0,]=T[rowSums(T)!=0,]/rowSums(T[rowSums(T)!=0,])
  pii=colSums(T)/n.n
  #print(sum(pii-pii0))
  if (length(pii)==length(pii0))
  {if (sum(pii-pii0) ==0 &i>3) {break}}
  
  pii0=pii
  kluster0=kluster
    ww=rep(NA, n.n)
    for (h in 1:n.n) {ww[h]=which(T[h,]==max(T[h,]))[1]}
      tbww=table(ww)
      grps=as.numeric(names(tbww))
      kluster=length(grps)
      pii=pii[grps]#sm[,1][-c(1:3)]+sm[1,1]
      Beta=Beta[,grps]
      T=T[,grps]  
      print(i)
      print(tbww)
  for (j in kluster:1)
  {
    for (w in 1:n.n) {inc[w]=T[w,j]==max(T[w,])}
    {Beta[,j]= solve(pvar+as.matrix(t(x[inc,])) %*% as.matrix(x[inc,])) %*%((pvar)%*%pBeta[,j]   + as.matrix(t(x[inc,])) %*% as.matrix(y[inc]))
    

    if (i>5)
    {
      xx=x
      xx$ww=ww; xx$y=y
      lm=lm(y~x+x2+as.factor(ww), data=xx)
      sm=summary(lm)[[4]]
      Beta[2,]=sm[2,1]; Beta[3,]=sm[3,1]
      Beta[1,1]=sm[1,1]

      if (nrow(sm)==(kluster+2))
      {Beta[1,2:kluster]=sm[,1][-c(1:3)]+sm[1,1]} else
      {if (j==kluster0)
      {grps=substr(row.names(sm),14,15)
      grps=grps[grps!=""]; grps=c(1,as.numeric(grps))
      kluster=length(grps)
      pii=pii[grps]#sm[,1][-c(1:3)]+sm[1,1]
      Beta=Beta[,grps]
      T=T[,grps]}
        print(i)
        Beta[1,2:kluster]=sm[,1][-c(1:3)]+sm[1,1]
      }
      
    }
    }
  } 
  inc=rep(NA, n.n)
  for (j in kluster:1)
  { 
    for (w in 1:n.n) {inc[w]=T[w,j]==max(T[w,])}
 
    if (sum(inc)==1)
    {sdd[j]=100} else
    {sdd[j]= apply((y[inc]-as.matrix((x[inc,]))%*%Beta[,j]), 2, sd)
    }
  }
  sdd=rep(mean(sdd[!is.na(sdd)&sdd!=0]), kluster)
}
   aic <- AIC(lm)
   print(aic)

par(mfrow=c(1,2))
plot(data$x, data$y)
for (j in 1:kluster) {
  points(data$x, Beta[1, j] + Beta[2, j] * data$x + Beta[3, j] * data$x2, cex = 0.6, col = j + 1)
}

T.max=ccol=rep(NA,n.n)

for (w in 1:n.n)
{
  ccol[w]=(which(T[w,]==max(T[w,])))+1
  T.max[w]=max(T[w,])
}


plot(data$x, data$y, col=ccol)


