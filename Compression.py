def Scal(a,b):
    scal=0
    for i in range(len(a)):
        scal+=a[i]*b[i]
    return scal

def sub(a,b):
    x=[0. for i in range(len(a))]
    for i in range(len(x)):
        x[i]=a[i]-b[i]
    return x

def mulNum(a,b):
    x=[0. for i in range(len(b))]
    for i in range(len(x)):
        x[i]= a*b[i]
    return x
def dev(a,b):
    x=[0. for i in range(len(a))]
    for i in range(len(x)):
        x[i]=a[i]/b
    return x
def S(a,b):
    x=[0. for i in range(len(a))]
    for i in range(len(x)):
        x[i]=a[i]+b[i]
    return x
def rec(a,b):
    x=[[0. for i in range(len(a[0]))]for j in range(len(a))]
    x=np.array(x)
    for k in range(len(a[0])):
        z=[[0. for i in range(len(a[0]))]for j in range(len(a))]
        z=np.array(z)
        for i in range(len(a)):
            z[i]=mulNum(a[i,k],b[k])
        for i in range(len(a)):
            for j in range(len(a[0])):
                x[i,j]+=z[i,j]
    return x
def Oj(x):
    Y=[[0.for i in range(len(x[0]))]for j in range(len(x))]
    w=[[0.for i in range(len(x[0]))]for j in range(len(x[0]))]
    Y=np.array(Y)
    w=np.array(w)
    for i in range(len(w)):
        w[0,i]=random.uniform(-1,1)
    for i in range(len(Y)):
        Y[i,0]=Scal(w[0],x[i])
    for k in range(len(x[0])-1):
        W=[[0. for i in range(len(x[0]))]for j in range(len(x))]
        for z in range(10000):
            W[0]=Norm(w[k])
            for i in range(1,len(W)):
                W[i]= S(W[i-1], mulNum(Y[i-1,k], dev(sub(x[i-1], mulNum(Y[i-1,k], W[i-1])), i)))
                Norm(W[i])
            w[k+1]=W[len(W)-1]
        for i in range(len(Y)):
            Y[i,k+1]=Scal(w[k+1],x[i])
    return Y
Реализация методов для выборки iris:
data=datasets.load_iris()
x=data.data
HyperCube(x)
averData=Aver(x)
Center(x,averData)
col=data.target
Y=Oj(x)

plt.scatter(Y[:,0],Y[:,1],c=col,s=20,cmap='Set1')
plt.title('Ирисы',fontsize=20)
plt.xlabel('1 главная компонента',fontsize=15)
plt.ylabel('2 главная компонента',fontsize=15)
plt.show()
plt.scatter(Y[:,2],Y[:,3],c=col,s=20,cmap='Set1')
plt.title('Ирисы',fontsize=20)
plt.xlabel('3 главная компонента',fontsize=15)
plt.ylabel('4 главная компонента',fontsize=15)
plt.show()
