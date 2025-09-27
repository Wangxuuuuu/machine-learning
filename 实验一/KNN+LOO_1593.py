import numpy as np
import time
## import... 安装所需要的依赖库

# def Oudistance(x1,x2):
#     return np.sqrt(np.sum((x1 - x2) ** 2))

def KNN(X_train,Y_train,x_test,k):
    # distances = [Oudistance(x_train,x_test) for x_train in X_train]
    distances=np.sqrt(np.sum((X_train-x_test)**2,axis=1))
    nearest_k_indexs=np.argsort(distances)[:k]
    nearest_k_labels = Y_train[nearest_k_indexs]
    counts = np.bincount(nearest_k_labels)
    maxnum = np.max(counts)
    if( np.sum(counts==maxnum) >1 ):
        maxindexs = np.where(counts==maxnum)[0]
        vote = np.random.choice(maxindexs)
    else:
        vote = np.argmax(counts)
    return vote

def loo_eval(X, y, k):
    print(f"开始在整个数据集semeion.data上进行LOO评估,k={k}")
    #计算时间
    time_start = time.time()
    N=len(X)
    correct=0
    for i in range(N):
        X_train = np.delete(X,i,0)
        Y_train = np.delete(y,i)
        x_test = X[i]
        y_true = y[i]
        y_pred = KNN(X_train,Y_train,x_test,k)
        if y_pred == y_true:
            correct += 1
    print("LOO评估完成,耗时: %.2f 秒" % (time.time() - time_start))
    acc = correct / N
    return acc


# 主流程
raw = np.loadtxt('semeion+handwritten+digital/semeion.data.txt')
X, y = raw[:, :256], np.argmax(raw[:, 256:], 1)

for k in [1, 3, 5]:
    acc = loo_eval(X, y, k)
    print(f'k={k}  LOO 准确率 = {acc:.4f}')

