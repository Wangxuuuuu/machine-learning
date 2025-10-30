
"""
模板（不依赖第三方ML库）：任务3 学习率曲线（基于手写GD）
- 目的：让学生实现线性回归的梯度下降训练函数，并比较不同学习率下的收敛曲线
- 本模板仅保留数据读取、划分与可视化骨架；算法实现留白
- 允许使用 numpy / pandas / matplotlib，禁止使用 sklearn
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
CSV = os.path.join(DATA_DIR, 'winequality-white.csv')
FIG_PATH = os.path.join(OUT_DIR, 'task3_lr_curves_template.png')

np.random.seed(42)


def train_test_split(X: np.ndarray, y: np.ndarray, test_ratio=0.2):
    n = X.shape[0]
    idx = np.random.permutation(n)
    test_size = int(n * test_ratio)
    return X[idx[test_size:]], y[idx[test_size:]], X[idx[:test_size]], y[idx[:test_size]]


def normalize(X_train: np.ndarray, X_test: np.ndarray):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))

# ============================
# TODO: 算法实现区域（学生填写）
# 目标：实现梯度下降训练线性回归，返回 (w, b, hist)
# 建议：BGD 每轮使用全部样本；可扩展为小批量 SGD
# def gd_train(X: np.ndarray, y: np.ndarray, lr=0.01, epochs=200):
#     # YOUR CODE HERE
#     # return w, b, history_mse_list
#     raise NotImplementedError
def gd_train(X: np.ndarray, y: np.ndarray, lr=0.01, epochs=200):
    """
    使用批量梯度下降训练线性回归模型
    参数:
        X: 特征矩阵 (m, n)
        y: 目标向量 (m,)
        lr: 学习率
        epochs: 迭代次数
    返回:
        w: 权重向量 (n,)
        b: 偏置项
        history_mse_list: 每次迭代的MSE历史记录
    """
    m, n = X.shape  # m: 样本数, n: 特征数
    # 初始化参数
    w = np.zeros(n)  # 权重初始化为0
    b = 0.0          # 偏置初始化为0
    # 记录每次迭代的MSE
    history_mse = []

    # 添加偏置项到特征矩阵（可选方案，这里我们显式处理b）
    # 另一种方案是将b作为w的一部分，这里选择分开处理更清晰
    
    # 批量梯度下降迭代
    for epoch in range(epochs):
        # 计算预测值
        y_pred = X @ w + b
        # 计算误差
        error = y_pred - y
        # 计算当前MSE并记录
        mse = np.mean(error ** 2)
        history_mse.append(mse)
        # 计算梯度
        # 对w的梯度: dJ/dw = (1/m) * X^T @ (y_pred - y)
        dw = (1/m) * (X.T @ error)
        # 对b的梯度: dJ/db = (1/m) * sum(y_pred - y)
        db = (1/m) * np.sum(error)
        # 更新参数
        w = w - lr * dw
        b = b - lr * db
        # 每50轮打印一次进度
        if epoch % 50 == 0:
            print(f'Epoch {epoch}, MSE: {mse:.4f}')
    
    return w, b, history_mse
# ============================


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(CSV, sep=';')
    X = df.iloc[:, :-1].to_numpy().astype(float)
    y = df.iloc[:, -1].to_numpy().astype(float)

    X_train, y_train, X_test, y_test = train_test_split(X, y, test_ratio=0.2)
    X_train, X_test = normalize(X_train, X_test)

    print('[Template] Implement gd_train(X_train, y_train) to compare learning rates.')

    # 如已实现，可取消注释进行多学习率对比与作图
    lrs = [0.001, 0.01, 0.05, 0.1]
    curves = {}
    epochs = 300
    for lr in lrs:
        w, b, hist = gd_train(X_train, y_train, lr=lr, epochs=epochs)
        train_m = mse(y_train, X_train @ w + b)
        test_m = mse(y_test, X_test @ w + b)
        curves[lr] = {'hist': hist, 'train_mse': train_m, 'test_mse': test_m}
        print(f'lr={lr:.3f} -> Train MSE={train_m:.4f}, Test MSE={test_m:.4f}')
    
    plt.figure(figsize=(7, 5))
    for lr, info in curves.items():
        plt.plot(range(1, len(info['hist']) + 1), info['hist'], label=f'lr={lr}')
    plt.xlabel('Epoch')
    plt.ylabel('Train MSE')
    plt.title('MSE convergence under different learning rates')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=150)
    plt.close()
    print('Saved figure:', FIG_PATH)


if __name__ == '__main__':
    main()