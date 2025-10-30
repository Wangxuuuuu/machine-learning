
"""
模板（不依赖第三方ML库）：任务2 梯度下降（BGD/SGD）
- 目的：给学生手写实现线性回归的梯度下降训练（任选 BGD 或 SGD）
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
FIG_PATH = os.path.join(OUT_DIR, 'task2_mse_curve_template.png')

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

# ============================
# TODO: 算法实现区域（学生填写）
# 目标：实现 BGD 或 SGD 训练线性回归，返回 (w, b, hist)
# 建议：BGD 每轮使用全部样本；SGD 可每次采样一个或小批量样本
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

    print('[Template] Implement gd_train(X_train, y_train) returning w, b, history.')

    # 如已实现，可取消注释进行训练与作图
    w, b, hist = gd_train(X_train, y_train, lr=0.05, epochs=300)
    
    # 评估
    y_train_pred = X_train @ w + b
    y_test_pred = X_test @ w + b
    train_mse = float(np.mean((y_train - y_train_pred) ** 2))
    test_mse = float(np.mean((y_test - y_test_pred) ** 2))
    print(f'Train MSE: {train_mse:.4f}')
    print(f'Test MSE:  {test_mse:.4f}')
    
    # 收敛曲线
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(hist) + 1), hist, label='Train MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('GD convergence curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=150)
    plt.close()
    print('Saved figure:', FIG_PATH)


if __name__ == '__main__':
    main()