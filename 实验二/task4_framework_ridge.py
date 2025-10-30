
"""
模板（不依赖第三方ML库）：任务4 岭回归（解析法）
- 目的：让学生实现岭回归的闭式解 (X^T X + λI)^{-1} X^T y
- 本模板仅保留数据读取、划分与评估骨架；算法实现留白
- 允许使用 numpy / pandas，禁止使用 sklearn
"""
import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
CSV = os.path.join(DATA_DIR, 'winequality-white.csv')

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
# 目标：实现岭回归闭式解，返回 (w, b)
# 提示：在特征后追加一列常数 1 以表示偏置 b
# def ridge_fit_closed_form(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
#     # YOUR CODE HERE: 返回包含 w 和 b 的向量 theta
#     # 形如: theta = (X'X + lam * I)^(-1) X'y
#     raise NotImplementedError

def ridge_fit_closed_form(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """
    使用岭回归的闭式解求解参数
    参数:
        X: 设计矩阵 (已包含偏置项)
        y: 目标向量
        lam: 正则化参数λ
    返回:
        theta: 参数向量 [w, b]
    """
    m, n = X.shape  # m: 样本数, n: 特征数(包括偏置项)
    
    # 构造正则化矩阵: λI
    # 注意: 通常不对偏置项进行正则化，所以最后一个元素设为0
    I = np.eye(n)
    I[-1, -1] = 0  # 不对偏置项正则化
    
    # 计算岭回归闭式解: θ = (X^T X + λI)^(-1) X^T y
    # 使用伪逆提高数值稳定性
    theta = np.linalg.pinv(X.T @ X + lam * I) @ X.T @ y
    
    return theta

# ============================


def main():
    df = pd.read_csv(CSV, sep=';')
    X = df.iloc[:, :-1].to_numpy().astype(float)
    y = df.iloc[:, -1].to_numpy().astype(float)

    X_train, y_train, X_test, y_test = train_test_split(X, y, test_ratio=0.2)
    X_train, X_test = normalize(X_train, X_test)

    print('[Template] Implement ridge_fit_closed_form(X_train, y_train, lam).')

    # 测试不同的正则化参数
    lambda_values = [0, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    
    # 追加常数列用于偏置
    X_train_ext = np.column_stack([X_train, np.ones(X_train.shape[0])])
    X_test_ext = np.column_stack([X_test, np.ones(X_test.shape[0])])
    
    print("λ值\t训练MSE\t测试MSE\t参数范数\t泛化差距")
    print("-" * 50)
    
    train_errors = []
    test_errors = []
    param_norms = []
    
    for lam in lambda_values:
        theta = ridge_fit_closed_form(X_train_ext, y_train, lam)
        w, b = theta[:-1], float(theta[-1])
        
        # 计算预测和误差
        y_train_pred = X_train @ w + b
        y_test_pred = X_test @ w + b
        train_mse = mse(y_train, y_train_pred)
        test_mse = mse(y_test, y_test_pred)
        
        # 计算参数范数（不包括偏置项）
        param_norm = np.linalg.norm(w)
        generalization_gap = test_mse - train_mse
        
        train_errors.append(train_mse)
        test_errors.append(test_mse)
        param_norms.append(param_norm)
        
        print(f"{lam}\t{train_mse:.4f}\t{test_mse:.4f}\t{param_norm:.4f}\t{generalization_gap:.4f}")

    # # 如已实现，可取消注释进行评估
    # lam = 1.0
    # # 追加常数列用于偏置
    # X_train_ext = np.column_stack([X_train, np.ones(X_train.shape[0])])
    # X_test_ext = np.column_stack([X_test, np.ones(X_test.shape[0])])
    # theta = ridge_fit_closed_form(X_train_ext, y_train, lam)
    # w, b = theta[:-1], float(theta[-1])
    # train_mse = mse(y_train, X_train @ w + b)
    # test_mse = mse(y_test, X_test @ w + b)
    # print(f'lambda={lam} -> Train MSE: {train_mse:.4f}')
    # print(f'lambda={lam} -> Test MSE:  {test_mse:.4f}')


if __name__ == '__main__':
    main()