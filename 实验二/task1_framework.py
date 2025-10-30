
"""
模板（不依赖第三方ML库）：任务1 正规方程 Normal Equation
- 目的：给学生手写实现 θ = (X^T X)^{-1} X^T y 的位置
- 本模板仅保留数据读取、划分与可视化骨架；算法实现留白
- 允许使用 numpy / pandas / matplotlib，禁止使用 sklearn 等ML库
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
REG_CSV = os.path.join(DATA_DIR, 'dataset_regression.csv')
FIG_PATH = os.path.join(OUT_DIR, 'task1_fit_template.png')

np.random.seed(42)


def train_test_split_xy(x: np.ndarray, y: np.ndarray, test_ratio: float = 0.2):
    n = x.shape[0]
    idx = np.random.permutation(n)
    test_size = int(n * test_ratio)
    return x[idx[test_size:]], y[idx[test_size:]], x[idx[:test_size]], y[idx[:test_size]]


# ============================
# TODO: 算法实现区域（学生填写）
# 目标：实现正规方程，返回 w, b 使 y ≈ w x + b（单特征）
# 提示：构造设计矩阵 Xb = [x, 1]，令 theta = (Xb^T Xb)^{-1} Xb^T y
# def normal_equation_fit(x_train: np.ndarray, y_train: np.ndarray):
#     # YOUR CODE HERE
#     # return w, b
#     raise NotImplementedError

def normal_equation_fit(x_train: np.ndarray, y_train: np.ndarray):
    """
    使用正规方程求解线性回归参数
    参数:
        x_train: 训练特征 (m,)
        y_train: 训练目标 (m,)
    返回:
        w: 斜率
        b: 截距
    """
    # 1. 构造设计矩阵 Xb = [x, 1]
    # 添加偏置项（全1列）
    Xb = np.column_stack((x_train, np.ones_like(x_train)))
    
    # 2. 计算正规方程: θ = (Xb^T Xb)^{-1} Xb^T y
    # 使用伪逆提高数值稳定性
    theta = np.linalg.pinv(Xb.T @ Xb) @ Xb.T @ y_train
    
    # 3. 提取参数
    w = theta[0]  # 斜率
    b = theta[1]  # 截距
    
    return w, b

# ============================


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(REG_CSV):
        print('Missing data file at:', REG_CSV)
        print('Please place dataset_regression.csv under data/.')
        return

    # 读取两列数值：第一列为x，第二列为y
    df = pd.read_csv(REG_CSV)
    num_df = df.select_dtypes(include=[np.number])
    x = num_df.iloc[:, 0].to_numpy().astype(float)
    y = num_df.iloc[:, 1].to_numpy().astype(float)

    x_train, y_train, x_test, y_test = train_test_split_xy(x, y, test_ratio=0.2)

    print('[Template] This is a student skeleton.')
    print('Please implement normal_equation_fit(x_train, y_train) that returns w, b.')

    # 如已实现，可取消下方注释运行可视化与评估
    w, b = normal_equation_fit(x_train, y_train)
    y_train_pred = w * x_train + b
    y_test_pred = w * x_test + b
    train_mse = float(np.mean((y_train - y_train_pred) ** 2))
    test_mse = float(np.mean((y_test - y_test_pred) ** 2))
    print(f'Train MSE: {train_mse:.4f}\nTest MSE: {test_mse:.4f}')
    
    # 预测5个新样本
    x_new = np.array([-12, -3, 0, 4.5, 11])
    y_new_pred = w * x_new + b
    print('5 new predictions:')
    for xi, yi in zip(x_new, y_new_pred):
        print(f'  x={xi:.2f} -> y_pred={yi:.3f}')
    
    # 作图（英文标签）
    plt.figure(figsize=(6, 4))
    plt.scatter(x_train, y_train, s=12, color='#1f77b4', label='Train data')
    xs = np.linspace(np.min(x_train), np.max(x_train), 200)
    ys = w * xs + b
    plt.plot(xs, ys, color='#d62728', label=f'Fit: y={w:.3f}x+{b:.3f}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=150)
    plt.close()
    print('Saved figure:', FIG_PATH)


if __name__ == '__main__':
    main()
