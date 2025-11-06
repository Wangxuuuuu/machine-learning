"""
实验四：贝叶斯分类器

实验目标：
在数据集上应用贝叶斯规则进行分类，计算分类错误率，分析实验结果

数据已生成，请自行实现贝叶斯分类算法
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出目录
os.makedirs('out', exist_ok=True)

# ============================================================
# 数据生成
# ============================================================

print('='*60)
print('实验四：贝叶斯分类器')
print('='*60)

# 生成数据集1：高分离度
print('\n生成数据集1 (高分离度)...')
X1, y1 = make_classification(
    n_samples=500,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_classes=2,
    n_clusters_per_class=1,
    class_sep=2.0,
    random_state=42
)

# 生成数据集2：低分离度  
print('生成数据集2 (低分离度)...')
X2, y2 = make_classification(
    n_samples=500,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_classes=2,
    n_clusters_per_class=1,
    class_sep=0.5,
    random_state=42
)

# 划分训练集和测试集
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3, random_state=42)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3, random_state=42)

print(f'\n数据集1: 训练集{X1_train.shape[0]}样本, 测试集{X1_test.shape[0]}样本')
print(f'数据集2: 训练集{X2_train.shape[0]}样本, 测试集{X2_test.shape[0]}样本')

# ============================================================
# 贝叶斯分类器实现
# ============================================================

class BayesianClassifier:
    """
    贝叶斯分类器
    基于高斯假设实现贝叶斯规则分类
    """
    
    def __init__(self):
        self.priors_ = None  # 先验概率 P(C_i)
        self.means_ = None   # 各类别特征均值
        self.vars_ = None    # 各类别特征方差
        self.classes_ = None # 类别标签
        self._epsilon = 1e-9  # 防止除零和log(0)
    
    def fit(self, X, y):
        """
        训练贝叶斯分类器
        计算每个类别的均值、方差和先验概率
        
        输入:
            X: 训练数据特征, shape=(n_samples, n_features)
            y: 训练数据标签, shape=(n_samples,)
        """
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        
        # 初始化参数
        self.means_ = np.zeros((n_classes, n_features))
        self.vars_ = np.zeros((n_classes, n_features))
        self.priors_ = np.zeros(n_classes)
        
        # 参数估计
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]  # 属于类别c的所有样本
            # 计算均值
            self.means_[idx, :] = np.mean(X_c, axis=0)
            # 计算方差
            self.vars_[idx, :] = np.var(X_c, axis=0)
            # 计算先验概率
            self.priors_[idx] = X_c.shape[0] / float(n_samples)
        
        print("训练完成！参数信息:")
        for i, c in enumerate(self.classes_):
            print(f"  类别{c}: 先验概率={self.priors_[i]:.3f}, "
                  f"均值={self.means_[i]}, 方差={self.vars_[i]}")
        
        return self
    
    def _gaussian_pdf(self, X, mean, var):
        """
        计算高斯概率密度函数 (PDF)
        公式: f(x) = (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))
        """
        numerator = np.exp(-((X - mean) ** 2) / (2 * (var + self._epsilon)))
        denominator = np.sqrt(2 * np.pi * (var + self._epsilon))
        return numerator / denominator
    
    def _calculate_log_likelihood(self, X):
        """
        计算对数似然 log P(X|C_i)
        朴素贝叶斯假设: 特征条件独立
        log P(X|C_i) = Σ log P(x_j|C_i)
        """
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        log_likelihoods = np.zeros((n_samples, n_classes))
        
        for i in range(n_classes):
            mean = self.means_[i, :]
            var = self.vars_[i, :]
            # 计算每个特征的高斯PDF
            pdfs = self._gaussian_pdf(X, mean, var)
            # 防止除零错误
            pdfs[pdfs == 0] = self._epsilon
            # 计算对数似然（特征独立，所以是乘积的对数=对数的和）
            log_pdfs = np.log(pdfs)
            log_likelihoods[:, i] = np.sum(log_pdfs, axis=1)
        
        return log_likelihoods
    
    def predict(self, X):
        """
        贝叶斯分类预测
        使用最大后验概率规则 (MAP)
        
        决策规则: C* = argmax P(C_i|X) = argmax [P(X|C_i) × P(C_i)]
        对数形式: C* = argmax [log P(X|C_i) + log P(C_i)]
        """
        # 计算对数似然
        log_likelihoods = self._calculate_log_likelihood(X)
        
        # 计算对数先验
        log_priors = np.log(self.priors_)
        
        # 计算对数后验 = 对数似然 + 对数先验
        log_posteriors = log_likelihoods + log_priors
        
        # 选择后验概率最大的类别
        predictions = np.argmax(log_posteriors, axis=1)
        
        return predictions
    
    def score(self, X, y):
        """计算分类准确率"""
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

# ============================================================
# 决策边界可视化函数
# ============================================================

def plot_decision_boundary(model, X, y, title, filename):
    """
    绘制决策边界和数据分布
    
    输入:
        model: 训练好的模型
        X: 数据特征
        y: 真实标签
        title: 图表标题
        filename: 保存文件名
    """
    # 创建网格点
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))
    
    # 预测网格点类别
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid_points)
    Z = Z.reshape(xx.shape)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制决策边界
    from matplotlib.colors import ListedColormap
    ax.contourf(xx, yy, Z, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']), alpha=0.6)
    
    # 绘制数据点
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(['#FF0000', '#0000FF']),
               edgecolor='k', s=20, label='数据点')
    
    # 标记错误分类的点
    predictions = model.predict(X)
    errors = (y != predictions)
    if np.any(errors):
        ax.scatter(X[errors, 0], X[errors, 1], c='yellow', marker='x', 
                   s=100, linewidths=3, label='错误分类')
    
    # 计算错误率
    error_rate = 1 - accuracy_score(y, predictions)
    
    # 设置图形属性
    ax.set_title(f'{title}\n错误率: {error_rate:.4f}')
    ax.set_xlabel('特征1')
    ax.set_ylabel('特征2')
    ax.legend()
    
    # 保存图形
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'  图片已保存: {filename}')
    
    return error_rate

# ============================================================
# 实验执行部分
# ============================================================

def run_experiment():
    """运行贝叶斯分类器实验"""
    
    # 创建贝叶斯分类器实例
    clf = BayesianClassifier()
    
    # 数据集1：高分离度
    print('\n' + '='*60)
    print('数据集1 (高分离度) - 贝叶斯分类')
    print('='*60)
    
    # 训练模型
    print('\n[步骤1] 训练贝叶斯分类器...')
    clf.fit(X1_train, y1_train)
    
    # 测试模型
    print('\n[步骤2] 测试分类性能...')
    accuracy1 = clf.score(X1_test, y1_test)
    error_rate1 = 1 - accuracy1
    print(f'  测试集准确率: {accuracy1:.4f}')
    print(f'  测试集错误率: {error_rate1:.4f}')
    
    # 可视化决策边界
    print('\n[步骤3] 生成决策边界图...')
    error1 = plot_decision_boundary(clf, X1, y1, 
                                   '数据集1 - 贝叶斯分类器决策边界 (高分离度)',
                                   'out/实验四_数据集1_决策边界.png')
    
    # 数据集2：低分离度
    print('\n' + '='*60)
    print('数据集2 (低分离度) - 贝叶斯分类')
    print('='*60)
    
    # 训练模型
    print('\n[步骤1] 训练贝叶斯分类器...')
    clf.fit(X2_train, y2_train)
    
    # 测试模型
    print('\n[步骤2] 测试分类性能...')
    accuracy2 = clf.score(X2_test, y2_test)
    error_rate2 = 1 - accuracy2
    print(f'  测试集准确率: {accuracy2:.4f}')
    print(f'  测试集错误率: {error_rate2:.4f}')
    
    # 可视化决策边界
    print('\n[步骤3] 生成决策边界图...')
    error2 = plot_decision_boundary(clf, X2, y2, 
                                   '数据集2 - 贝叶斯分类器决策边界 (低分离度)',
                                   'out/实验四_数据集2_决策边界.png')
    
    return accuracy1, error_rate1, accuracy2, error_rate2

# ============================================================
# 示例：可视化数据分布
# ============================================================

print('\n生成数据分布图...')
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 数据集1
axes[0].scatter(X1[y1==0, 0], X1[y1==0, 1], c='red', label='类别0', alpha=0.6, edgecolors='k')
axes[0].scatter(X1[y1==1, 0], X1[y1==1, 1], c='blue', label='类别1', alpha=0.6, edgecolors='k')
axes[0].set_title('数据集1 (高分离度)')
axes[0].set_xlabel('特征1')
axes[0].set_ylabel('特征2')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 数据集2
axes[1].scatter(X2[y2==0, 0], X2[y2==0, 1], c='red', label='类别0', alpha=0.6, edgecolors='k')
axes[1].scatter(X2[y2==1, 0], X2[y2==1, 1], c='blue', label='类别1', alpha=0.6, edgecolors='k')
axes[1].set_title('数据集2 (低分离度)')
axes[1].set_xlabel('特征1')
axes[1].set_ylabel('特征2')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('out/实验四_数据分布.png', dpi=100, bbox_inches='tight')
plt.close()
print('数据分布图已保存: out/实验四_数据分布.png')

# ============================================================
# 运行实验
# ============================================================

print('\n' + '='*60)
print('开始运行贝叶斯分类器实验')
print('='*60)

try:
    # 运行实验
    acc1, err1, acc2, err2 = run_experiment()
    
    # 实验总结
    print('\n' + '='*60)
    print('实验总结')
    print('='*60)
    print(f'数据集1 (高分离度): 准确率 = {acc1:.4f}, 错误率 = {err1:.4f}')
    print(f'数据集2 (低分离度): 准确率 = {acc2:.4f}, 错误率 = {err2:.4f}')
    
    # 性能对比分析
    accuracy_diff = abs(acc1 - acc2)
    print(f'\n性能分析:')
    print(f'  准确率差异: {accuracy_diff:.4f}')
    
    if accuracy_diff > 0.1:
        print('  → 数据分离度对分类性能有显著影响')
        print('  → 高分离度数据更容易被贝叶斯分类器正确分类')
    else:
        print('  → 数据分离度对分类性能影响较小')
    
    print('\n' + '='*60)
    print('实验四完成！所有结果已保存至out目录')
    print('='*60)
    
except Exception as e:
    print(f'\n⚠️ 实验执行过程中出现错误: {e}')
    print('请检查代码实现是否正确')