import numpy as np
import matplotlib.pyplot as plt

class StratifiedSampler:
    """分层采样器"""
    
    def __init__(self, test_size=0.3, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
    
    def split(self, X, y):
        """执行分层采样"""
        np.random.seed(self.random_state)
        X_train_list, X_test_list = [], []
        y_train_list, y_test_list = [], []
        
        # 对每个类别分别采样
        for class_label in range(10):
            class_indices = np.where(y == class_label)[0]
            np.random.shuffle(class_indices)
            n_test = int(len(class_indices) * self.test_size)
            
            # 分割训练集和测试集
            test_indices = class_indices[:n_test]
            train_indices = class_indices[n_test:]
            
            X_train_list.append(X[train_indices])
            X_test_list.append(X[test_indices])
            y_train_list.append(y[train_indices])
            y_test_list.append(y[test_indices])
        
        # 合并所有类别的数据
        X_train = np.vstack(X_train_list)
        X_test = np.vstack(X_test_list)
        y_train = np.hstack(y_train_list)
        y_test = np.hstack(y_test_list)
        
        # 打乱顺序
        train_shuffle = np.random.permutation(len(X_train))
        test_shuffle = np.random.permutation(len(X_test))
        
        return X_train[train_shuffle], X_test[test_shuffle], y_train[train_shuffle], y_test[test_shuffle]

class GaussianNaiveBayes:
    """
    高斯朴素贝叶斯分类器 
    特征条件独立 + 高斯分布假设 + MAP规则
    """
    
    def __init__(self, epsilon=1e-9):
        """
        初始化分类器
        epsilon: 平滑参数，防止除零错误
        """
        self.priors = None      # 先验概率 P(C_i)
        self.means = None       # 均值 μ (10×256矩阵)
        self.variances = None   # 方差 σ² (10×256矩阵) 
        self.classes = None     # 类别标签 [0,1,...,9]
        self.epsilon = epsilon  # 平滑参数
    
    def fit(self, X_train, y_train):
        """
        训练阶段：估计高斯分布参数
        思路：
        1. 确定类别数量（0-9共10类）
        2. 对每个类别，计算：
           - 先验概率：该类样本数 / 总样本数
           - 均值：该类所有样本每个特征的平均值
           - 方差：该类所有样本每个特征的方差
        """
        print("开始训练高斯朴素贝叶斯分类器...")
        
        # 获取所有类别（0-9）
        self.classes = np.unique(y_train)
        n_classes = len(self.classes)
        n_features = X_train.shape[1]  # 应该是256
        
        # 初始化参数矩阵
        self.priors = np.zeros(n_classes)
        self.means = np.zeros((n_classes, n_features))
        self.variances = np.zeros((n_classes, n_features))
        
        # 对每个类别进行参数估计
        for i, c in enumerate(self.classes):
            # 获取属于当前类别的所有样本
            X_c = X_train[y_train == c]
            
            # 计算先验概率
            self.priors[i] = len(X_c) / len(X_train)
            
            # 计算均值和方差
            self.means[i] = np.mean(X_c, axis=0)
            self.variances[i] = np.var(X_c, axis=0)
            
            print(f"类别 {c}: 样本数={len(X_c)}, 先验概率={self.priors[i]:.4f}")
        
        print("训练完成！")
        return self
    
    def _gaussian_pdf(self, x, mean, var):
        """
        高斯概率密度函数
        公式: P(x|μ,σ²) = (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))
        """
        # 防止方差为0
        var = var + self.epsilon
        coefficient = 1.0 / np.sqrt(2 * np.pi * var)
        exponent = -((x - mean) ** 2) / (2 * var)
        return coefficient * np.exp(exponent)
    
    def predict(self, X_test):
        """
        预测阶段：使用MAP规则进行分类
        思路：
        1. 对每个测试样本，计算属于每个类别的对数后验概率
        2. 对数后验概率 = 对数似然 + 对数先验
        3. 选择最大后验概率对应的类别
        """
        print("开始预测...")
        n_samples = X_test.shape[0]
        n_classes = len(self.classes)
        
        # 存储每个样本对每个类别的对数后验概率
        log_posteriors = np.zeros((n_samples, n_classes))
        
        for i in range(n_samples):  # 遍历每个测试样本
            for j in range(n_classes):  # 遍历每个类别
                # 计算对数似然：log P(X|C_j)
                # 由于特征独立，联合概率 = 各个特征概率的乘积
                # 对数形式：log(乘积) = 各个log的和
                log_likelihood = 0
                for k in range(X_test.shape[1]):  # 遍历每个特征（像素）
                    pdf = self._gaussian_pdf(X_test[i, k], self.means[j, k], self.variances[j, k])
                    # 防止概率为0
                    if pdf == 0:
                        pdf = self.epsilon
                    log_likelihood += np.log(pdf)
                
                # 计算对数先验：log P(C_j)
                log_prior = np.log(self.priors[j])
                
                # 对数后验概率 = 对数似然 + 对数先验
                log_posteriors[i, j] = log_likelihood + log_prior
        
        # 选择最大后验概率对应的类别
        predictions = np.argmax(log_posteriors, axis=1)
        print("预测完成！")
        return predictions
    
    def score(self, X, y):
        """计算分类准确率"""
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy

def load_data(filename):
    """加载数据文件"""
    raw_data = np.loadtxt(filename)
    X = raw_data[:, :256]  # 前256列是特征（像素值）
    y_one_hot = raw_data[:, 256:]  # 后10列是one-hot编码的标签
    y = np.argmax(y_one_hot, axis=1)  # 转换为类别标签
    return X, y

def main():
    """主程序：完整的实验流程"""
    
    # 1. 加载数据
    print("=== 步骤1: 加载数据 ===")
    X, y = load_data('semeion.data.txt')
    print(f"原始数据形状: X={X.shape}, y={y.shape}")
    print(f"类别分布: {np.bincount(y)}")
    
    # 2. 分层采样
    print("\n=== 步骤2: 分层采样（7:3比例）===")
    sampler = StratifiedSampler(test_size=0.3, random_state=42)
    X_train, X_test, y_train, y_test = sampler.split(X, y)
    
    print(f"训练集: {len(X_train)} 个样本")
    print(f"测试集: {len(X_test)} 个样本")
    print(f"训练集类别分布: {np.bincount(y_train)}")
    print(f"测试集类别分布: {np.bincount(y_test)}")
    
    # 3. 创建并训练分类器
    print("\n=== 步骤3: 训练朴素贝叶斯分类器 ===")
    gnb = GaussianNaiveBayes()
    gnb.fit(X_train, y_train)
    
    # 4. 预测并评估
    print("\n=== 步骤4: 模型评估 ===")
    train_accuracy = gnb.score(X_train, y_train)
    test_accuracy = gnb.score(X_test, y_test)
    
    print(f"训练集准确率: {train_accuracy:.4f}")
    print(f"测试集准确率: {test_accuracy:.4f}")
    
    # 5. 基础任务完成确认
    print("\n=== 基础任务完成! ===")
    print("✓ 分层采样已实现（7:3比例）")
    print("✓ 高斯朴素贝叶斯分类器已实现") 
    print("✓ 准确率计算已实现")

if __name__ == "__main__":
    main()