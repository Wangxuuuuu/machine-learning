import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
import seaborn as sns

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体（如果系统没有该字体，可以改为其他中文字体如'Microsoft YaHei'）
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

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

class ModelEvaluator:
    """模型评估器：实现混淆矩阵、精度、召回率、F1值计算"""
    
    def __init__(self, y_true, y_pred, classes):
        """
        初始化评估器
        Parameters:
            y_true: 真实标签
            y_pred: 预测标签
            classes: 类别列表 [0,1,...,9]
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.classes = classes
        self.n_classes = len(classes)
        
    def compute_confusion_matrix(self):
        """计算混淆矩阵"""
        print("=== 计算混淆矩阵 ===")
        
        # 初始化10×10的混淆矩阵
        cm = np.zeros((self.n_classes, self.n_classes), dtype=int)
        
        # 遍历每个样本，填充混淆矩阵
        for true_label, pred_label in zip(self.y_true, self.y_pred):
            cm[true_label, pred_label] += 1
        
        self.confusion_matrix = cm
        return cm
    
    def plot_confusion_matrix(self, cm, normalize=False):
        """可视化混淆矩阵"""
        plt.figure(figsize=(10, 8))
        
        if normalize:
            # 归一化混淆矩阵（按行）
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                       xticklabels=self.classes, yticklabels=self.classes)
            plt.title('归一化混淆矩阵')
        else:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.classes, yticklabels=self.classes)
            plt.title('混淆矩阵')
        
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.tight_layout()
        plt.show()
    
    def compute_classification_metrics(self):
        """计算每个类别的精度、召回率和F1值"""
        print("\n=== 计算分类指标 ===")
        
        # 初始化存储各指标的字典
        self.precision_dict = {}
        self.recall_dict = {}
        self.f1_dict = {}
        
        # 为每个类别计算指标
        for class_label in self.classes:
            # 计算TP, FP, FN
            tp = self.confusion_matrix[class_label, class_label]
            fp = np.sum(self.confusion_matrix[:, class_label]) - tp
            fn = np.sum(self.confusion_matrix[class_label, :]) - tp
            
            # 计算精度、召回率、F1值
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # 存储结果
            self.precision_dict[class_label] = precision
            self.recall_dict[class_label] = recall
            self.f1_dict[class_label] = f1
            
            print(f"数字 {class_label}:")
            print(f"  TP={tp}, FP={fp}, FN={fn}")
            print(f"  精度={precision:.4f}, 召回率={recall:.4f}, F1值={f1:.4f}")
            print("-" * 40)
    
    def compute_macro_average(self):
        """计算宏平均（各类别指标的平均值）"""
        print("\n=== 宏平均指标 ===")
        
        macro_precision = np.mean(list(self.precision_dict.values()))
        macro_recall = np.mean(list(self.recall_dict.values()))
        macro_f1 = np.mean(list(self.f1_dict.values()))
        
        print(f"宏平均精度: {macro_precision:.4f}")
        print(f"宏平均召回率: {macro_recall:.4f}")
        print(f"宏平均F1值: {macro_f1:.4f}")
        
        return macro_precision, macro_recall, macro_f1
    
    def compute_micro_average(self):
        """微平均计算"""
        print("\n=== 微平均指标===")
        
        # 计算总的TP, FP, FN
        total_tp = np.sum(np.diag(self.confusion_matrix))
        total_fp = np.sum(self.confusion_matrix) - total_tp
        total_fn = total_fp  # 在多分类中，总的FP=FN
        
        # 总样本数
        total_samples = np.sum(self.confusion_matrix)
        
        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
        
        print(f"微平均精度: {micro_precision:.4f}")
        print(f"微平均召回率: {micro_recall:.4f}") 
        print(f"微平均F1值: {micro_f1:.4f}")
        print(f"总样本数: {total_samples}")
        print(f"总TP: {total_tp}")
        
        return micro_precision, micro_recall, micro_f1
    
    def comprehensive_evaluation(self):
        """综合评估：执行所有计算"""
        # 1. 计算混淆矩阵
        cm = self.compute_confusion_matrix()
        
        # 2. 可视化混淆矩阵
        self.plot_confusion_matrix(cm, normalize=False)
        self.plot_confusion_matrix(cm, normalize=True)
        
        # 3. 计算分类指标
        self.compute_classification_metrics()
        
        # 4. 计算平均指标
        self.compute_macro_average()
        self.compute_micro_average()
        
        return cm

# 修改主函数，加入评估部分
def main():
    """主程序：完整的实验流程"""
    
    # 1. 加载数据
    print("=== 步骤1: 加载数据 ===")
    X_train = np.load('X_train.npy')
    X_test = np.load('X_test.npy')
    y_train = np.load('y_train.npy')
    y_test = np.load('y_test.npy')
    
    print(f"训练集: {X_train.shape}")
    print(f"测试集: {X_test.shape}")
    
    # 2. 创建并训练分类器
    print("\n=== 步骤2: 训练朴素贝叶斯分类器 ===")
    gnb = GaussianNaiveBayes()
    gnb.fit(X_train, y_train)
    
    # 3. 预测
    print("\n=== 步骤3: 模型预测 ===")
    y_pred = gnb.predict(X_test)
    
    # 4. 基础准确率评估
    accuracy = np.mean(y_pred == y_test)
    print(f"\n基础准确率: {accuracy:.4f}")
    
    # 5. 中级任务：多维度评估
    print("\n" + "="*50)
    print("开始中级任务：多维度模型评估")
    print("="*50)
    
    # 创建评估器
    evaluator = ModelEvaluator(y_test, y_pred, classes=np.arange(10))
    
    # 执行综合评估
    cm = evaluator.comprehensive_evaluation()
    
    # 6. 结果总结
    print("\n" + "="*50)
    print("中级任务完成总结")
    print("="*50)
    print("✓ 混淆矩阵计算与可视化 - 完成")
    print("✓ 精度/召回率/F1值计算 - 完成") 
    print("✓ 宏平均/微平均计算 - 完成")
    print("✓ 多维度模型评估 - 完成")

if __name__ == "__main__":
    main()