import numpy as np
import random
import pickle
import os

class GeneticNeuralNetwork:
    def __init__(self, input_size=49, hidden_layers=[16, 8], output_size=4):
        """
        :param input_size: 输入层神经元数量
        :param hidden_layers: 隐藏层神经元数量列表
        :param output_size: 输出层神经元数量（动作空间）
        """
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size

        # 随机初始化网络权重和偏置
        self.weights = []
        self.biases = []

        # 输入层到第一隐藏层
        self.weights.append(np.random.randn(input_size, hidden_layers[0]) * 0.1)
        self.biases.append(np.random.randn(1, hidden_layers[0]) * 0.1)  # 改为随机初始化

        # 隐藏层之间
        for i in range(1, len(hidden_layers)):
            self.weights.append(np.random.randn(hidden_layers[i - 1], hidden_layers[i]) * 0.1)
            self.biases.append(np.random.randn(1, hidden_layers[i]) * 0.1)

        # 输出层
        self.weights.append(np.random.randn(hidden_layers[-1], output_size) * 0.1)
        self.biases.append(np.random.randn(1, output_size) * 0.1)

        self.fitness = 0

    def relu(self, x):
        """ReLU激活函数"""
        return np.maximum(0, x)

    def softmax(self, x):
        """Softmax输出层，将输出转换为概率分布"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def forward(self, inputs):
        layer = inputs
        for i in range(len(self.weights)):
            layer = np.dot(layer, self.weights[i]) + self.biases[i]
            if i != len(self.weights) - 1:  # 如果不是最后一层
                layer = self.relu(layer)

        # 最后一层使用softmax
        return self.softmax(layer)

    def print_weights_biases(self):
        """打印可直接复制到游戏环境的权重和偏置代码"""
        print("\n===== 可直接复制的神经网络参数 =====")
        print("nn.weights = [")
        for w in self.weights:
            print("    np.array([")
            for row in w:
                print("        [" + ", ".join(f"{x:.3f}" for x in row) + "],")
            print("    ]),")
        print("]")

        print("\nnn.biases = [")
        for b in self.biases:
            print("    np.array([")
            # 偏置只有一行，直接打印
            print("        [" + ", ".join(f"{x:.3f}" for x in b[0]) + "],")
            print("    ]),")
        print("]")
        print("===============================\n")

    def mutate(self, mutation_rate=0.2, mutation_strength=0.2):
        for i in range(len(self.weights)):
            # 权重突变
            mutation_mask = np.random.random(self.weights[i].shape) < mutation_rate
            self.weights[i] += mutation_mask * np.random.normal(0, mutation_strength, self.weights[i].shape)

            # 偏置突变（新增部分）
            bias_mask = np.random.random(self.biases[i].shape) < mutation_rate
            self.biases[i] += bias_mask * np.random.normal(0, mutation_strength, self.biases[i].shape)

            # 10%概率大变异（同时作用于权重和偏置）
            if random.random() < 0.1:
                # 权重大变异
                big_mut_w = np.random.normal(0, mutation_strength * 3, self.weights[i].shape)
                self.weights[i] += big_mut_w * (np.random.random(self.weights[i].shape) < 0.2)

                # 偏置大变异
                big_mut_b = np.random.normal(0, mutation_strength * 3, self.biases[i].shape)
                self.biases[i] += big_mut_b * (np.random.random(self.biases[i].shape) < 0.2)


def crossover(parent1, parent2):
    child = GeneticNeuralNetwork(
        input_size=parent1.input_size,
        hidden_layers=parent1.hidden_layers,
        output_size=parent1.output_size
    )

    for i in range(len(parent1.weights)):
        if random.random() < 0.8:  # 70%概率使用块交叉
            # 随机切分块交叉
            split_row = random.randint(1, parent1.weights[i].shape[0] - 1)
            split_col = random.randint(1, parent1.weights[i].shape[1] - 1)
            child.weights[i] = np.block([
                [parent1.weights[i][:split_row, :split_col], parent2.weights[i][:split_row, split_col:]],
                [parent2.weights[i][split_row:, :split_col], parent1.weights[i][split_row:, split_col:]]
            ])
        else:  # 30%概率使用逐点交叉
            mask = np.random.random(parent1.weights[i].shape) > 0.5
            child.weights[i] = np.where(mask, parent1.weights[i], parent2.weights[i])

        # 偏置统一用逐点交叉
        b_mask = np.random.random(parent1.biases[i].shape) > 0.5
        child.biases[i] = np.where(b_mask, parent1.biases[i], parent2.biases[i])

    return child


def save_population(population, filename='saved_population.pkl'):
    """保存当前种群到文件"""
    with open(filename, 'wb') as f:
        pickle.dump(population, f)
    print(f"种群已保存到 {filename}")

def load_population(filename='saved_population.pkl'):
    """从文件加载种群"""
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            population = pickle.load(f)
        print(f"从 {filename} 加载了上次保存的种群")
        return population
    else:
        print(f"未找到保存文件 {filename}，使用新种群")
        return None

def genetic_algorithm(population_size=8, generations=10, load_previous=False):
    # 初始化种群
    if load_previous:
        population = load_population()

    if not load_previous or population is None:
        population = [GeneticNeuralNetwork() for _ in range(population_size)]

    for generation in range(generations):
        print(f"\n=== 第 {generation + 1} 代 ===")

        # 手动输入适应度
        fitness_list = []
        for i, nn in enumerate(population):
            nn.print_weights_biases()
            user_input = input(f"请输入第 {i + 1} 个神经网络的适应度(或输入'exit'保存退出): ")

            if user_input.lower() == 'exit':
                save_population(population)
                print("程序已退出")
                return population, None  # 提前退出时不返回最佳个体

            try:
                fitness = float(user_input)
                fitness_list.append(fitness)
            except ValueError:
                print("无效输入，请输入数字或'exit'")
                i -= 1  # 重新输入
                continue


        # 分配适应度
        for nn, fitness in zip(population, fitness_list):
            nn.fitness = fitness

        # 选择最佳个体
        population = sorted(population, key=lambda x: x.fitness, reverse=True)
        best_nn = population[0]

        print(f"\n本代最佳适应度: {best_nn.fitness}")
        print("最佳神经网络参数已保留到下一代")

        # 创建新一代
        new_population = [best_nn]  # 保留最佳个体

        while len(new_population) < population_size:
            # 确保选择两个不同的父代
            top_half = population[:population_size // 2]  # 前50%优秀个体
            parent1 = max(random.sample(top_half, 2), key=lambda x: x.fitness)
            parent2 = max(random.sample([p for p in top_half if p != parent1], 2), key=lambda x: x.fitness)

            child = crossover(parent1, parent2)
            child.mutate()
            new_population.append(child)

        population = new_population

    # 返回最终种群和最佳个体
    best_nn = max(population, key=lambda x: x.fitness)
    print("\n=== 训练结束 ===")
    print(f"最终最佳适应度: {best_nn.fitness}")
    print("最佳神经网络参数:")
    best_nn.print_weights_biases()
    return population, best_nn


# 使用示例
if __name__ == "__main__":
    # 加载上次保存的种群(True)或新建种群(False)
    final_population, best_network = genetic_algorithm(
        population_size=8,
        generations=100,
        load_previous=True  # 改为True以加载上次保存的
    )