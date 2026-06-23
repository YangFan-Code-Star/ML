import torch
from torch import nn
from d2l import torch as d2l
import time
import json
import random

def train_and_evaluate(num_hiddens, num_layers, lr, num_epochs, batch_size=256):
    """训练模型并返回结果"""
    # 设置随机种子
    torch.manual_seed(42)
    
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    
    num_inputs, num_outputs = 784, 10
    
    # 构建网络
    layers = []
    layers.append(nn.Flatten())
    
    # 第一层
    layers.append(nn.Linear(num_inputs, num_hiddens))
    layers.append(nn.ReLU())
    
    # 隐藏层
    for _ in range(num_layers - 1):
        layers.append(nn.Linear(num_hiddens, num_hiddens))
        layers.append(nn.ReLU())
    
    # 输出层
    layers.append(nn.Linear(num_hiddens, num_outputs))
    
    net = nn.Sequential(*layers)
    
    # 初始化权重
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)
    net.apply(init_weights)
    
    # 训练
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    
    device = d2l.try_gpu()
    net.to(device)
    
    train_losses = []
    train_accs = []
    test_accs = []
    
    for epoch in range(num_epochs):
        net.train()
        metric = d2l.Accumulator(3)
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
        
        train_loss = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]
        
        # 测试准确率
        net.eval()
        test_metric = d2l.Accumulator(2)
        with torch.no_grad():
            for X, y in test_iter:
                X, y = X.to(device), y.to(device)
                test_metric.add(d2l.accuracy(net(X), y), y.numel())
        test_acc = test_metric[0] / test_metric[1]
        
        train_losses.append(float(train_loss))
        train_accs.append(float(train_acc))
        test_accs.append(float(test_acc))
    
    return {
        'num_hiddens': num_hiddens,
        'num_layers': num_layers,
        'lr': lr,
        'num_epochs': num_epochs,
        'final_train_loss': train_losses[-1],
        'final_train_acc': train_accs[-1],
        'final_test_acc': test_accs[-1],
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_accs': test_accs
    }

def run_experiments():
    results = {}
    
    # ============ 实验1: 改变num_hiddens ============
    print("=" * 60)
    print("实验1: 改变隐藏单元数 num_hiddens")
    print("=" * 60)
    hiddens_list = [32, 64, 128, 256, 512, 1024]
    results_hiddens = []
    for h in hiddens_list:
        print(f"\n测试 num_hiddens={h}...")
        result = train_and_evaluate(num_hiddens=h, num_layers=1, lr=0.1, num_epochs=10)
        results_hiddens.append(result)
        print(f"  训练准确率: {result['final_train_acc']:.4f}, 测试准确率: {result['final_test_acc']:.4f}")
    results['hiddens_experiment'] = results_hiddens
    
    # ============ 实验2: 添加更多隐藏层 ============
    print("\n" + "=" * 60)
    print("实验2: 改变隐藏层数")
    print("=" * 60)
    layers_list = [1, 2, 3]
    results_layers = []
    for l in layers_list:
        print(f"\n测试 num_layers={l}...")
        result = train_and_evaluate(num_hiddens=256, num_layers=l, lr=0.1, num_epochs=10)
        results_layers.append(result)
        print(f"  训练准确率: {result['final_train_acc']:.4f}, 测试准确率: {result['final_test_acc']:.4f}")
    results['layers_experiment'] = results_layers
    
    # ============ 实验3: 改变学习率 ============
    print("\n" + "=" * 60)
    print("实验3: 改变学习率")
    print("=" * 60)
    lr_list = [0.001, 0.01, 0.1, 0.5, 1.0]
    results_lr = []
    for lr in lr_list:
        print(f"\n测试 lr={lr}...")
        result = train_and_evaluate(num_hiddens=256, num_layers=1, lr=lr, num_epochs=10)
        results_lr.append(result)
        print(f"  训练准确率: {result['final_train_acc']:.4f}, 测试准确率: {result['final_test_acc']:.4f}")
    results['lr_experiment'] = results_lr
    
    # ============ 实验4: 联合优化 ============
    print("\n" + "=" * 60)
    print("实验4: 联合优化超参数（随机搜索）")
    print("=" * 60)
    
    random.seed(42)
    
    search_space = {
        'num_hiddens': [64, 128, 256, 512],
        'num_layers': [1, 2, 3],
        'lr': [0.01, 0.05, 0.1, 0.3, 0.5],
        'num_epochs': [10, 20]
    }
    
    n_trials = 15
    best_result = None
    best_test_acc = 0
    results_joint = []
    
    print(f"\n进行{n_trials}次随机搜索...")
    for i in range(n_trials):
        h = random.choice(search_space['num_hiddens'])
        l = random.choice(search_space['num_layers'])
        lr = random.choice(search_space['lr'])
        ep = random.choice(search_space['num_epochs'])
        
        print(f"\n试验 {i+1}/{n_trials}: hiddens={h}, layers={l}, lr={lr}, epochs={ep}")
        result = train_and_evaluate(num_hiddens=h, num_layers=l, lr=lr, num_epochs=ep)
        results_joint.append(result)
        print(f"  测试准确率: {result['final_test_acc']:.4f}")
        
        if result['final_test_acc'] > best_test_acc:
            best_test_acc = result['final_test_acc']
            best_result = result
            print(f"  *** 新的最佳结果! ***")
    
    results['joint_optimization'] = {
        'best_result': best_result,
        'all_trials': results_joint
    }
    
    # 打印汇总
    print("\n" + "=" * 60)
    print("实验结果汇总")
    print("=" * 60)
    
    print("\n【实验1】num_hiddens 影响:")
    for r in results['hiddens_experiment']:
        print(f"  num_hiddens={r['num_hiddens']:4d}: 训练准确率={r['final_train_acc']:.4f}, 测试准确率={r['final_test_acc']:.4f}")
    
    print("\n【实验2】隐藏层数影响:")
    for r in results['layers_experiment']:
        print(f"  num_layers={r['num_layers']}: 训练准确率={r['final_train_acc']:.4f}, 测试准确率={r['final_test_acc']:.4f}")
    
    print("\n【实验3】学习率影响:")
    for r in results['lr_experiment']:
        print(f"  lr={r['lr']:6.3f}: 训练准确率={r['final_train_acc']:.4f}, 测试准确率={r['final_test_acc']:.4f}")
    
    print("\n【实验4】联合优化最佳结果:")
    br = results['joint_optimization']['best_result']
    print(f"  最佳超参数: num_hiddens={br['num_hiddens']}, num_layers={br['num_layers']}, lr={br['lr']}, num_epochs={br['num_epochs']}")
    print(f"  最佳测试准确率: {br['final_test_acc']:.4f}")
    
    # 保存结果
    with open('d:/Work/ML/d2l/code/outputs/experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n结果已保存到 outputs/experiment_results.json")
    return results

if __name__ == '__main__':
    run_experiments()
