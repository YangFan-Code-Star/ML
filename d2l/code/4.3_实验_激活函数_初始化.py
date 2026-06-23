import torch
from torch import nn
from d2l import torch as d2l
import json

def train_and_evaluate(net, lr=0.1, num_epochs=10, batch_size=256):
    """训练模型并返回结果"""
    torch.manual_seed(42)
    
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    
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
        'final_train_acc': train_accs[-1],
        'final_test_acc': test_accs[-1],
        'train_accs': train_accs,
        'test_accs': test_accs
    }

def build_net(num_hiddens_list, activation_fn):
    """构建网络，支持多层和不同激活函数"""
    layers = [nn.Flatten()]
    
    # 输入层到第一个隐藏层
    layers.append(nn.Linear(784, num_hiddens_list[0]))
    layers.append(activation_fn())
    
    # 隐藏层之间
    for i in range(1, len(num_hiddens_list)):
        layers.append(nn.Linear(num_hiddens_list[i-1], num_hiddens_list[i]))
        layers.append(activation_fn())
    
    # 输出层
    layers.append(nn.Linear(num_hiddens_list[-1], 10))
    
    return nn.Sequential(*layers)

if __name__ == '__main__':
    results = {}
    
    # ============ 实验1: 不同隐藏层配置 ============
    print("=" * 60)
    print("实验1: 不同隐藏层配置")
    print("=" * 60)
    
    configs = [
        ([256], 0.1, "1层, 256单元, lr=0.1"),
        ([256], 0.05, "1层, 256单元, lr=0.05"),
        ([256], 0.2, "1层, 256单元, lr=0.2"),
        ([256, 128], 0.1, "2层, 256->128, lr=0.1"),
        ([256, 128], 0.05, "2层, 256->128, lr=0.05"),
        ([512, 256], 0.1, "2层, 512->256, lr=0.1"),
        ([256, 128, 64], 0.1, "3层, 256->128->64, lr=0.1"),
        ([256, 128, 64], 0.05, "3层, 256->128->64, lr=0.05"),
    ]
    
    results_layers = []
    for hiddens, lr, desc in configs:
        print(f"\n测试: {desc}...")
        net = build_net(hiddens, nn.ReLU)
        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, std=0.01)
        net.apply(init_weights)
        
        result = train_and_evaluate(net, lr=lr, num_epochs=10)
        result['config'] = desc
        results_layers.append(result)
        print(f"  训练准确率: {result['final_train_acc']:.4f}, 测试准确率: {result['final_test_acc']:.4f}")
    
    results['layer_configs'] = results_layers
    
    # ============ 实验2: 不同激活函数 ============
    print("\n" + "=" * 60)
    print("实验2: 不同激活函数")
    print("=" * 60)
    
    activations = [
        (nn.ReLU, "ReLU"),
        (nn.Sigmoid, "Sigmoid"),
        (nn.Tanh, "Tanh"),
        (nn.LeakyReLU, "LeakyReLU"),
        (nn.ELU, "ELU"),
    ]
    
    results_activations = []
    for act_fn, act_name in activations:
        print(f"\n测试激活函数: {act_name}...")
        net = build_net([256], act_fn)
        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, std=0.01)
        net.apply(init_weights)
        
        result = train_and_evaluate(net, lr=0.1, num_epochs=10)
        result['activation'] = act_name
        results_activations.append(result)
        print(f"  训练准确率: {result['final_train_acc']:.4f}, 测试准确率: {result['final_test_acc']:.4f}")
    
    results['activations'] = results_activations
    
    # ============ 实验3: 不同权重初始化方法 ============
    print("\n" + "=" * 60)
    print("实验3: 不同权重初始化方法")
    print("=" * 60)
    
    init_methods = [
        ("normal_0.01", lambda m: nn.init.normal_(m.weight, std=0.01) if type(m) == nn.Linear else None),
        ("normal_0.05", lambda m: nn.init.normal_(m.weight, std=0.05) if type(m) == nn.Linear else None),
        ("xavier_uniform", lambda m: nn.init.xavier_uniform_(m.weight) if type(m) == nn.Linear else None),
        ("xavier_normal", lambda m: nn.init.xavier_normal_(m.weight) if type(m) == nn.Linear else None),
        ("kaiming_uniform", lambda m: nn.init.kaiming_uniform_(m.weight, nonlinearity='relu') if type(m) == nn.Linear else None),
        ("kaiming_normal", lambda m: nn.init.kaiming_normal_(m.weight, nonlinearity='relu') if type(m) == nn.Linear else None),
        ("zeros", lambda m: nn.init.zeros_(m.weight) if type(m) == nn.Linear else None),
    ]
    
    results_inits = []
    for init_name, init_fn in init_methods:
        print(f"\n测试初始化方法: {init_name}...")
        net = build_net([256], nn.ReLU)
        net.apply(init_fn)
        
        result = train_and_evaluate(net, lr=0.1, num_epochs=10)
        result['init_method'] = init_name
        results_inits.append(result)
        print(f"  训练准确率: {result['final_train_acc']:.4f}, 测试准确率: {result['final_test_acc']:.4f}")
    
    results['init_methods'] = results_inits
    
    # 打印汇总
    print("\n" + "=" * 60)
    print("实验结果汇总")
    print("=" * 60)
    
    print("\n【实验1】隐藏层配置:")
    for r in results['layer_configs']:
        print(f"  {r['config']}: 训练={r['final_train_acc']:.4f}, 测试={r['final_test_acc']:.4f}")
    
    print("\n【实验2】激活函数:")
    for r in results['activations']:
        print(f"  {r['activation']:12s}: 训练={r['final_train_acc']:.4f}, 测试={r['final_test_acc']:.4f}")
    
    print("\n【实验3】初始化方法:")
    for r in results['init_methods']:
        print(f"  {r['init_method']:18s}: 训练={r['final_train_acc']:.4f}, 测试={r['final_test_acc']:.4f}")
    
    # 保存结果
    with open('d:/Work/ML/d2l/code/outputs/experiment_4.3_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n结果已保存到 outputs/experiment_4.3_results.json")
