# FoundationPose 从零部署运行指南（Windows + Docker + RTX 50 系列）

本文档记录如何在一台全新的 Windows 电脑上，从 0 开始部署并运行 FoundationPose。重点适用于 RTX 50 系列显卡，例如 RTX 5070 Ti、RTX 5080、RTX 5090。

如果你的显卡是 RTX 30/40 系列，也可以参考本文档的整体流程，但不一定需要 RTX50 专用镜像。RTX 50 系列必须使用支持 `sm_120` 的 CUDA 12.8+ / PyTorch 2.7+，否则会遇到：

```text
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

## 1. 整体思路

FoundationPose 官方推荐 Docker 部署。不要直接在 Windows 宿主机的 Python 或 Conda 环境里运行 `run_demo.py`，因为它依赖 CUDA、PyTorch3D、NVDiffRast、Open3D、C++ 扩展等，Windows 原生环境容易出现依赖和编译问题。

推荐方案：

1. Windows 安装 NVIDIA 显卡驱动。
2. Windows 安装 Docker Desktop，并启用 WSL2 后端。
3. 下载 FoundationPose 项目代码。
4. 下载官方权重和 demo 数据。
5. 使用 `docker/Dockerfile.rtx50` 构建 RTX50 专用镜像。
6. 在 Docker 容器中编译 FoundationPose 的 `mycpp` 扩展。
7. 在 Docker 容器中运行 `run_demo.py --debug 0`。

## 2. 新电脑硬件和系统要求

建议配置：

- Windows 10/11 64 位。
- NVIDIA 独立显卡，推荐显存 12GB 以上。
- RTX 50 系列显卡需要 CUDA 12.8+ / PyTorch 2.7+ 环境。
- 内存建议 32GB 以上。
- 磁盘可用空间建议至少 80GB。
- 网络需要能下载 Docker 镜像、Python 包、GitHub 代码、Google Drive 权重和数据。

本文档已验证环境：

```text
OS: Windows
GPU: NVIDIA GeForce RTX 5070 Ti
Docker image: foundationpose-rtx50:latest
CUDA in container: 12.8
PyTorch: 2.7.0+cu128
GPU capability: sm_120
```

## 3. 安装 NVIDIA 驱动

先安装 NVIDIA 官方驱动：

1. 打开 NVIDIA 驱动下载页面：
   <https://www.nvidia.cn/Download/index.aspx>
2. 选择你的显卡型号，例如 RTX 5070 Ti。
3. 下载并安装最新 Game Ready Driver 或 Studio Driver。
4. 安装完成后重启电脑。
5. 打开 PowerShell，检查驱动是否正常：

```powershell
nvidia-smi
```

如果正常，会看到显卡型号、驱动版本、CUDA Version 等信息。

注意：这里显示的 `CUDA Version` 是驱动支持的最高 CUDA 版本，不代表你已经安装了 CUDA Toolkit。使用 Docker 时通常不需要在 Windows 宿主机单独安装 CUDA Toolkit。

## 4. 安装 Docker Desktop

下载 Docker Desktop：

<https://www.docker.com/products/docker-desktop/>

安装时建议：

1. 勾选使用 WSL2 后端。
2. 安装完成后重启电脑。
3. 打开 Docker Desktop，等待 Docker Engine 启动。
4. 打开 PowerShell，检查 Docker：

```powershell
docker version
docker ps
```

检查 Docker 是否能访问 GPU：

```powershell
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

如果能在容器输出里看到 NVIDIA 显卡，说明 Docker GPU 支持正常。

如果该命令拉取镜像很慢，可以先继续后面的 FoundationPose 镜像构建，因为本文提供的 `Dockerfile.rtx50` 已使用镜像源。

## 5. 下载 FoundationPose 代码

建议在 `D:\Work\ML` 这类路径下存放项目。

打开 PowerShell：

```powershell
cd D:\Work\ML
git clone https://github.com/NVlabs/FoundationPose.git FoundationPose-main
cd FoundationPose-main
```

如果你不是从 GitHub 克隆，而是复制已有项目目录，也可以，但需要确保目录结构完整。

最终目录类似：

```text
D:\Work\ML\FoundationPose-main
├─ run_demo.py
├─ estimater.py
├─ datareader.py
├─ Utils.py
├─ docker
├─ weights
├─ demo_data
└─ ...
```

## 6. 准备 RTX50 专用 Docker 文件

本项目里需要有以下两个文件：

```text
docker/Dockerfile.rtx50
docker/run_container_rtx50.ps1
```

其中 `Dockerfile.rtx50` 用于构建支持 RTX 50 系列的环境，基础镜像是：

```dockerfile
docker.m.daocloud.io/pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel
```

关键点：

- PyTorch 版本为 `2.7.0+cu128`。
- CUDA 为 `12.8`。
- 支持 `sm_120`。
- 安装 `nvdiffrast` 和 `pytorch3d` 时使用 `--no-build-isolation`。
- 使用 `opencv-python-headless`，避免 Docker 无图形界面时出现 OpenCV 窗口问题。

如果新电脑上的项目没有这两个文件，把当前项目的这两个文件复制过去。

## 7. 下载权重和 demo 数据

FoundationPose demo 需要两类文件：

1. 网络权重，放到 `weights/`。
2. demo 数据，放到 `demo_data/`。

### 7.1 下载权重

官方 README 中的权重地址：

<https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i?usp=sharing>

下载后放到：

```text
FoundationPose-main/weights/
```

至少需要这两个目录：

```text
weights/
├─ 2023-10-28-18-33-37/
│  ├─ model_best.pth
│  └─ config.yml
└─ 2024-01-11-20-02-45/
   ├─ model_best.pth
   └─ config.yml
```

其中：

- `2023-10-28-18-33-37` 是 refiner 权重。
- `2024-01-11-20-02-45` 是 scorer 权重。

### 7.2 下载 demo 数据

官方 demo 数据地址：

<https://drive.google.com/drive/folders/1pRyFmxYXmAnpku7nGRioZaKrVJtIsroP?usp=sharing>

下载并解压后放到：

```text
FoundationPose-main/demo_data/
```

至少需要默认 demo 数据：

```text
demo_data/
└─ mustard0/
   ├─ mesh/
   │  └─ textured_simple.obj
   ├─ rgb/
   ├─ depth/
   ├─ masks/
   └─ cam_K.txt
```

`run_demo.py` 默认会读取：

```text
demo_data/mustard0/mesh/textured_simple.obj
demo_data/mustard0
```

## 8. 构建 RTX50 Docker 镜像

进入项目根目录：

```powershell
cd D:\Work\ML\FoundationPose-main
```

构建镜像：

```powershell
docker build -t foundationpose-rtx50:latest -f .\docker\Dockerfile.rtx50 .
```

也可以使用 PowerShell 脚本构建：

```powershell
powershell -ExecutionPolicy Bypass -File .\docker\run_container_rtx50.ps1 -Build -Command "python -c 'print(123)'"
```

第一次构建会比较久，因为要下载：

- PyTorch CUDA 12.8 基础镜像。
- Ubuntu 系统依赖。
- Python 依赖。
- `nvdiffrast` 源码并编译。
- `pytorch3d` 源码并编译。

如果网络慢，可能需要几十分钟到数小时。构建成功后，检查镜像：

```powershell
docker images foundationpose-rtx50
```

## 9. 验证 PyTorch 是否支持 RTX50

运行：

```powershell
docker run --rm --gpus all foundationpose-rtx50:latest python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0)); print(torch.cuda.get_arch_list()); print(torch.randn(1, device='cuda'))"
```

正常输出应包含类似信息：

```text
2.7.0+cu128
12.8
True
NVIDIA GeForce RTX 5070 Ti
(12, 0)
['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120', 'compute_120']
tensor([...], device='cuda:0')
```

关键检查点：

- `torch.cuda.is_available()` 必须是 `True`。
- `torch.cuda.get_device_capability(0)` 应该是 `(12, 0)`。
- `torch.cuda.get_arch_list()` 必须包含 `sm_120`。
- CUDA tensor 创建不能报错。

如果这里报 `no kernel image is available for execution on the device`，说明你没有使用正确的 RTX50 镜像，或者 PyTorch 不是 CUDA 12.8+ 版本。

## 10. 编译 FoundationPose 的 mycpp 扩展

FoundationPose 需要编译 `mycpp` 扩展，否则姿态聚类等功能可能不可用。

在项目根目录运行：

```powershell
docker run --rm --gpus all -v "${PWD}:/workspace/FoundationPose" -w /workspace/FoundationPose foundationpose-rtx50:latest bash -lc 'set -euo pipefail; export FORCE_CUDA=1; export TORCH_CUDA_ARCH_LIST=12.0; cd mycpp; mkdir -p build; cd build; cmake .. -DPYTHON_EXECUTABLE=$(which python); make -j$(nproc)'
```

成功后会看到：

```text
[100%] Built target mycpp
```

并在项目中生成类似文件：

```text
mycpp/build/mycpp.cpython-311-x86_64-linux-gnu.so
```

注意：不要在 Windows 宿主机直接编译这个 `.so`，它是给 Linux Docker 容器使用的。

## 11. 运行 demo

推荐先用无窗口模式运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\docker\run_container_rtx50.ps1 -Command "python run_demo.py --debug 0"
```

也可以直接使用 `docker run`：

```powershell
docker run --rm --gpus all -v "${PWD}:/workspace/FoundationPose" -w /workspace/FoundationPose foundationpose-rtx50:latest python run_demo.py --debug 0
```

第一次运行会稍慢，因为 `warp-lang` 等组件可能会在线编译 CUDA kernel。

运行成功后，会在项目目录生成：

```text
debug/
└─ ob_in_cam/
   ├─ xxxxxxxxx.txt
   ├─ xxxxxxxxx.txt
   └─ ...
```

每个 `.txt` 文件是对应帧的 4x4 位姿矩阵。

成功运行时，日志末尾可能包含：

```text
Warp ... initialized
Devices:
  "cuda:0" : "NVIDIA GeForce RTX 5070 Ti" (... sm_120 ...)
```

## 12. 关于 debug 参数

`run_demo.py` 默认是：

```text
--debug 1
```

但在 Windows + Docker 下，不建议直接用 `--debug 1`，因为脚本会执行：

```python
cv2.imshow('1', vis[...,::-1])
cv2.waitKey(1)
```

Docker 容器默认没有图形窗口环境，容易导致显示失败。

建议：

- 首次验证使用 `--debug 0`。
- 如果只需要位姿结果，继续使用 `--debug 0`。
- 如果需要保存可视化图像，可以尝试 `--debug 2`，但仍不建议使用 `cv2.imshow`。
- 如果要真正显示 OpenCV 窗口，需要额外配置 Windows X Server 或改代码保存图片，不在本文默认流程内。

## 13. 使用自己的物体和数据

默认 demo 使用 mustard bottle：

```text
--mesh_file demo_data/mustard0/mesh/textured_simple.obj
--test_scene_dir demo_data/mustard0
```

如果要换自己的物体，需要准备：

1. 物体 CAD 模型，例如 `.obj`。
2. RGB 图像序列。
3. 深度图序列。
4. 第一帧物体 mask。
5. 相机内参。

运行示例：

```powershell
powershell -ExecutionPolicy Bypass -File .\docker\run_container_rtx50.ps1 -Command "python run_demo.py --mesh_file /workspace/FoundationPose/demo_data/my_object/mesh/model.obj --test_scene_dir /workspace/FoundationPose/demo_data/my_object --debug 0"
```

注意容器内路径是：

```text
/workspace/FoundationPose
```

而不是 Windows 路径 `D:\Work\ML\FoundationPose-main`。

## 14. 常见问题

### 14.1 `ModuleNotFoundError: No module named 'trimesh'`

原因：你在 Windows 宿主机 Python 里运行了 `python run_demo.py`，而不是在 Docker 里运行。

错误方式：

```powershell
python run_demo.py
```

正确方式：

```powershell
powershell -ExecutionPolicy Bypass -File .\docker\run_container_rtx50.ps1 -Command "python run_demo.py --debug 0"
```

### 14.2 `CUDA error: no kernel image is available for execution on the device`

原因：PyTorch 不支持 RTX 50 系列的 `sm_120`。

解决：

1. 使用 `foundationpose-rtx50:latest` 镜像。
2. 确认 PyTorch 是 `2.7.0+cu128` 或更高。
3. 确认 `torch.cuda.get_arch_list()` 包含 `sm_120`。

检查命令：

```powershell
docker run --rm --gpus all foundationpose-rtx50:latest python -c "import torch; print(torch.__version__); print(torch.cuda.get_arch_list())"
```

### 14.3 Docker 构建时 Ubuntu 包下载失败

可能原因：网络或 Ubuntu 官方源临时失败。

本文的 `Dockerfile.rtx50` 已做了两件事：

- 将 apt 源切到 `mirrors.aliyun.com`。
- 给 apt 增加 `Acquire::Retries=5`。

如果仍然失败，直接重新执行构建命令即可：

```powershell
docker build -t foundationpose-rtx50:latest -f .\docker\Dockerfile.rtx50 .
```

Docker 会复用成功的缓存层。

### 14.4 `nvdiffrast` 编译失败，提示需要 `--no-build-isolation`

原因：`nvdiffrast` 构建时需要访问当前环境里的 PyTorch。

解决：确保 Dockerfile 中安装命令是：

```bash
python -m pip install --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast.git"
```

### 14.5 `cv2.imshow` 无法显示窗口

原因：Docker 容器默认没有 Windows 图形显示环境。

解决：使用无窗口模式：

```powershell
powershell -ExecutionPolicy Bypass -File .\docker\run_container_rtx50.ps1 -Command "python run_demo.py --debug 0"
```

### 14.6 找不到权重文件

错误通常类似：

```text
FileNotFoundError: ... weights/.../model_best.pth
```

检查目录：

```powershell
dir .\weights
dir .\weights\2023-10-28-18-33-37
dir .\weights\2024-01-11-20-02-45
```

确认两个目录里都有 `model_best.pth` 和 `config.yml`。

### 14.7 找不到 demo 数据

错误通常发生在读取 mesh、RGB、depth、mask 时。

检查：

```powershell
dir .\demo_data\mustard0
dir .\demo_data\mustard0\mesh
```

确认存在：

```text
demo_data/mustard0/mesh/textured_simple.obj
```

## 15. 推荐的一键流程

如果项目、权重和 demo 数据都已经准备好，在新电脑上通常只需要执行：

```powershell
cd D:\Work\ML\FoundationPose-main

docker build -t foundationpose-rtx50:latest -f .\docker\Dockerfile.rtx50 .

docker run --rm --gpus all foundationpose-rtx50:latest python -c "import torch; print(torch.__version__); print(torch.cuda.get_arch_list()); print(torch.randn(1, device='cuda'))"

docker run --rm --gpus all -v "${PWD}:/workspace/FoundationPose" -w /workspace/FoundationPose foundationpose-rtx50:latest bash -lc 'set -euo pipefail; export FORCE_CUDA=1; export TORCH_CUDA_ARCH_LIST=12.0; cd mycpp; mkdir -p build; cd build; cmake .. -DPYTHON_EXECUTABLE=$(which python); make -j$(nproc)'

powershell -ExecutionPolicy Bypass -File .\docker\run_container_rtx50.ps1 -Command "python run_demo.py --debug 0"
```

如果最后一条命令正常结束，并且生成了：

```text
debug/ob_in_cam/*.txt
```

说明 FoundationPose 已经可以在新电脑上正常运行。

## 16. 不建议的做法

不要直接在 Windows 宿主机运行：

```powershell
python run_demo.py
```

原因：

- 宿主机 Python 可能缺少依赖。
- Windows 原生环境编译 PyTorch3D、NVDiffRast、C++ 扩展难度高。
- RTX 50 系列需要特定 PyTorch CUDA 版本。
- `run_demo.py` 内部有 Linux 风格命令：

```python
os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')
```

这行在 Docker Linux 容器里没问题，但在 Windows 宿主机里不适合直接运行。

## 17. 最终检查清单

新电脑部署完成后，逐项确认：

- `nvidia-smi` 能看到显卡。
- `docker ps` 能正常执行。
- `docker run --rm --gpus all ... nvidia-smi` 能在容器中看到显卡。
- `foundationpose-rtx50:latest` 镜像存在。
- `torch.cuda.is_available()` 是 `True`。
- `torch.cuda.get_arch_list()` 包含 `sm_120`。
- `weights/2023-10-28-18-33-37/model_best.pth` 存在。
- `weights/2024-01-11-20-02-45/model_best.pth` 存在。
- `demo_data/mustard0/mesh/textured_simple.obj` 存在。
- `mycpp/build/mycpp.cpython-311-x86_64-linux-gnu.so` 已生成。
- `python run_demo.py --debug 0` 在 Docker 中能完整跑完。
- `debug/ob_in_cam/` 下生成位姿 `.txt` 文件。

