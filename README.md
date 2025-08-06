# img_view_cv - 图像处理查看器

基于 OpenCV + Win32 GDI 的图像处理和查看器项目。

## 功能特性

- **图像显示**: 复用 reference/hello_rust 的 Win32/GDI 思路（菜单、文件对话框、StretchDIBits、HALFTONE、UTF-16 中文支持）
- **图像处理**: 严格按照 reference/py_spec.py 的逻辑实现，确保功能一致性
- **透视变换**: 色卡检测和透视矫正，支持多种检测方法
- **调试输出**: 在 ./debug_out 目录保存中间处理结果

## 技术架构

### 模块结构

```
src/
├── main.rs           - Win32窗口和菜单系统
├── ocv_bridge.rs     - OpenCV Mat 与 Win32 像素格式转换
└── cv_ops.rs         - 图像处理算法（严格遵循Python规格）
```

### 核心功能

1. **ocv_bridge.rs**
   - `ImageBuf` 结构：BGRA像素缓冲区
   - `mat_to_bgra()`: Mat转换为Win32显示格式
   - `save_debug_image()`: 保存调试图像

2. **cv_ops.rs**
   - `find_color_card_simplified()`: 简化版色卡检测
   - `perspective_transform_simplified()`: 自动透视变换
   - 通用图像处理函数：灰度、高斯、Canny、阈值等

3. **main.rs**
   - Win32窗口管理和DPI感知
   - 中文菜单系统
   - 图像加载和显示
   - 实时图像处理

## 构建和运行

### 前置要求

1. **Rust 工具链**
   ```bash
   rustup install stable
   ```

2. **OpenCV 依赖**
   - Windows: 确保系统已安装 OpenCV
   - 或使用 vcpkg 安装 OpenCV

### 编译项目

```bash
# 克隆或下载项目
cd img_view_cv

# 编译项目
cargo build --release

# 运行
cargo run
```

## 使用说明

### 基本操作

1. **启动程序**
   ```bash
   cargo run
   ```

2. **加载图像**
   - 使用菜单 "文件" → "打开文件"
   - 支持格式：JPG, PNG, BMP, TIFF

3. **图像处理操作**
   - **基础处理**: 灰度转换、高斯模糊
   - **边缘检测**: Canny边缘检测、自适应阈值
   - **形态学**: 闭运算、膨胀操作
   - **轮廓检测**: 轮廓查找和显示
   - **透视变换**: 自动色卡检测和矫正

### 测试透视变换

1. **准备测试图像**
   - 包含矩形色卡的图像
   - 确保色卡边界清晰可见

2. **操作流程**
   ```
   文件 → 打开文件 → 选择图像
   处理 → 透视变换(简化) 
   ```

3. **查看结果**
   - 窗口显示变换后的图像
   - 调试图像保存在 `./debug_out/` 目录

### 调试输出

程序会在 `./debug_out/` 目录生成以下调试图像：

```
debug_out/
├── 01_original.png           - 原始图像
├── 02_grayscale.png          - 灰度图像
├── 03_best_canny_edges.png   - 最佳Canny边缘
├── 04_method2_binary.png     - 方法2二值化
├── 05_method2_binary_inv.png - 方法2反转二值化
├── 06_method3_adaptive.png   - 方法3自适应阈值
└── 07_perspective_result.png - 透视变换结果
```

## 算法规格

### 色卡检测流程

严格按照 `reference/py_spec.py` 实现：

1. **方法1**: 多组Canny参数试探
   - 参数组: (30,100), (50,150), (100,200), (80,160)
   - 形态学闭运算 + 膨胀
   - 面积过滤: 图像总面积的5%
   - 综合评分: 面积权重70% + 形状权重30%

2. **方法2**: 简单二值化 (OTSU)
   - 正常和反转二值化
   - 最小面积阈值: 图像总面积的1%

3. **方法3**: 自适应二值化
   - 高斯自适应阈值
   - 块大小: 11, C值: 2.0

### 四点检测

- 轮廓近似精度因子: [0.02, 0.03, 0.04, 0.05, 0.06]
- 点排序: 左上、右上、右下、左下
- 基于坐标和差值的排序算法

## 性能特性

- **DPI感知**: Per-Monitor V2 DPI 支持
- **高质量渲染**: HALFTONE 拉伸模式
- **内存管理**: 智能指针和RAII模式
- **错误处理**: 全面的 `anyhow::Result` 错误处理

## 开发说明

### 代码约束

1. **Python逻辑优先**: 数值流程与 Python 版本严格一致
2. **可复现**: 调试输出与 Python 版本对齐
3. **错误处理**: 全部返回 `Result<_>`，不使用 `panic`
4. **多通道支持**: 1/3/4 通道 Mat 都能正确显示

### 扩展开发

- 修改 `cv_ops.rs` 添加新的图像处理算法
- 在 `main.rs` 中添加新的菜单项
- 使用 `ocv_bridge::save_debug_image()` 保存调试图像

## 许可证

本项目基于参考代码开发，用于学习和研究目的。
