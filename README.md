# Image-processing-rust (`img_view_cv`)

Windows 原生图像查看 / 处理工具。界面用 Win32 + GDI，图像算法走 OpenCV，交互抠图用 MobileSAM（ViT-T ONNX）。

适合对着照片、扫描件、色卡做查看、矫正、测色和分割，而不是通用相册。

仓库：<https://github.com/sutony1/Image-processing-rust>

色卡向导和命令行备忘：[docs/色卡向导与CLI.md](docs/色卡向导与CLI.md)

## Windows 便携版

从 [GitHub Releases](https://github.com/sutony1/Image-processing-rust/releases/latest) 下载 Windows x64 ZIP，解压整个目录后运行 `img_view_cv.exe`。压缩包内含 SAM 模型、ONNX Runtime 和使用说明。

## 功能

### 文件

- 打开 / 保存 / 另存为 / 关闭（JPG、PNG、BMP、TIFF 等）
- 「保存」覆盖当前文件。「另存为」可在可缩放的窗口中选择位置和格式，保存成功后切换到新文件。
- 拖拽文件到窗口打开
- 同目录上一张 / 下一张（窗口左右箭头）
- 标准 Windows 程序启动：只出图形窗口，不再附带控制台

### 查看

- 滚轮缩放（以鼠标位置为中心）
- 拖拽平移
- 大图智能加载：长边超过 2048px 先缩小显示，保存和透视时再读原图
- Per-Monitor V2 DPI，HALFTONE 拉伸
- EXIF 拍摄信息面板（机身、光圈、快门、ISO、焦距、色彩空间）

### 透视矫正

- 「透视矫正 → 手动四点矫正」适用于画作、文档等平面。依次点击左上、右上、右下、左下，第四点后显示展平结果，再按需要保存。
- 大图按原始分辨率计算四点矫正。

### 色卡测色

- **色卡测色向导**：按当前进度往下走，每步有对话框说明
- 拍歪的照片：框选出整张色卡（SAM 抠图）→ 自动拉正（失败可改手动四点）→ 读 24 色
- 已经拉正、铺满画面的色卡：向导里选「否」，直接读色
- 标题栏显示进度：未框选 / 已抠图待拉正 / 已拉正待读色 / 已读取颜色
- 未完成的步骤在菜单里是灰色的

### 处理

- 灰度、Canny、自适应阈值
- 恢复原图
- 旋转：90° / 180° / 任意角
- 拉直：画线后按水平校正
- 调整大小（插值、质量）
- 批量缩图（长边限制、质量、插值）

### 选框

- 框选后裁剪
- 框选后涂抹（模糊）

### SAM 分割

- 「SAM交互分割抠像」：加载模型并编码当前图，按住左键拖框包住对象，松开后分割
- 结果用绿色半透明遮罩叠在原图上
- 「抠图并显示」：按 mask 抠出对象
- 灰度图会禁用 SAM 菜单
- 模型文件：程序旁的 `vit_t_encoder.onnx`、`vit_t_decoder.onnx`（MobileSAM ViT-T）

透视、色卡等中间步骤会写到 `./debug_out/`。

## 界面菜单

| 菜单 | 内容 |
|---|---|
| 文件 | 打开、保存、另存为、关闭、退出 |
| 透视矫正 | 手动四点矫正、色卡测色向导、框出色卡、自动/手动拉正色卡、读取色块、恢复原图 |
| 处理 | 恢复原图、灰度、Canny、自适应阈值、拍摄信息、缩放、旋转 |
| SAM分割 | 交互分割抠像、抠图并显示 |
| 拉直 | 画直线、水平校正 |
| 选框工具 | 选框、裁剪、涂抹 |
| 批量处理 | 批量缩图 |
| 帮助 | 使用说明（带插图的本地网页） |

## 色卡用法

1. 打开色卡照片
2. 菜单 **透视矫正 → 色卡测色向导**
3. 拍歪的照片选「是」，按提示拖框包住色卡；已经拍正的选「否」直接读色
4. 向导会依次问：自动拉正 → 读取 24 色

也可以不用向导，按菜单里的 1、2、3 单步做。自动透视失败时用「手动四点」。

## 环境要求

- Windows 10 / 11 x64
- [Rust stable](https://rustup.rs/)
- 本机已安装 OpenCV，并且 `opencv` crate 能链接到它（常用 vcpkg 或官方 OpenCV Windows 包）
- SAM 功能需要两个 ONNX 文件与 exe 放在同一目录

没有 OpenCV 时可以编译失败；没有 ONNX 时程序仍能看图和处理，只是 SAM 无法初始化。

## 编译运行

```powershell
git clone https://github.com/sutony1/Image-processing-rust.git
cd Image-processing-rust

cargo build --release
```

可执行文件：

```
target\release\img_view_cv.exe
```

双击运行。调试编译用 `cargo build`，产物在 `target\debug\img_view_cv.exe`。

模型文件放在 exe 同目录，或项目根目录：

```
vit_t_encoder.onnx
vit_t_decoder.onnx
```

仓库里已包含这两份 MobileSAM ViT-T 模型。程序按顺序查找：exe 目录 → 环境变量 `SAM_MODEL_DIR` → 当前工作目录 → 源码根目录。

## 命令行（给脚本 / AI 操作）

查看器启动后在本机 `127.0.0.1:17332` 听命令（不出网）。`--cli` 靠这个端口控制**已经打开的窗口**。没有 MCP。

```powershell
.\target\debug\img_view_cv.exe --cli status
.\target\debug\img_view_cv.exe --cli open .\help\images\photo-color-card.jpg
.\target\debug\img_view_cv.exe --cli screenshot
.\target\debug\img_view_cv.exe --cli auto-perspective
.\target\debug\img_view_cv.exe --cli extract-colors
```

窗口没开时，`--cli` 会自动启动查看器。完整命令见 `--cli help` 和 [docs/色卡向导与CLI.md](docs/色卡向导与CLI.md)。

## SAM 用法

1. 打开一张彩色照片
2. 菜单 **SAM分割 → SAM交互分割抠像**（首次会加载模型并编码，标题栏提示已加载）
3. 按住左键拖出一个包住对象的方框，松开
4. 标题会显示「正在分割」，完成后对象上出现绿色遮罩
5. 需要抠图时再点 **抠图并显示**

方框可以拖到图像黑边外再松开，坐标会钳到画面边缘。失败会弹出对话框。再分割另一个对象时，重新点一次「SAM交互分割抠像」。

## 源码结构

```
src/
├── main.rs        窗口、菜单、鼠标交互、缩放平移、SAM 框选
├── ocv_bridge.rs  OpenCV Mat ↔ Win32 BGRA
├── cv_ops.rs      色卡检测、透视、Canny、阈值等
└── sam_core.rs    MobileSAM 编码 / 解码（ONNX Runtime）
```

主要依赖：`opencv`、`winapi`、`ort`、`ndarray`、`image`、`kamadak-exif`、`anyhow`。

## 便携目录

`img_view_cv_portable/` 里是 exe + 两个 ONNX + `config.yaml`，可整夹拷走。开发时以源码和 `cargo build` 为准。

## 已知范围

- 仅 Windows
- SAM 在 CPU 上分割大图可能要数秒，请等标题栏提示
- 色卡自动透视依赖画面里有比较完整的矩形色卡；不行就用手动四点

## 许可证

用于学习和研究。
