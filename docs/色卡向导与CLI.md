# 色卡向导与命令行

本文记录：**「透视矫正」菜单向导**，以及 **`--cli` 控制口**。带插图的用户说明在 `help/index.html`（程序里：**帮助 → 使用说明**）。

仓库：<https://github.com/sutony1/Image-processing-rust>  
例图：`.\help\images\photo-color-card.jpg`

MCP 已经去掉。AI 和控制脚本只用 `--cli`。

---

## 1. 色卡测色向导

测色是流水线，不是平铺工具：

1. 拍歪的照片：SAM 框选抠出色卡 → 自动透视拉正（失败改手动四点）→ 读 24 色  
2. 已经拍正、铺满画面：直接读 24 色  

菜单栏：`文件 | 透视矫正 | 处理 | SAM分割 | 拉直 | 选框工具 | 批量处理 | 帮助`

**透视矫正**菜单：

| 项 | 作用 |
|---|---|
| 手动四点矫正 | 将画作、文档等平面展平，依次点击左上、右上、右下、左下 |
| 色卡测色向导 | 初学者入口。按当前进度弹对话框往下走 |
| 1. 框出色卡 (SAM抠图) | 加载 SAM，拖框包住整张色卡，松开后自动抠图 |
| 2. 拉正色卡 (自动透视) | 抠完后才能点（否则灰色） |
| 手动四点拉正色卡 | 依次点左上、右上、右下、左下 |
| 3. 读取色块颜色 | 拉正后才能点 |
| 恢复到打开时的原图 | 回到刚打开的照片，色卡进度清零 |

「处理」里不再放透视和读色。

标题栏进度：`色卡：未框选` → `已抠图，待拉正` → `已拉正，待读色` → `已读取颜色`

向导问「这是拍歪的色卡照片吗？」：是 = 拖框抠图再拉正；否 = 直接读色。

---

## 2. CLI 和控制口（127.0.0.1）

`--cli` 是**另一个进程**，要给已经打开的窗口发命令，所以 GUI 会在本机听：

**`127.0.0.1:17332`**（可用环境变量 `IMG_VIEW_CV_PORT` 改）

这不是网页服务，也不出网。只有本机上的 `--cli` 往这个端口写一行 JSON。没有这个端口，`--cli open` / `--cli screenshot` 就没法碰到窗口。

```
img_view_cv.exe --cli open 某图.jpg
        │
        │  TCP 127.0.0.1:17332
        ▼
  正在运行的查看器窗口（真正改图、截图）
```

窗口没开时，`--cli` 会先启动查看器，再连端口。

协议：`src/ipc.rs`  
CLI：`src/cli.rs`  
执行：`handle_ipc`（`src/main.rs`）

### 命令

```powershell
.\target\debug\img_view_cv.exe --cli status
.\target\debug\img_view_cv.exe --cli open .\help\images\photo-color-card.jpg
.\target\debug\img_view_cv.exe --cli auto-perspective
.\target\debug\img_view_cv.exe --cli extract-colors
.\target\debug\img_view_cv.exe --cli sam-crop 280 160 1260 1960
.\target\debug\img_view_cv.exe --cli save .\out.png
.\target\debug\img_view_cv.exe --cli screenshot
.\target\debug\img_view_cv.exe --cli screenshot .
.\target\debug\img_view_cv.exe --cli help
```

`screenshot` 存两张 PNG：客户区画面 + 整个活动窗口（标题栏和菜单）。

向导里的「是/否」对话框仍是给人点的。脚本应直接：`open` →（可选 `sam-crop`）→ `auto-perspective` → `extract-colors`。

---

## 3. 其它相关改动

- 启动不再弹 DOS 窗口
- 去掉 WIA、去掉 MCP
- SAM 拖框捕获鼠标；色卡流程里松开后自动抠图
- 中文路径用 `imdecode`
- 帮助：**帮助 → 使用说明**

---

## 4. 使用入口

1. 从仓库根目录运行上述编译版命令。
2. AI 操作软件用 `--cli`，不要找 MCP
3. 自己测色：打开照片 → **透视矫正 → 色卡测色向导**
4. 本文：`docs/色卡向导与CLI.md`
