use std::ptr::null_mut;
use winapi::um::winuser::{*, EnableWindow, SetForegroundWindow, LoadCursorW, IDC_ARROW, IDC_HAND, COLOR_WINDOW, CS_HREDRAW, CS_VREDRAW, UnregisterClassW, EnableMenuItem};

use winapi::um::libloaderapi::GetModuleHandleW;
use winapi::um::wingdi::*;
use winapi::um::commdlg::*;
// 添加拖拽支持所需的导入
use winapi::um::shellapi::{DragAcceptFiles, DragQueryFileW, DragFinish};

use winapi::shared::windef::*;
use winapi::shared::minwindef::*;
use winapi::um::shellscalingapi::*;
use opencv::{core, imgcodecs, imgproc};
use opencv::prelude::*;
use anyhow::{Result, Context};

mod ocv_bridge;
mod cv_ops;
mod sam_core;
mod wia_scanner;

// 菜单项ID常量
const MENU_OPEN_FILE: u32 = 1000;
const MENU_TO_GRAY: u32 = 1001;
const MENU_GAUSSIAN: u32 = 1002;
const MENU_CANNY: u32 = 1003;
const MENU_ADAPTIVE_THRESHOLD: u32 = 1004;
const MENU_MORPH_CLOSE: u32 = 1005;
const MENU_DILATE: u32 = 1006;
const MENU_FIND_CONTOURS: u32 = 1007;
const MENU_PERSPECTIVE_TRANSFORM: u32 = 1008;
const MENU_MANUAL_PERSPECTIVE: u32 = 1031; // 新增：手动透视变换
const MENU_EXTRACT_COLOR_BLOCKS: u32 = 1009;
const MENU_TEMPLATE_MATCHING: u32 = 1010;
const MENU_RESTORE: u32 = 1011;
const MENU_SAVE_FILE: u32 = 1012;
// 添加SAM相关菜单项
const MENU_SAM_INTERACTIVE: u32 = 1013; // 合并的SAM交互分割抠像菜单
const MENU_SAM_EXTRACT_MASK: u32 = 1015; // 新增：抠图并显示
// 新增：选框工具和裁剪菜单
const MENU_SELECTION_TOOL: u32 = 1016; // 选框工具
const MENU_CROP_SELECTION: u32 = 1017; // 裁剪选框区域
const MENU_BLUR_SELECTION: u32 = 1030; // 涂抹选框区域
// 新增：调整图像大小菜单
const MENU_RESIZE_IMAGE: u32 = 1018; // 调整图像大小
// 新增：图像旋转菜单项
const MENU_ROTATE_180: u32 = 1019; // 180度旋转
const MENU_ROTATE_90_CW: u32 = 1020; // 顺时针90度旋转
const MENU_ROTATE_90_CCW: u32 = 1021; // 逆时针90度旋转
const MENU_ROTATE_ANGLE: u32 = 1022; // 任意角度旋转
// 新增：退出菜单项
const MENU_EXIT: u32 = 1023; // 退出程序
// 新增：关闭当前文件菜单项
const MENU_CLOSE_FILE: u32 = 1024; // 关闭当前文件
// 新增：拉直菜单项
const MENU_STRAIGHTEN: u32 = 1025; // 拉直功能（现在作为子菜单容器）
const MENU_STRAIGHTEN_DRAW_LINE: u32 = 1026; // 画直线
const MENU_STRAIGHTEN_HORIZONTAL: u32 = 1027; // 水平校正

// 新增：批量处理菜单项
const MENU_BATCH_PROCESS: u32 = 1028; // 批量处理菜单
const MENU_BATCH_RESIZE: u32 = 1029; // 批量缩图
// 新增：WIA图像采集菜单项
const MENU_WIA_IMPORT: u32 = 1032; // 导入(WIA支持)
// 新增：拍摄信息面板菜单项
const MENU_EXIF_PANEL: u32 = 1033; // 拍摄信息面板

// 滑块控件常量
const TBS_HORZ: u32 = 0x0000;
const TBS_AUTOTICKS: u32 = 0x0001;
const TBS_ENABLESELRANGE: u32 = 0x0020;

// MAKELONG 宏定义
macro_rules! MAKELONG {
    ($low:expr, $high:expr) => {
        (($low as u32) | (($high as u32) << 16)) as u32
    };
}
// Trackbar 消息常量 - 使用正确的 WinAPI 值
const TBM_SETRANGE: u32 = 0x0406; // WM_USER + 6
const TBM_SETPOS: u32 = 0x0405;   // WM_USER + 5  
const TBM_GETPOS: u32 = 0x0400;   // WM_USER + 0

// 编辑框通知常量
const EN_CHANGE: u16 = 0x0300;

// 鼠标滚轮相关常量
const WM_MOUSEWHEEL: u32 = 0x020A;
const WHEEL_DELTA: i16 = 120;

// 滚动条常量
const SB_LINEUP: u16 = 0;
const SB_LINEDOWN: u16 = 1;
const SB_PAGEUP: u16 = 2;
const SB_PAGEDOWN: u16 = 3;
const SB_THUMBPOSITION: u16 = 4;
const SB_THUMBTRACK: u16 = 5;
const SB_ENDSCROLL: u16 = 8;

// 对话框控件ID
const IDC_SAVE_BUTTON: i32 = 1005;
const IDC_CANCEL_BUTTON: i32 = 1006;
const IDC_QUALITY_TRACKBAR: i32 = 1007; // 质量滑块ID
const IDC_WIDTH_EDIT: i32 = 1008;  // 宽度输入框ID
const IDC_HEIGHT_EDIT: i32 = 1009; // 高度输入框ID

// 新增：批量缩图对话框控件ID
const IDC_INPUT_FOLDER_EDIT: i32 = 2001; // 输入文件夹路径输入框
const IDC_INPUT_FOLDER_BUTTON: i32 = 2002; // 输入文件夹选择按钮
const IDC_OUTPUT_FOLDER_EDIT: i32 = 2003; // 输出文件夹路径输入框
const IDC_OUTPUT_FOLDER_BUTTON: i32 = 2004; // 输出文件夹选择按钮
const IDC_LONG_EDGE_EDIT: i32 = 2005; // 长边限制输入框
const IDC_START_CONVERT_BUTTON: i32 = 2006; // 开始转换按钮
const IDC_BATCH_QUALITY_TRACKBAR: i32 = 2007; // 批量处理质量滑块
const IDC_BATCH_QUALITY_LABEL: i32 = 2008; // 批量处理质量标签
const IDC_BATCH_INTERPOLATION_COMBO: i32 = 2009; // 批量处理插值算法下拉框

// 插值算法常量
const INTER_NEAREST: i32 = 0;      // 最近邻插值
const INTER_LINEAR: i32 = 1;       // 双线性插值（默认）
const INTER_CUBIC: i32 = 2;        // 双三次插值
const INTER_AREA: i32 = 3;         // 区域插值
const INTER_LANCZOS4: i32 = 4;     // Lanczos插值

// 拖拽文件相关常量
const WM_DROPFILES: u32 = 0x0233;

// HDROP类型定义 - 使用正确的winapi类型
use winapi::um::shellapi::HDROP;

// 调整图像大小对话框结果
#[derive(Debug, Clone)]
struct ResizeResult {
    new_width: i32,
    new_height: i32,
    quality: i32,
    interpolation: i32, // 新增：插值算法
}

// 对话框状态管理
struct DialogState {
    hwnd: HWND,
    width_edit: HWND,
    height_edit: HWND,

    quality_trackbar: HWND,
    quality_label: HWND,
    new_info_label: HWND,
    interpolation_combo: HWND, // 新增：插值算法选择下拉框
    is_active: bool,
    parent_hwnd: HWND,
    original_aspect_ratio: f64, // 新增：原始宽高比
    aspect_ratio_locked: bool, // 新增：是否锁定宽高比
    dialog_result: Option<ResizeResult>, // 新增：存储对话框结果
    should_exit: bool, // 新增：标记是否应该退出对话框
}

// 全局变量存储图像数据
#[allow(static_mut_refs)]
static mut IMAGE_STATE: Option<ImageState> = None;

// 全局对话框状态
#[allow(static_mut_refs)]
static mut DIALOG_STATE: Option<DialogState> = None;

// 全局主窗口句柄
#[allow(static_mut_refs)]
static mut MAIN_HWND: Option<HWND> = None;

// EXIF面板相关全局变量
#[allow(static_mut_refs)]
static mut EXIF_PANEL_HWND: Option<HWND> = None; // EXIF面板窗口句柄
#[allow(static_mut_refs)]
static mut EXIF_PANEL_VISIBLE: bool = false; // EXIF面板是否可见

// EXIF数据结构
#[derive(Debug, Clone, Default)]
struct ExifData {
    camera_model: String,
    aperture: String,      // f/4.0
    shutter_speed: String, // 1/800
    iso: String,           // ISO-200
    focal_length: String,  // 110mm
    color_space: String,   // sRGB / Adobe RGB
}

struct ImageState {
    original_bgr: Option<core::Mat>,
    current_mat: Option<core::Mat>,
    display_pixels: Vec<u8>,
    current_width: i32,
    current_height: i32,
    current_file_path: Option<String>, // 新增：存储当前图像文件路径
    sam_processor: Option<sam_core::SAMProcessor>, // 新增
    selecting: bool,
    select_start: Option<(i32,i32)>,
    select_end: Option<(i32,i32)>,
    current_mask: Option<(Vec<u8>, i32, i32)>, // 新增：存储当前mask信息
    // 新增：选框工具相关状态
    selection_mode: bool, // 是否处于选框模式
    selection_start: Option<(i32,i32)>, // 选框起始点
    selection_end: Option<(i32,i32)>, // 选框结束点
    // 新增：缩放相关状态
    zoom_factor: f32, // 缩放比例，1.0为适应窗口大小
    zoom_offset_x: f32, // 缩放后的图像偏移量（窗口坐标系）
    zoom_offset_y: f32, // 缩放后的图像偏移量（窗口坐标系）
    // 新增：拉直工具相关状态
    straighten_mode: bool, // 是否处于拉直模式
    straighten_start: Option<(i32,i32)>, // 拉直线起始点
    straighten_end: Option<(i32,i32)>, // 拉直线结束点
    // 新增：手动透视变换相关状态
    manual_perspective_mode: bool, // 是否处于手动透视变换模式
    manual_perspective_points: Vec<(i32, i32)>, // 用户选择的四个角点
    // 新增：灰度图状态跟踪
    is_grayscale: bool, // 当前图像是否为灰度图或黑白图
    // 新增：平移相关状态
    pan_mode: bool, // 是否处于平移模式
    pan_start: Option<(i32, i32)>, // 平移起始点（窗口坐标）
    last_pan_offset: (f32, f32), // 上次平移的偏移量
    // 新增：悬浮导航箭头相关状态
    show_nav_arrows: bool, // 是否显示导航箭头
    mouse_in_left_arrow: bool, // 鼠标是否在左侧箭头区域
    mouse_in_right_arrow: bool, // 鼠标是否在右侧箭头区域
    image_list: Vec<String>, // 当前目录的图片文件列表
    current_image_index: usize, // 当前图片在列表中的索引
    // 新增：智能加载相关状态
    display_max_size: i32, // 显示用的最大尺寸限制（默认2048px）
    full_resolution_path: Option<String>, // 原图文件路径
    is_full_resolution_loaded: bool, // 当前original_bgr是否为完整分辨率
}

impl ImageState {
    fn new() -> Self {
        Self {
            original_bgr: None,
            current_mat: None,
            display_pixels: Vec::new(),
            current_width: 0,
            current_height: 0,
            current_file_path: None, // 新增：当前文件路径
            sam_processor: None, // 新增
            selecting: false,
            select_start: None,
            select_end: None,
            current_mask: None, // 新增
            selection_mode: false, // 新增：选框模式
            selection_start: None, // 新增：选框起始点
            selection_end: None, // 新增：选框结束点
            zoom_factor: 1.0, // 新增：缩放比例
            zoom_offset_x: 0.0, // 新增：图像偏移量
            zoom_offset_y: 0.0, // 新增：图像偏移量
            straighten_mode: false, // 新增：拉直模式
            straighten_start: None, // 新增：拉直线起始点
            straighten_end: None, // 新增：拉直线结束点
            manual_perspective_mode: false, // 新增：手动透视变换模式
            manual_perspective_points: Vec::new(), // 新增：用户选择的四个角点
            is_grayscale: false, // 新增：灰度图状态
            pan_mode: false, // 新增：平移模式
            pan_start: None, // 新增：平移起始点
            last_pan_offset: (0.0, 0.0), // 新增：上次平移偏移量
            show_nav_arrows: false, // 新增：导航箭头显示状态
            mouse_in_left_arrow: false, // 新增：鼠标在左箭头区域
            mouse_in_right_arrow: false, // 新增：鼠标在右箭头区域
            image_list: Vec::new(), // 新增：图片列表
            current_image_index: 0, // 新增：当前图片索引
            display_max_size: 2048, // 新增：显示最大尺寸
            full_resolution_path: None, // 新增：完整分辨率路径
            is_full_resolution_loaded: false, // 新增：是否已加载完整分辨率
        }
    }
    
    fn load_image(&mut self, path: &str) -> Result<()> {
        println!("加载图像: {}", path);
        
        // 检查文件是否存在
        if !std::path::Path::new(path).exists() {
            return Err(anyhow::anyhow!("文件不存在: {}", path));
        }
        
        // 使用image库快速获取图片尺寸（不解码全图）
        let (img_w, img_h) = match image::io::Reader::open(path) {
            Ok(reader) => {
                match reader.into_dimensions() {
                    Ok(dimensions) => dimensions,
                    Err(_) => {
                        // 如果获取尺寸失败，回退到备用方法
                        println!("无法获取图片尺寸，使用备用方法");
                        return self.load_image_fallback(path);
                    }
                }
            }
            Err(_) => {
                // 如果打开文件失败，回退到备用方法
                println!("无法打开图片文件，使用备用方法");
                return self.load_image_fallback(path);
            }
        };
        
        let max_dim = img_w.max(img_h);
        let is_large_image = max_dim > self.display_max_size as u32;
        
        if is_large_image {
            println!("🔍 检测到大图 ({}x{})，加载缩小版本以提升速度", img_w, img_h);
        }
        
        // 尝试使用OpenCV读取图像为BGR格式
        let mut bgr = match imgcodecs::imread(path, imgcodecs::IMREAD_COLOR) {
            Ok(mat) => {
                if mat.empty() {
                    // 如果OpenCV返回空图像，尝试备用方法
                    println!("OpenCV返回空图像，尝试备用方法");
                    return self.load_image_fallback(path);
                }
                mat
            }
            Err(e) => {
                // 如果OpenCV读取失败，尝试使用标准库读取然后转换
                println!("OpenCV直接读取失败: {}, 尝试备用方法", e);
                return self.load_image_fallback(path);
            }
        };
        
        // 如果是大图，缩小到display_max_size
        if is_large_image {
            let scale = self.display_max_size as f32 / max_dim as f32;
            let new_w = (img_w as f32 * scale) as i32;
            let new_h = (img_h as f32 * scale) as i32;
            
            let mut resized = core::Mat::default();
            imgproc::resize(&bgr, &mut resized, 
                core::Size::new(new_w, new_h), 
                0.0, 0.0, 
                imgproc::INTER_AREA)
                .context("缩放图像失败")?;
            
            bgr = resized;
            println!("✅ 已缩放至: {}x{}", new_w, new_h);
        }
        
        println!("图像尺寸: {}x{}, 通道数: {}", bgr.cols(), bgr.rows(), bgr.channels());
        
        // 转换为显示像素
        let image_buf = ocv_bridge::mat_to_bgra(&bgr)?;
        
        let bgr_clone = bgr.try_clone()?;
        self.original_bgr = Some(bgr);
        self.current_mat = Some(bgr_clone);
        self.display_pixels = image_buf.pixels;
        self.current_width = image_buf.w;
        self.current_height = image_buf.h;
        self.current_file_path = Some(path.to_string()); // 存储文件路径
        
        // 更新智能加载状态
        self.full_resolution_path = Some(path.to_string());
        self.is_full_resolution_loaded = !is_large_image;
        
        // 重置缩放状态
        self.reset_zoom();
        
        // 重置选框工具状态
        self.reset_selection_tools();
        
        // 更新图片列表
        self.update_image_list();
        
        if is_large_image {
            println!("⚡ 图像快速加载完成（使用缩小版本）");
        } else {
            println!("✅ 图像加载完成");
        }
        
        // 更新EXIF面板（如果打开）
        unsafe { update_exif_panel_if_visible(); }
        
        Ok(())
    }
    
    fn load_image_fallback(&mut self, path: &str) -> Result<()> {
        println!("使用备用方法加载图像...");
        
        // 使用标准库读取文件到内存
        let image_data = std::fs::read(path)
            .context("无法读取图像文件到内存")?;
        
        // 使用OpenCV从内存数据解码图像
        let data_vec = opencv::core::Vector::from_slice(&image_data);
        let mut bgr = imgcodecs::imdecode(&data_vec, imgcodecs::IMREAD_COLOR)
            .context("无法解码图像数据")?;
        
        if bgr.empty() {
            return Err(anyhow::anyhow!("解码后的图像为空，可能是不支持的格式"));
        }
        
        let img_w = bgr.cols() as u32;
        let img_h = bgr.rows() as u32;
        let max_dim = img_w.max(img_h);
        let is_large_image = max_dim > self.display_max_size as u32;
        
        if is_large_image {
            println!("🔍 检测到大图 ({}x{})，缩小以提升速度", img_w, img_h);
            let scale = self.display_max_size as f32 / max_dim as f32;
            let new_w = (img_w as f32 * scale) as i32;
            let new_h = (img_h as f32 * scale) as i32;
            
            let mut resized = core::Mat::default();
            imgproc::resize(&bgr, &mut resized, 
                core::Size::new(new_w, new_h), 
                0.0, 0.0, 
                imgproc::INTER_AREA)
                .context("缩放图像失败")?;
            
            bgr = resized;
            println!("✅ 已缩放至: {}x{}", new_w, new_h);
        }
        
        println!("图像尺寸: {}x{}, 通道数: {}", bgr.cols(), bgr.rows(), bgr.channels());
        
        // 转换为显示像素
        let image_buf = ocv_bridge::mat_to_bgra(&bgr)?;
        
        let bgr_clone = bgr.try_clone()?;
        self.original_bgr = Some(bgr);
        self.current_mat = Some(bgr_clone);
        self.display_pixels = image_buf.pixels;
        self.current_width = image_buf.w;
        self.current_height = image_buf.h;
        self.current_file_path = Some(path.to_string()); // 存储文件路径
        
        // 更新智能加载状态
        self.full_resolution_path = Some(path.to_string());
        self.is_full_resolution_loaded = !is_large_image;
        
        // 重置缩放状态
        self.reset_zoom();
        
        // 重置选框工具状态
        self.reset_selection_tools();
        
        // 更新图片列表
        self.update_image_list();
        
        if is_large_image {
            println!("⚡ 图像快速加载完成（备用方法，使用缩小版本）");
        } else {
            println!("✅ 图像加载成功（备用方法）");
        }
        
        // 更新EXIF面板（如果打开）
        unsafe { update_exif_panel_if_visible(); }
        
        Ok(())
    }
    
    fn update_display_from_mat(&mut self, mat: &core::Mat) -> Result<()> {
        let image_buf = ocv_bridge::mat_to_bgra(mat)?;
        self.current_mat = Some(mat.try_clone()?);
        self.display_pixels = image_buf.pixels;
        self.current_width = image_buf.w;
        self.current_height = image_buf.h;
        Ok(())
    }
    
    fn has_image(&self) -> bool {
        self.original_bgr.is_some()
    }

    // 新增：按需加载完整分辨率图像
    fn ensure_full_resolution(&mut self) -> Result<()> {
        // 如果已经是完整分辨率，直接返回
        if self.is_full_resolution_loaded {
            return Ok(());
        }
        
        // 获取原图路径
        let path = self.full_resolution_path.as_ref()
            .ok_or_else(|| anyhow::anyhow!("无文件路径"))?
            .clone();
        
        println!("⏳ 加载完整分辨率图像: {}", path);
        
        // 加载完整分辨率图像
        let full_bgr = imgcodecs::imread(&path, imgcodecs::IMREAD_COLOR)
            .context("无法读取完整分辨率图像")?;
        
        if full_bgr.empty() {
            return Err(anyhow::anyhow!("加载完整分辨率失败：图像为空"));
        }
        
        println!("✅ 完整分辨率加载完成: {}x{}", full_bgr.cols(), full_bgr.rows());
        
        // 更新状态
        self.original_bgr = Some(full_bgr.try_clone()?);
        self.current_mat = Some(full_bgr);
        self.is_full_resolution_loaded = true;
        
        Ok(())
    }

    // 新增：更新当前目录的图片列表
    fn update_image_list(&mut self) {
        if let Some(ref current_path) = self.current_file_path {
            self.image_list = Self::get_images_in_directory(current_path);
            // 找到当前图片在列表中的索引
            if let Some(index) = self.image_list.iter().position(|p| p == current_path) {
                self.current_image_index = index;
            }
            println!("📁 当前目录共有 {} 张图片，当前是第 {} 张", self.image_list.len(), self.current_image_index + 1);
        }
    }

    // 新增：获取指定图片所在目录的所有图片文件
    fn get_images_in_directory(current_file: &str) -> Vec<String> {
        use std::path::Path;
        use std::fs;
        
        let path = Path::new(current_file);
        let dir = match path.parent() {
            Some(d) => d,
            None => return Vec::new(),
        };
        
        let mut images = Vec::new();
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                if let Ok(path) = entry.path().canonicalize() {
                    if let Some(ext) = path.extension() {
                        let ext_lower = ext.to_string_lossy().to_lowercase();
                        if matches!(ext_lower.as_str(), "jpg" | "jpeg" | "png" | "bmp" | "gif" | "webp" | "tiff" | "tif") {
                            if let Some(path_str) = path.to_str() {
                                images.push(path_str.to_string());
                            }
                        }
                    }
                }
            }
        }
        
        // 按文件名排序
        images.sort();
        images
    }

    // 新增：加载下一张图片
    fn load_next_image(&mut self) -> Result<()> {
        if self.image_list.is_empty() {
            return Err(anyhow::anyhow!("图片列表为空"));
        }
        
        if self.current_image_index < self.image_list.len() - 1 {
            self.current_image_index += 1;
            let next_path = self.image_list[self.current_image_index].clone();
            self.load_image(&next_path)?;
            println!("➡️ 加载下一张图片 ({}/{})", self.current_image_index + 1, self.image_list.len());
        } else {
            println!("⚠️ 已经是最后一张图片了");
        }
        Ok(())
    }

    // 新增：加载上一张图片
    fn load_previous_image(&mut self) -> Result<()> {
        if self.image_list.is_empty() {
            return Err(anyhow::anyhow!("图片列表为空"));
        }
        
        if self.current_image_index > 0 {
            self.current_image_index -= 1;
            let prev_path = self.image_list[self.current_image_index].clone();
            self.load_image(&prev_path)?;
            println!("⬅️ 加载上一张图片 ({}/{})", self.current_image_index + 1, self.image_list.len());
        } else {
            println!("⚠️ 已经是第一张图片了");
        }
        Ok(())
    }

    // 新增：检查是否可以导航到上一张
    fn can_go_previous(&self) -> bool {
        !self.image_list.is_empty() && self.current_image_index > 0
    }

    // 新增：检查是否可以导航到下一张
    fn can_go_next(&self) -> bool {
        !self.image_list.is_empty() && self.current_image_index < self.image_list.len() - 1
    }

    // 新增：更新鼠标位置并检测是否在箭头区域
    fn update_nav_arrows(&mut self, mx: i32, my: i32, window_width: i32, window_height: i32) {
        // 箭头区域：窗口左侧和右侧各100像素宽的区域
        const ARROW_AREA_WIDTH: i32 = 100;
        const ARROW_AREA_HEIGHT: i32 = 80; // 箭头高度
        
        // 左侧箭头区域：垂直居中
        let left_arrow_top = (window_height - ARROW_AREA_HEIGHT) / 2;
        let left_arrow_bottom = left_arrow_top + ARROW_AREA_HEIGHT;
        self.mouse_in_left_arrow = mx < ARROW_AREA_WIDTH 
            && my >= left_arrow_top 
            && my <= left_arrow_bottom
            && self.can_go_previous();
        
        // 右侧箭头区域：垂直居中
        let right_arrow_left = window_width - ARROW_AREA_WIDTH;
        let right_arrow_top = (window_height - ARROW_AREA_HEIGHT) / 2;
        let right_arrow_bottom = right_arrow_top + ARROW_AREA_HEIGHT;
        self.mouse_in_right_arrow = mx >= right_arrow_left 
            && my >= right_arrow_top 
            && my <= right_arrow_bottom
            && self.can_go_next();
        
        // 如果鼠标在箭头区域或者图片列表有多张图，显示箭头
        self.show_nav_arrows = (self.mouse_in_left_arrow || self.mouse_in_right_arrow || 
                                mx < ARROW_AREA_WIDTH || mx >= right_arrow_left) 
                                && self.image_list.len() > 1;
    }

    fn init_sam_processor(&mut self) -> Result<()> {
        if self.sam_processor.is_none() {
            self.sam_processor = Some(sam_core::SAMProcessor::new()?);
            println!("✅ SAM处理器初始化成功");
        }
        Ok(())
    }

    // 新增：设置当前mask
    fn set_current_mask(&mut self, mask: Vec<u8>, mask_w: i32, mask_h: i32) {
        self.current_mask = Some((mask, mask_w, mask_h));
    }

    // 新增：检查是否有可用的mask
    fn has_mask(&self) -> bool {
        self.current_mask.is_some()
    }

    // 新增：获取当前mask
    fn get_current_mask(&self) -> Option<&(Vec<u8>, i32, i32)> {
        self.current_mask.as_ref()
    }

    // 新增：选框工具相关方法
    fn start_selection_mode(&mut self) {
        self.selection_mode = true;
        self.selection_start = None;
        self.selection_end = None;
        println!("🖱️ 选框工具已激活：请在图像上按下并拖拽以框选区域，释放鼠标完成框选");
    }

    fn end_selection_mode(&mut self) {
        self.selection_mode = false;
    }

    fn has_selection(&self) -> bool {
        self.selection_start.is_some() && self.selection_end.is_some()
    }

    fn get_selection_rect(&self) -> Option<(i32, i32, i32, i32)> {
        if let (Some((x1, y1)), Some((x2, y2))) = (self.selection_start, self.selection_end) {
            let left = x1.min(x2);
            let top = y1.min(y2);
            let right = x1.max(x2);
            let bottom = y1.max(y2);
            Some((left, top, right, bottom))
        } else {
            None
        }
    }

    // 缩放相关方法
    fn reset_zoom(&mut self) {
        self.zoom_factor = 1.0;
        self.zoom_offset_x = 0.0;
        self.zoom_offset_y = 0.0;
    }

    // 重置选框工具状态
    fn reset_selection_tools(&mut self) {
        self.selection_mode = false;
        self.selection_start = None;
        self.selection_end = None;
        self.selecting = false;
        self.select_start = None;
        self.select_end = None;
        self.current_mask = None;
        self.is_grayscale = false; // 重置灰度图状态
        self.pan_mode = false; // 重置平移状态
        self.pan_start = None;
        self.last_pan_offset = (0.0, 0.0);
        println!("🔄 选框工具状态已重置");
    }

    // 更新SAM菜单的可用性
    fn update_sam_menu_state(&self, hwnd: HWND) {
        unsafe {
            if self.is_grayscale {
                // 如果是灰度图，禁用SAM菜单项
                EnableMenuItem(GetMenu(hwnd), MENU_SAM_INTERACTIVE as u32, MF_BYCOMMAND | MF_GRAYED);
                EnableMenuItem(GetMenu(hwnd), MENU_SAM_EXTRACT_MASK as u32, MF_BYCOMMAND | MF_GRAYED);
                println!("⚠️ SAM菜单已禁用 - 当前图像为灰度图/黑白图");
            } else {
                // 如果是彩色图，启用SAM菜单项
                EnableMenuItem(GetMenu(hwnd), MENU_SAM_INTERACTIVE as u32, MF_BYCOMMAND | MF_ENABLED);
                EnableMenuItem(GetMenu(hwnd), MENU_SAM_EXTRACT_MASK as u32, MF_BYCOMMAND | MF_ENABLED);
                println!("✅ SAM菜单已启用 - 当前图像为彩色图");
            }
        }
    }

    // 判断是否可以平移图像
    fn can_pan(&self) -> bool {
        if !self.has_image() {
            return false;
        }
        
        // 只要有图像且不在其他工具模式下，就可以平移
        // 这样用户可以在任何缩放状态下自由移动图像位置
        true
    }

    // 开始平移模式
    fn start_pan(&mut self, start_x: i32, start_y: i32) {
        self.pan_mode = true;
        self.pan_start = Some((start_x, start_y));
        self.last_pan_offset = (self.zoom_offset_x, self.zoom_offset_y);
        println!("🖐️ 开始平移模式");
    }

    // 更新平移偏移
    fn update_pan(&mut self, current_x: i32, current_y: i32) {
        if let Some((start_x, start_y)) = self.pan_start {
            let delta_x = current_x - start_x;
            let delta_y = current_y - start_y;
            
            self.zoom_offset_x = self.last_pan_offset.0 + delta_x as f32;
            self.zoom_offset_y = self.last_pan_offset.1 + delta_y as f32;
        }
    }

    // 结束平移模式
    fn end_pan(&mut self) {
        self.pan_mode = false;
        self.pan_start = None;
        self.last_pan_offset = (self.zoom_offset_x, self.zoom_offset_y);
        println!("🖐️ 结束平移模式");
    }

    // 设置手型光标
    fn set_hand_cursor(&self, _hwnd: HWND) {
        unsafe {
            let hand_cursor = LoadCursorW(null_mut(), IDC_HAND);
            SetCursor(hand_cursor);
        }
    }

    // 设置箭头光标
    fn set_arrow_cursor(&self, _hwnd: HWND) {
        unsafe {
            let arrow_cursor = LoadCursorW(null_mut(), IDC_ARROW);
            SetCursor(arrow_cursor);
        }
    }

    // 关闭当前文件，重置到初始状态
    fn close_current_file(&mut self) {
        self.original_bgr = None;
        self.current_mat = None;
        self.display_pixels.clear();
        self.current_width = 0;
        self.current_height = 0;
        self.current_file_path = None;
        self.sam_processor = None;
        self.selecting = false;
        self.select_start = None;
        self.select_end = None;
        self.current_mask = None;
        self.selection_mode = false;
        self.selection_start = None;
        self.selection_end = None;
        self.straighten_mode = false;
        self.straighten_start = None;
        self.straighten_end = None;
        self.is_grayscale = false; // 重置灰度图状态
        self.pan_mode = false; // 重置平移状态
        self.pan_start = None;
        // 重置智能加载状态
        self.full_resolution_path = None;
        self.is_full_resolution_loaded = false;
        self.last_pan_offset = (0.0, 0.0);
        self.reset_zoom();
        println!("✅ 当前文件已关闭，状态已重置");
    }

    fn apply_zoom(&mut self, delta: f32, mouse_x: f32, mouse_y: f32, window_width: i32, window_height: i32) {
        if !self.has_image() {
            return;
        }

        let old_zoom = self.zoom_factor;
        
        // 计算新的缩放比例，限制范围在0.1x到10x之间
        if delta > 0.0 {
            self.zoom_factor *= 1.1; // 放大（更平缓）
        } else {
            self.zoom_factor /= 1.1; // 缩小（更平缓）
        }
        self.zoom_factor = self.zoom_factor.clamp(0.1, 10.0);

        // 如果缩放比例没有变化，直接返回
        if (self.zoom_factor - old_zoom).abs() < 0.001 {
            return;
        }

        // 获取缩放前鼠标在图像中的位置
        let (old_img_x, old_img_y, old_display_w, old_display_h) = calculate_scaled_size_with_zoom_factor(
            self, old_zoom, window_width, window_height
        );
        
        // 鼠标在缩放前图像中的相对位置（0.0-1.0）
        let mouse_rel_x = if old_display_w > 0 {
            (mouse_x - old_img_x as f32) / old_display_w as f32
        } else {
            0.5
        };
        let mouse_rel_y = if old_display_h > 0 {
            (mouse_y - old_img_y as f32) / old_display_h as f32
        } else {
            0.5
        };
        
        // 获取缩放后的图像显示尺寸
        let (new_img_x, new_img_y, new_display_w, new_display_h) = calculate_scaled_size_with_zoom_factor(
            self, self.zoom_factor, window_width, window_height
        );
        
        // 计算鼠标在缩放后图像中应该的位置
        let new_mouse_img_x = new_img_x as f32 + mouse_rel_x * new_display_w as f32;
        let new_mouse_img_y = new_img_y as f32 + mouse_rel_y * new_display_h as f32;
        
        // 调整偏移，使鼠标指向的图像内容保持在相同的窗口位置
        self.zoom_offset_x += mouse_x - new_mouse_img_x;
        self.zoom_offset_y += mouse_y - new_mouse_img_y;
    }

    fn calculate_base_display_rect(&self, window_width: i32, window_height: i32) -> (i32, i32, i32, i32) {
        if !self.has_image() {
            return (0, 0, 0, 0);
        }
        
        let img_ratio = self.current_width as f64 / self.current_height as f64;
        let window_ratio = window_width as f64 / window_height as f64;
        
        let (base_width, base_height) = if img_ratio > window_ratio {
            // 图片更宽，以窗口宽度为准
            let base_width = window_width;
            let base_height = (window_width as f64 / img_ratio) as i32;
            (base_width, base_height)
        } else {
            // 图片更高，以窗口高度为准
            let base_height = window_height;
            let base_width = (window_height as f64 * img_ratio) as i32;
            (base_width, base_height)
        };
        
        // 居中显示
        let x = (window_width - base_width) / 2;
        let y = (window_height - base_height) / 2;
        
        (x, y, base_width, base_height)
    }
}

// 使用OpenCV内存编码 + Rust文件写入，确保Windows下Unicode路径不乱码
fn save_mat_with_unicode_path(mat: &core::Mat, path: &str) -> Result<String> {
    // 解析并规范扩展名
    let mut path_buf = std::path::PathBuf::from(path);
    let ext_opt = path_buf.extension().and_then(|s| s.to_str()).map(|s| s.to_ascii_lowercase());

    // 选择编码扩展（imencode需要以'.'开头），以及最终文件扩展（不带点）
    let (encode_ext, file_ext): (&str, &str) = match ext_opt.as_deref() {
        Some("png") => (".png", "png"),
        Some("jpg") | Some("jpeg") => (".jpg", "jpg"),
        Some("bmp") => (".bmp", "bmp"),
        Some("tif") | Some("tiff") => (".tiff", "tiff"),
        _ => (".png", "png"),
    };

    // 若没有扩展或不支持，则补上默认扩展
    if ext_opt.is_none() || !matches!(ext_opt.as_deref(), Some("png" | "jpg" | "jpeg" | "bmp" | "tif" | "tiff")) {
        path_buf.set_extension(file_ext);
    }

    // 使用OpenCV编码到内存
    let mut buf = core::Vector::<u8>::new();
    imgcodecs::imencode(encode_ext, mat, &mut buf, &core::Vector::new())
        .context("OpenCV 内存编码失败")?;

    // 通过Rust文件API写入，确保Unicode路径正确
    std::fs::write(&path_buf, buf.to_vec())
        .with_context(|| format!("写入文件失败: {}", path_buf.display()))?;

    Ok(path_buf.to_string_lossy().to_string())
}

// 创建菜单
unsafe fn create_menu() -> HMENU {
    let menu_bar = CreateMenu();
    
    // 创建文件菜单
    let file_menu = CreatePopupMenu();
    let open_file_text = "打开文件(&O)...";
    let open_file_utf16: Vec<u16> = open_file_text.encode_utf16().chain(std::iter::once(0)).collect();
    AppendMenuW(file_menu, MF_STRING, MENU_OPEN_FILE as usize, open_file_utf16.as_ptr());
    
    // 添加WIA导入菜单项
    let wia_import_text = "导入(WIA支持)(&W)...";
    let wia_import_utf16: Vec<u16> = wia_import_text.encode_utf16().chain(std::iter::once(0)).collect();
    AppendMenuW(file_menu, MF_STRING, MENU_WIA_IMPORT as usize, wia_import_utf16.as_ptr());
    
    AppendMenuW(file_menu, MF_SEPARATOR, 0, null_mut());
    
    let save_file_text = "保存文件(&S)...";
    let save_file_utf16: Vec<u16> = save_file_text.encode_utf16().chain(std::iter::once(0)).collect();
    AppendMenuW(file_menu, MF_STRING, MENU_SAVE_FILE as usize, save_file_utf16.as_ptr());
    
    // 添加关闭当前文件菜单项
    let close_file_text = "关闭当前文件(&C)";
    let close_file_utf16: Vec<u16> = close_file_text.encode_utf16().chain(std::iter::once(0)).collect();
    AppendMenuW(file_menu, MF_STRING, MENU_CLOSE_FILE as usize, close_file_utf16.as_ptr());
    
    // 添加分隔符
    AppendMenuW(file_menu, MF_SEPARATOR, 0, null_mut());
    
    // 添加退出菜单项
    let exit_text = "退出(&X)";
    let exit_utf16: Vec<u16> = exit_text.encode_utf16().chain(std::iter::once(0)).collect();
    AppendMenuW(file_menu, MF_STRING, MENU_EXIT as usize, exit_utf16.as_ptr());
    
    // 创建处理菜单
    let process_menu = CreatePopupMenu();
    
    // 恢复原图
    let restore_text = "恢复原图(&R)";
    let restore_utf16: Vec<u16> = restore_text.encode_utf16().chain(std::iter::once(0)).collect();
    AppendMenuW(process_menu, MF_STRING, MENU_RESTORE as usize, restore_utf16.as_ptr());

    AppendMenuW(process_menu, MF_SEPARATOR, 0, null_mut());

    // 基础处理
    let to_gray_text = "转换为灰度图(&G)";
    let to_gray_utf16: Vec<u16> = to_gray_text.encode_utf16().chain(std::iter::once(0)).collect();
    AppendMenuW(process_menu, MF_STRING, MENU_TO_GRAY as usize, to_gray_utf16.as_ptr());
    
    // 暂时隐藏高斯模糊菜单项
    // let gaussian_text = "高斯模糊(&B)";
    // let gaussian_utf16: Vec<u16> = gaussian_text.encode_utf16().chain(std::iter::once(0)).collect();
    // AppendMenuW(process_menu, MF_STRING, MENU_GAUSSIAN as usize, gaussian_utf16.as_ptr());
    
    AppendMenuW(process_menu, MF_SEPARATOR, 0, null_mut());
    
    // 边缘检测
    let canny_text = "Canny边缘检测(&C)";
    let canny_utf16: Vec<u16> = canny_text.encode_utf16().chain(std::iter::once(0)).collect();
    AppendMenuW(process_menu, MF_STRING, MENU_CANNY as usize, canny_utf16.as_ptr());
    
    let threshold_text = "自适应阈值(&T)";
    let threshold_utf16: Vec<u16> = threshold_text.encode_utf16().chain(std::iter::once(0)).collect();
    AppendMenuW(process_menu, MF_STRING, MENU_ADAPTIVE_THRESHOLD as usize, threshold_utf16.as_ptr());
    
    AppendMenuW(process_menu, MF_SEPARATOR, 0, null_mut());
    
    // 形态学操作 - 暂时隐藏
    // let morph_text = "形态学闭运算(&M)";
    // let morph_utf16: Vec<u16> = morph_text.encode_utf16().chain(std::iter::once(0)).collect();
    // AppendMenuW(process_menu, MF_STRING, MENU_MORPH_CLOSE as usize, morph_utf16.as_ptr());
    
    // let dilate_text = "膨胀操作(&D)";
    // let dilate_utf16: Vec<u16> = dilate_text.encode_utf16().chain(std::iter::once(0)).collect();
    // AppendMenuW(process_menu, MF_STRING, MENU_DILATE as usize, dilate_utf16.as_ptr());
    
    AppendMenuW(process_menu, MF_SEPARATOR, 0, null_mut());
    
    // 轮廓检测 - 暂时隐藏
    // let contours_text = "轮廓检测(&F)";
    // let contours_utf16: Vec<u16> = contours_text.encode_utf16().chain(std::iter::once(0)).collect();
    // AppendMenuW(process_menu, MF_STRING, MENU_FIND_CONTOURS as usize, contours_utf16.as_ptr());
    
            // 透视变换
        let perspective_text = "自动透视变换(&P)";
        let perspective_utf16: Vec<u16> = perspective_text.encode_utf16().chain(std::iter::once(0)).collect();
        AppendMenuW(process_menu, MF_STRING, MENU_PERSPECTIVE_TRANSFORM as usize, perspective_utf16.as_ptr());
    
    // 手动透视变换
    let manual_perspective_text = "手动透视变换(&M)";
    let manual_perspective_utf16: Vec<u16> = manual_perspective_text.encode_utf16().chain(std::iter::once(0)).collect();
    AppendMenuW(process_menu, MF_STRING, MENU_MANUAL_PERSPECTIVE as usize, manual_perspective_utf16.as_ptr());
    
    AppendMenuW(process_menu, MF_SEPARATOR, 0, null_mut());
    
    // 色卡分析 - 暂时隐藏模板匹配
    // let template_match_text = "模板匹配定位色卡(&T)";
    // let template_match_utf16: Vec<u16> = template_match_text.encode_utf16().chain(std::iter::once(0)).collect();
    // AppendMenuW(process_menu, MF_STRING, MENU_TEMPLATE_MATCHING as usize, template_match_utf16.as_ptr());
    
    let extract_color_text = "读取色卡色块颜色(&E)";
    let extract_color_utf16: Vec<u16> = extract_color_text.encode_utf16().chain(std::iter::once(0)).collect();
    AppendMenuW(process_menu, MF_STRING, MENU_EXTRACT_COLOR_BLOCKS as usize, extract_color_utf16.as_ptr());
    
    AppendMenuW(process_menu, MF_SEPARATOR, 0, null_mut());
    
    // 拍摄信息面板
    let exif_panel_text = "拍摄信息面板(&X)";
    let exif_panel_utf16: Vec<u16> = exif_panel_text.encode_utf16().chain(std::iter::once(0)).collect();
    AppendMenuW(process_menu, MF_STRING, MENU_EXIF_PANEL as usize, exif_panel_utf16.as_ptr());
    
    AppendMenuW(process_menu, MF_SEPARATOR, 0, null_mut());
    
    
    // 调整图像大小
    let resize_text = "调整图像大小(&I)";
    let resize_utf16: Vec<u16> = resize_text.encode_utf16().chain(std::iter::once(0)).collect();
    AppendMenuW(process_menu, MF_STRING, MENU_RESIZE_IMAGE as usize, resize_utf16.as_ptr());
    
    AppendMenuW(process_menu, MF_SEPARATOR, 0, null_mut());
    
    // 图像旋转子菜单
    let rotation_menu = CreatePopupMenu();
    
    // 180度旋转
    let rotate_180_text = "180度(&1)";
    let rotate_180_utf16: Vec<u16> = rotate_180_text.encode_utf16().chain(std::iter::once(0)).collect();
    AppendMenuW(rotation_menu, MF_STRING, MENU_ROTATE_180 as usize, rotate_180_utf16.as_ptr());
    
    // 顺时针90度
    let rotate_90_cw_text = "顺时针90度(&A)";
    let rotate_90_cw_utf16: Vec<u16> = rotate_90_cw_text.encode_utf16().chain(std::iter::once(0)).collect();
    AppendMenuW(rotation_menu, MF_STRING, MENU_ROTATE_90_CW as usize, rotate_90_cw_utf16.as_ptr());
    
    // 逆时针90度
    let rotate_90_ccw_text = "逆时针90度(&O)";
    let rotate_90_ccw_utf16: Vec<u16> = rotate_90_ccw_text.encode_utf16().chain(std::iter::once(0)).collect();
    AppendMenuW(rotation_menu, MF_STRING, MENU_ROTATE_90_CCW as usize, rotate_90_ccw_utf16.as_ptr());
    
    // 任意角度
    let rotate_angle_text = "任意角度(&A)";
    let rotate_angle_utf16: Vec<u16> = rotate_angle_text.encode_utf16().chain(std::iter::once(0)).collect();
    AppendMenuW(rotation_menu, MF_STRING, MENU_ROTATE_ANGLE as usize, rotate_angle_utf16.as_ptr());
    
    // 将旋转子菜单添加到处理菜单
    let rotate_text = "图像旋转(&G)";
    let rotate_utf16: Vec<u16> = rotate_text.encode_utf16().chain(std::iter::once(0)).collect();
    AppendMenuW(process_menu, MF_POPUP, rotation_menu as usize, rotate_utf16.as_ptr());
    
    // 添加菜单到菜单栏
    let file_text = "文件(&F)";
    let file_utf16: Vec<u16> = file_text.encode_utf16().chain(std::iter::once(0)).collect();
    AppendMenuW(menu_bar, MF_POPUP, file_menu as usize, file_utf16.as_ptr());
    
    let process_text = "处理(&P)";
    let process_utf16: Vec<u16> = process_text.encode_utf16().chain(std::iter::once(0)).collect();
    AppendMenuW(menu_bar, MF_POPUP, process_menu as usize, process_utf16.as_ptr());
    
    // 添加SAM菜单
    let sam_menu = CreatePopupMenu();
    let sam_interactive_text = "SAM交互分割抠像(&I)";
    let sam_interactive_utf16: Vec<u16> = sam_interactive_text.encode_utf16().chain(std::iter::once(0)).collect();
    AppendMenuW(sam_menu, MF_STRING, MENU_SAM_INTERACTIVE as usize, sam_interactive_utf16.as_ptr());
    
    // 新增：抠图并显示命令
    let sam_extract_text = "抠图并显示(&E)";
    let sam_extract_utf16: Vec<u16> = sam_extract_text.encode_utf16().chain(std::iter::once(0)).collect();
    AppendMenuW(sam_menu, MF_STRING, MENU_SAM_EXTRACT_MASK as usize, sam_extract_utf16.as_ptr());
    
    let sam_text = "SAM分割(&S)";
    let sam_utf16: Vec<u16> = sam_text.encode_utf16().chain(std::iter::once(0)).collect();
    AppendMenuW(menu_bar, MF_POPUP, sam_menu as usize, sam_utf16.as_ptr());
    
    // 新增：拉直子菜单
    let straighten_menu = CreatePopupMenu();
    let draw_line_text = "画直线(&D)";
    let draw_line_utf16: Vec<u16> = draw_line_text.encode_utf16().chain(std::iter::once(0)).collect();
    AppendMenuW(straighten_menu, MF_STRING, MENU_STRAIGHTEN_DRAW_LINE as usize, draw_line_utf16.as_ptr());
    
    let horizontal_text = "水平校正(&H)";
    let horizontal_utf16: Vec<u16> = horizontal_text.encode_utf16().chain(std::iter::once(0)).collect();
    AppendMenuW(straighten_menu, MF_STRING, MENU_STRAIGHTEN_HORIZONTAL as usize, horizontal_utf16.as_ptr());
    
    let straighten_text = "拉直(&T)";
    let straighten_utf16: Vec<u16> = straighten_text.encode_utf16().chain(std::iter::once(0)).collect();
    AppendMenuW(menu_bar, MF_POPUP, straighten_menu as usize, straighten_utf16.as_ptr());
    
    // 新增：选框工具和裁剪菜单
    let selection_menu = CreatePopupMenu();
    let selection_tool_text = "选框工具(&S)";
    let selection_tool_utf16: Vec<u16> = selection_tool_text.encode_utf16().chain(std::iter::once(0)).collect();
    AppendMenuW(selection_menu, MF_STRING, MENU_SELECTION_TOOL as usize, selection_tool_utf16.as_ptr());
    
    let crop_selection_text = "裁剪选框区域(&C)";
    let crop_selection_utf16: Vec<u16> = crop_selection_text.encode_utf16().chain(std::iter::once(0)).collect();
    AppendMenuW(selection_menu, MF_STRING, MENU_CROP_SELECTION as usize, crop_selection_utf16.as_ptr());
    
    let blur_selection_text = "涂抹选框区域(&B)";
    let blur_selection_text_utf16: Vec<u16> = blur_selection_text.encode_utf16().chain(std::iter::once(0)).collect();
    AppendMenuW(selection_menu, MF_STRING, MENU_BLUR_SELECTION as usize, blur_selection_text_utf16.as_ptr());
    
            let selection_text = "选框工具(&S)";
        let selection_utf16: Vec<u16> = selection_text.encode_utf16().chain(std::iter::once(0)).collect();
        AppendMenuW(menu_bar, MF_POPUP, selection_menu as usize, selection_utf16.as_ptr());
        
        // 新增：批量处理菜单
        let batch_menu = CreatePopupMenu();
        let batch_resize_text = "批量缩图(&R)";
        let batch_resize_utf16: Vec<u16> = batch_resize_text.encode_utf16().chain(std::iter::once(0)).collect();
        AppendMenuW(batch_menu, MF_STRING, MENU_BATCH_RESIZE as usize, batch_resize_utf16.as_ptr());
        
        let batch_text = "批量处理(&B)";
        let batch_text_utf16: Vec<u16> = batch_text.encode_utf16().chain(std::iter::once(0)).collect();
        AppendMenuW(menu_bar, MF_POPUP, batch_menu as usize, batch_text_utf16.as_ptr());
        
        menu_bar
}

// 显示文件打开对话框
unsafe fn show_open_file_dialog(hwnd: HWND) -> Option<String> {
    let mut file_name = [0u16; 260]; // MAX_PATH
    let filter = "图像文件\0*.jpg;*.jpeg;*.png;*.bmp;*.tiff\0所有文件\0*.*\0\0";
    let filter_utf16: Vec<u16> = filter.encode_utf16().collect();
    let title_utf16: Vec<u16> = "选择图像文件\0".encode_utf16().collect();
    
    let mut ofn = OPENFILENAMEW {
        lStructSize: std::mem::size_of::<OPENFILENAMEW>() as u32,
        hwndOwner: hwnd,
        hInstance: null_mut(),
        lpstrFilter: filter_utf16.as_ptr(),
        lpstrCustomFilter: null_mut(),
        nMaxCustFilter: 0,
        nFilterIndex: 1,
        lpstrFile: file_name.as_mut_ptr(),
        nMaxFile: file_name.len() as u32,
        lpstrFileTitle: null_mut(),
        nMaxFileTitle: 0,
        lpstrInitialDir: null_mut(),
        lpstrTitle: title_utf16.as_ptr(),
        Flags: OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST,
        nFileOffset: 0,
        nFileExtension: 0,
        lpstrDefExt: null_mut(),
        lCustData: 0,
        lpfnHook: None,
        lpTemplateName: null_mut(),
        pvReserved: null_mut(),
        dwReserved: 0,
        FlagsEx: 0,
    };
    
    if GetOpenFileNameW(&mut ofn) != 0 {
        // 转换UTF-16路径为String
        let len = file_name.iter().position(|&c| c == 0).unwrap_or(file_name.len());
        let path = String::from_utf16_lossy(&file_name[..len]);
        Some(path)
    } else {
        None
    }
}

// 显示文件保存对话框
unsafe fn show_save_file_dialog(hwnd: HWND) -> Option<String> {
    let mut file_name = [0u16; 260]; // MAX_PATH
    let filter = "PNG 文件\0*.png\0JPEG 文件\0*.jpg;*.jpeg\0BMP 文件\0*.bmp\0TIFF 文件\0*.tiff\0所有文件\0*.*\0\0";
    let filter_utf16: Vec<u16> = filter.encode_utf16().collect();
    let title_utf16: Vec<u16> = "选择保存位置\0".encode_utf16().collect();
    let def_ext_utf16: Vec<u16> = "png\0".encode_utf16().collect();
    
    let mut ofn = OPENFILENAMEW {
        lStructSize: std::mem::size_of::<OPENFILENAMEW>() as u32,
        hwndOwner: hwnd,
        hInstance: null_mut(),
        lpstrFilter: filter_utf16.as_ptr(),
        lpstrCustomFilter: null_mut(),
        nMaxCustFilter: 0,
        nFilterIndex: 1,
        lpstrFile: file_name.as_mut_ptr(),
        nMaxFile: file_name.len() as u32,
        lpstrFileTitle: null_mut(),
        nMaxFileTitle: 0,
        lpstrInitialDir: null_mut(),
        lpstrTitle: title_utf16.as_ptr(),
        Flags: OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST,
        nFileOffset: 0,
        nFileExtension: 0,
        lpstrDefExt: def_ext_utf16.as_ptr(),
        lCustData: 0,
        lpfnHook: None,
        lpTemplateName: null_mut(),
        pvReserved: null_mut(),
        dwReserved: 0,
        FlagsEx: 0,
    };
    
    if GetSaveFileNameW(&mut ofn) != 0 {
        let len = file_name.iter().position(|&c| c == 0).unwrap_or(file_name.len());
        let path = String::from_utf16_lossy(&file_name[..len]);
        Some(path)
    } else {
        None
    }
}

// 显示带有"下一步"按钮的对话框
unsafe fn show_next_step_dialog(hwnd: HWND, title: &str, message: &str) -> bool {
    let title_utf16: Vec<u16> = title.encode_utf16().chain(std::iter::once(0)).collect();
    let message_utf16: Vec<u16> = message.encode_utf16().chain(std::iter::once(0)).collect();
    
    let result = MessageBoxW(
        hwnd,
        message_utf16.as_ptr(),
        title_utf16.as_ptr(),
        MB_OKCANCEL | MB_ICONINFORMATION
    );
    
    result == IDOK
}

// 计算保持宽高比的缩放尺寸
fn calculate_scaled_size(img_width: i32, img_height: i32, window_width: i32, window_height: i32) -> (i32, i32, i32, i32) {
    let img_ratio = img_width as f64 / img_height as f64;
    let window_ratio = window_width as f64 / window_height as f64;
    
    let (scaled_width, scaled_height) = if img_ratio > window_ratio {
        // 图片更宽，以窗口宽度为准
        let scaled_width = window_width;
        let scaled_height = (window_width as f64 / img_ratio) as i32;
        (scaled_width, scaled_height)
    } else {
        // 图片更高，以窗口高度为准
        let scaled_height = window_height;
        let scaled_width = (window_height as f64 * img_ratio) as i32;
        (scaled_width, scaled_height)
    };
    
    // 居中显示
    let x = (window_width - scaled_width) / 2;
    let y = (window_height - scaled_height) / 2;
    
    (x, y, scaled_width, scaled_height)
}

// 计算支持缩放的图像显示尺寸和位置
fn calculate_scaled_size_with_zoom(state: &ImageState, window_width: i32, window_height: i32) -> (i32, i32, i32, i32) {
    calculate_scaled_size_with_zoom_factor(state, state.zoom_factor, window_width, window_height)
}

// 计算指定缩放比例下的图像显示尺寸和位置
fn calculate_scaled_size_with_zoom_factor(state: &ImageState, zoom_factor: f32, window_width: i32, window_height: i32) -> (i32, i32, i32, i32) {
    if !state.has_image() {
        return (0, 0, 0, 0);
    }
    
    // 获取基础适应窗口的尺寸
    let (base_x, base_y, base_w, base_h) = state.calculate_base_display_rect(window_width, window_height);
    
    // 应用指定的缩放比例
    let zoomed_w = (base_w as f32 * zoom_factor) as i32;
    let zoomed_h = (base_h as f32 * zoom_factor) as i32;
    
    // 计算最终位置（基础位置 + 偏移）
    let final_x = base_x + state.zoom_offset_x as i32;
    let final_y = base_y + state.zoom_offset_y as i32;
    
    (final_x, final_y, zoomed_w, zoomed_h)
}

// 更新窗口标题，显示SAM模型状态
unsafe fn update_window_title_with_sam_status(hwnd: HWND, state: &ImageState, sam_loaded: bool) {
    let base_title = if state.has_image() {
        // 获取文件路径
        if let Some(ref file_path) = state.current_file_path {
            // 转换为绝对路径
            let abs_path = match std::fs::canonicalize(file_path) {
                Ok(path) => path.to_string_lossy().to_string(),
                Err(_) => file_path.clone(), // 如果无法获取绝对路径，使用原路径
            };
            
            // 去掉Windows长路径前缀 "\\?\"
            let clean_path = if abs_path.starts_with("\\\\?\\") {
                &abs_path[4..]
            } else {
                &abs_path
            };
            
            format!("图像查看器 - {}x{} - {}", 
                state.current_width, 
                state.current_height,
                clean_path
            )
        } else {
            format!("图像查看器 - {}x{}", 
                state.current_width, 
                state.current_height
            )
        }
    } else {
        "图像查看器".to_string()
    };
    
    let full_title = if sam_loaded {
        format!("{} - SAM模型已加载，请按住鼠标左键拖动方框包围对象外围", base_title)
    } else {
        base_title
    };
    
    let title_utf16: Vec<u16> = full_title.encode_utf16().chain(std::iter::once(0)).collect();
    SetWindowTextW(hwnd, title_utf16.as_ptr());
}

// 更新窗口标题，显示拉直模式状态
unsafe fn update_window_title_with_straighten_mode(hwnd: HWND, state: &ImageState) {
    let base_title = if state.has_image() {
        // 获取文件路径
        if let Some(ref file_path) = state.current_file_path {
            // 转换为绝对路径
            let abs_path = match std::fs::canonicalize(file_path) {
                Ok(path) => path.to_string_lossy().to_string(),
                Err(_) => file_path.clone(), // 如果无法获取绝对路径，使用原路径
            };
            
            // 去掉Windows长路径前缀 "\\?\"
            let clean_path = if abs_path.starts_with("\\\\?\\") {
                &abs_path[4..]
            } else {
                &abs_path
            };
            
            format!("图像查看器 - {}x{} - {}", 
                state.current_width, 
                state.current_height,
                clean_path
            )
        } else {
            format!("图像查看器 - {}x{}", 
                state.current_width, 
                state.current_height
            )
        }
    } else {
        "图像查看器".to_string()
    };
    
    let sam_suffix = if state.sam_processor.as_ref().map_or(false, |sam| sam.is_encoded()) {
        " - SAM模型已加载"
    } else {
        ""
    };
    
    let full_title = if state.straighten_mode {
        format!("{}{} - 请画一条拉直线", base_title, sam_suffix)
    } else {
        format!("{}{}", base_title, sam_suffix)
    };
    
    let title_utf16: Vec<u16> = full_title.encode_utf16().chain(std::iter::once(0)).collect();
    SetWindowTextW(hwnd, title_utf16.as_ptr());
}

// 将窗口坐标映射到图像坐标
unsafe fn window_to_image_coords(hwnd: HWND, img_w: i32, img_h: i32, win_x: i32, win_y: i32) -> Option<(i32,i32)> {
    let mut rect: RECT = std::mem::zeroed();
    GetClientRect(hwnd, &mut rect);
    let window_width = rect.right - rect.left;
    let window_height = rect.bottom - rect.top;
    let (x0, y0, scaled_w, scaled_h) = calculate_scaled_size(img_w, img_h, window_width, window_height);
    if win_x < x0 || win_y < y0 || win_x >= x0 + scaled_w || win_y >= y0 + scaled_h {
        return None;
    }
    let rel_x = win_x - x0;
    let rel_y = win_y - y0;
    if scaled_w <= 0 || scaled_h <= 0 { return None; }
    let img_x = (rel_x as i64 * img_w as i64 / scaled_w as i64) as i32;
    let img_y = (rel_y as i64 * img_h as i64 / scaled_h as i64) as i32;
    Some((img_x.clamp(0, img_w-1), img_y.clamp(0, img_h-1)))
}

// 支持缩放的窗口坐标到图像坐标映射
unsafe fn window_to_image_coords_with_zoom(hwnd: HWND, state: &ImageState, win_x: i32, win_y: i32) -> Option<(i32,i32)> {
    if !state.has_image() {
        return None;
    }
    
    let mut rect: RECT = std::mem::zeroed();
    GetClientRect(hwnd, &mut rect);
    let window_width = rect.right - rect.left;
    let window_height = rect.bottom - rect.top;
    
    let (x0, y0, scaled_w, scaled_h) = calculate_scaled_size_with_zoom(state, window_width, window_height);
    
    // 检查点击是否在图像范围内
    if win_x < x0 || win_y < y0 || win_x >= x0 + scaled_w || win_y >= y0 + scaled_h {
        return None;
    }
    
    let rel_x = win_x - x0;
    let rel_y = win_y - y0;
    if scaled_w <= 0 || scaled_h <= 0 { return None; }
    
    // 转换到图像坐标
    let img_x = (rel_x as i64 * state.current_width as i64 / scaled_w as i64) as i32;
    let img_y = (rel_y as i64 * state.current_height as i64 / scaled_h as i64) as i32;
    
    Some((img_x.clamp(0, state.current_width-1), img_y.clamp(0, state.current_height-1)))
}

fn overlay_mask_on_current(state: &mut ImageState, mask: &[u8], mask_w: i32, mask_h: i32) -> Result<()> {
    use opencv::{imgproc, core as cvcore};
    let current = state.current_mat.as_ref().unwrap();
    let mut result = current.try_clone()?;

    let mut mask_mat = cvcore::Mat::new_rows_cols_with_default(mask_h, mask_w, cvcore::CV_8UC1, cvcore::Scalar::all(0.0))?;
    unsafe {
        let total = (mask_w * mask_h) as usize;
        std::ptr::copy_nonoverlapping(mask.as_ptr(), mask_mat.data_mut() as *mut u8, total);
    }
    let mut mask_resized = cvcore::Mat::default();
    imgproc::resize(&mask_mat, &mut mask_resized, cvcore::Size{ width: state.current_width, height: state.current_height }, 0.0, 0.0, imgproc::INTER_NEAREST)?;

    // 将0/1转换为0/255
    let mut mask_bin = cvcore::Mat::default();
    mask_resized.convert_to(&mut mask_bin, cvcore::CV_8UC1, 255.0, 0.0)?;

    // 生成纯色覆盖图
    let color_overlay = cvcore::Mat::new_rows_cols_with_default(state.current_height, state.current_width, cvcore::CV_8UC3, cvcore::Scalar::new(0.0, 255.0, 0.0, 0.0))?; // 绿色
    let mut blended = cvcore::Mat::default();
    cvcore::add_weighted(&color_overlay, 0.4, current, 0.6, 0.0, &mut blended, -1)?;

    // 仅在mask区域复制混合结果
    blended.copy_to_masked(&mut result, &mask_bin)?;

    state.update_display_from_mat(&result)?;
    Ok(())
}

// 新增：抠图并显示功能
fn extract_mask_and_display(state: &mut ImageState) -> Result<()> {
    use opencv::{imgproc, core as cvcore};
    
    // 检查是否有可用的mask
    let (mask, mask_w, mask_h) = match state.get_current_mask() {
        Some(mask_info) => mask_info.clone(),
        None => {
            return Err(anyhow::anyhow!("没有可用的mask，请先进行SAM分割"));
        }
    };
    
    // 检查是否有原始图像
    let original_img = match &state.original_bgr {
        Some(img) => img,
        None => {
            return Err(anyhow::anyhow!("没有原始图像"));
        }
    };
    
    println!("🔍 开始抠图处理...");
    println!("  mask尺寸: {}x{}", mask_w, mask_h);
    println!("  原图尺寸: {}x{}", original_img.cols(), original_img.rows());
    println!("  原图通道数: {}", original_img.channels());
    println!("  原图数据类型: {:?}", original_img.depth());
    
    // 分析mask数据
    let mut mask_stats = vec![0u32; 256];
    for &val in &mask {
        mask_stats[val as usize] += 1;
    }
    println!("  mask数据统计:");
    for (i, &count) in mask_stats.iter().enumerate().take(10) {
        if count > 0 {
            println!("    值{}: {}个像素", i, count);
        }
    }
    
    // 找到mask的边界框
    let mut y_min = mask_h;
    let mut y_max = 0;
    let mut x_min = mask_w;
    let mut x_max = 0;
    
    for y in 0..mask_h {
        for x in 0..mask_w {
            let idx = (y * mask_w + x) as usize;
            if idx < mask.len() && mask[idx] > 0 {
                y_min = y_min.min(y);
                y_max = y_max.max(y);
                x_min = x_min.min(x);
                x_max = x_max.max(x);
            }
        }
    }
    
    if y_min >= y_max || x_min >= x_max {
        return Err(anyhow::anyhow!("mask中没有有效区域"));
    }
    
    println!("  mask边界框: ({},{}) -> ({},{})", x_min, y_min, x_max, y_max);
    
    // 为边界框添加padding
    let padding = 10;
    y_min = (y_min - padding).max(0);
    y_max = (y_max + padding).min(mask_h - 1);
    x_min = (x_min - padding).max(0);
    x_max = (x_max + padding).min(mask_w - 1);
    
    println!("  添加padding后边界框: ({},{}) -> ({},{})", x_min, y_min, x_max, y_max);
    
    // 将mask坐标映射到原图坐标
    let scale_x = original_img.cols() as f64 / mask_w as f64;
    let scale_y = original_img.rows() as f64 / mask_h as f64;
    
    let orig_x_min = (x_min as f64 * scale_x) as i32;
    let orig_y_min = (y_min as f64 * scale_y) as i32;
    let orig_x_max = (x_max as f64 * scale_x) as i32;
    let orig_y_max = (y_max as f64 * scale_y) as i32;
    
    println!("  原图裁剪区域: ({},{}) -> ({},{})", orig_x_min, orig_y_min, orig_x_max, orig_y_max);
    
    // 裁剪原始图像
    let roi = cvcore::Rect::new(orig_x_min, orig_y_min, orig_x_max - orig_x_min, orig_y_max - orig_y_min);
    let cropped_img = match Mat::roi(original_img, roi) {
        Ok(roi_mat) => roi_mat.try_clone()?,
        Err(_) => {
            return Err(anyhow::anyhow!("裁剪图像失败，ROI超出边界"));
        }
    };
    
    println!("  裁剪后图像尺寸: {}x{}", cropped_img.cols(), cropped_img.rows());
    println!("  裁剪后图像通道数: {}", cropped_img.channels());
    println!("  裁剪后图像数据类型: {:?}", cropped_img.depth());
    
    // 裁剪对应的mask区域
    
    // 裁剪对应的mask区域
    let mask_roi = cvcore::Rect::new(x_min, y_min, x_max - x_min, y_max - y_min);
    let mut mask_mat = cvcore::Mat::new_rows_cols_with_default(mask_h, mask_w, cvcore::CV_8UC1, cvcore::Scalar::all(0.0))?;
    
    // 将mask数据复制到Mat中
    unsafe {
        let total = (mask_w * mask_h) as usize;
        std::ptr::copy_nonoverlapping(mask.as_ptr(), mask_mat.data_mut() as *mut u8, total);
    }
    
    let cropped_mask = match Mat::roi(&mask_mat, mask_roi) {
        Ok(roi_mat) => roi_mat.try_clone()?,
        Err(_) => {
            return Err(anyhow::anyhow!("裁剪mask失败"));
        }
    };
    
    println!("  裁剪后mask尺寸: {}x{}", cropped_mask.cols(), cropped_mask.rows());
    
    // 将mask缩放到裁剪图像的尺寸
    let mut resized_mask = cvcore::Mat::default();
    imgproc::resize(
        &cropped_mask, 
        &mut resized_mask, 
        cvcore::Size::new(cropped_img.cols(), cropped_img.rows()), 
        0.0, 0.0, 
        imgproc::INTER_NEAREST
    )?;
    
    println!("  缩放后mask尺寸: {}x{}", resized_mask.cols(), resized_mask.rows());
    
    // 检查缩放后mask的数据
    let mut mask_min = 255u8;
    let mut mask_max = 0u8;
    let mut mask_positive_count = 0;
    for y in 0..resized_mask.rows() {
        for x in 0..resized_mask.cols() {
            let val = unsafe { *resized_mask.ptr_2d(y, x)? };
            mask_min = mask_min.min(val);
            mask_max = mask_max.max(val);
            if val > 0 {
                mask_positive_count += 1;
            }
        }
    }
    println!("  缩放后mask值范围: [{}, {}]", mask_min, mask_max);
    println!("  缩放后mask正像素数: {} / {}", mask_positive_count, resized_mask.rows() * resized_mask.cols());
    
    // 创建结果图像（黑色背景）
    let mut result = cvcore::Mat::new_rows_cols_with_default(
        cropped_img.rows(), 
        cropped_img.cols(), 
        cvcore::CV_8UC3, 
        cvcore::Scalar::all(0.0)
    )?;
    
    println!("  结果图像尺寸: {}x{}", result.cols(), result.rows());
    println!("  结果图像通道数: {}", result.channels());
    println!("  结果图像数据类型: {:?}", result.depth());
    
    // 使用Python参考代码的方法：创建黑色背景，只在mask区域复制原图像素
    println!("  使用Python参考方法：创建黑色背景，应用mask...");
    
    // 结果图像已经是全零（黑色），现在只在mask区域复制原图像素
    let mut copied_pixels = 0;
    let mut skipped_pixels = 0;
    
    // 添加更详细的调试信息
    println!("  开始像素复制循环...");
    println!("  原图像尺寸: {}x{}", cropped_img.cols(), cropped_img.rows());
    println!("  mask尺寸: {}x{}", resized_mask.cols(), resized_mask.rows());
    println!("  结果图像尺寸: {}x{}", result.cols(), result.rows());
    
    // 检查前几个像素的值作为样本
    println!("  样本像素检查（前3x3区域）:");
    for y in 0..3.min(cropped_img.rows()) {
        for x in 0..3.min(cropped_img.cols()) {
            let mask_val = unsafe { *resized_mask.ptr_2d(y, x)? };
            // 对于BGR三通道图像，使用正确的像素访问方式
            let pixel_ptr = cropped_img.ptr_2d(y, x)?;
            let b_val = unsafe { *pixel_ptr };
            let g_val = unsafe { *pixel_ptr.add(1) };
            let r_val = unsafe { *pixel_ptr.add(2) };
            println!("    位置({},{}): mask={}, BGR=({},{},{})", x, y, mask_val, b_val, g_val, r_val);
        }
    }
    
    for y in 0..cropped_img.rows() {
        for x in 0..cropped_img.cols() {
            let mask_val = unsafe { *resized_mask.ptr_2d(y, x)? };
            if mask_val > 0 {
                // 复制原图像素到结果图
                let src_pixel_ptr = cropped_img.ptr_2d(y, x)?;
                let dst_pixel_ptr = result.ptr_2d_mut(y, x)?;
                unsafe { 
                    *dst_pixel_ptr = *src_pixel_ptr;
                    *dst_pixel_ptr.add(1) = *src_pixel_ptr.add(1);
                    *dst_pixel_ptr.add(2) = *src_pixel_ptr.add(2);
                }
                copied_pixels += 1;
                
                // 每1000个像素打印一次进度
                if copied_pixels % 1000 == 0 {
                    println!("    已复制 {} 个像素...", copied_pixels);
                }
            } else {
                // 非mask区域保持黑色（已经是0，无需处理）
                skipped_pixels += 1;
            }
        }
    }
    
    println!("  像素复制完成:");
    println!("    复制的像素: {}", copied_pixels);
    println!("    跳过的像素: {}", skipped_pixels);
    println!("    结果：mask区域显示原图像素，非mask区域为纯黑色");
    
    // 检查结果图像的前几个像素值
    println!("  结果图像样本像素检查（前3x3区域）:");
    for y in 0..3.min(result.rows()) {
        for x in 0..3.min(result.cols()) {
            let pixel_ptr = result.ptr_2d(y, x)?;
            let b_val = unsafe { *pixel_ptr };
            let g_val = unsafe { *pixel_ptr.add(1) };
            let r_val = unsafe { *pixel_ptr.add(2) };
            println!("    位置({},{}): BGR=({},{},{})", x, y, b_val, g_val, r_val);
        }
    }
    
    println!("✅ 抠图完成，结果尺寸: {}x{}", result.cols(), result.rows());
    
    // 更新显示
    state.update_display_from_mat(&result)?;
    
    Ok(())
}

// 处理菜单命令
unsafe fn handle_menu_command(hwnd: HWND, command_id: u32) {
    let state = IMAGE_STATE.as_mut().unwrap();
    
    match command_id {
        MENU_OPEN_FILE => {
            if let Some(file_path) = show_open_file_dialog(hwnd) {
                match state.load_image(&file_path) {
                    Ok(()) => {
                        println!("✅ 成功加载图像: {}", file_path);
                        
                        // 打开新文件时重置SAM处理器状态，避免上一次的模型影响新文件
                        if let Some(ref mut sam) = state.sam_processor {
                            sam.reset();
                            println!("🔄 SAM处理器已重置，避免影响新打开的文件");
                        }
                        
                        // 更新窗口标题（新图像加载时SAM状态为未加载）
                        update_window_title_with_sam_status(hwnd, state, false);
                        
                        // 更新SAM菜单状态（新图像默认为彩色图）
                        state.update_sam_menu_state(hwnd);
                        
                        InvalidateRect(hwnd, null_mut(), 1);
                    }
                    Err(e) => {
                        println!("❌ 加载图像失败: {}", e);
                        let error_msg = format!("无法加载图像文件:\n{}\n\n错误信息: {}", file_path, e);
                        let error_msg_utf16: Vec<u16> = error_msg.encode_utf16().chain(std::iter::once(0)).collect();
                        let title_utf16: Vec<u16> = "加载错误".encode_utf16().chain(std::iter::once(0)).collect();
                        MessageBoxW(hwnd, error_msg_utf16.as_ptr(), title_utf16.as_ptr(), MB_OK | MB_ICONERROR);
                    }
                }
            }
        }
        MENU_WIA_IMPORT => {
            println!("🖨️ 用户点击了WIA导入菜单");
            // Convert winapi HWND to windows crate HWND
            let hwnd_windows = windows::Win32::Foundation::HWND(hwnd as *mut std::ffi::c_void);
            match wia_scanner::acquire_image_from_scanner(hwnd_windows, None) {
                Ok(scanned_file_path) => {
                    println!("✅ 扫描完成，图像保存到: {}", scanned_file_path);
                    
                    // 自动加载扫描到的图像
                    match state.load_image(&scanned_file_path) {
                        Ok(()) => {
                            println!("✅ 成功加载扫描图像");
                            
                            // 重置SAM处理器状态
                            if let Some(ref mut sam) = state.sam_processor {
                                sam.reset();
                                println!("🔄 SAM处理器已重置");
                            }
                            
                            // 更新窗口标题
                            update_window_title_with_sam_status(hwnd, state, false);
                            
                            // 更新SAM菜单状态
                            state.update_sam_menu_state(hwnd);
                            
                            InvalidateRect(hwnd, null_mut(), 1);
                            
                            // 显示成功提示
                            let success_msg = format!("扫描完成！\n\n图像已保存到:\n{}\n\n图像已自动打开。", scanned_file_path);
                            let success_msg_utf16: Vec<u16> = success_msg.encode_utf16().chain(std::iter::once(0)).collect();
                            let title_utf16: Vec<u16> = "扫描成功".encode_utf16().chain(std::iter::once(0)).collect();
                            MessageBoxW(hwnd, success_msg_utf16.as_ptr(), title_utf16.as_ptr(), MB_OK | MB_ICONINFORMATION);
                        }
                        Err(e) => {
                            println!("❌ 加载扫描图像失败: {}", e);
                            let error_msg = format!("扫描完成，但无法加载图像:\n{}\n\n错误信息: {}", scanned_file_path, e);
                            let error_msg_utf16: Vec<u16> = error_msg.encode_utf16().chain(std::iter::once(0)).collect();
                            let title_utf16: Vec<u16> = "加载错误".encode_utf16().chain(std::iter::once(0)).collect();
                            MessageBoxW(hwnd, error_msg_utf16.as_ptr(), title_utf16.as_ptr(), MB_OK | MB_ICONERROR);
                        }
                    }
                }
                Err(e) => {
                    println!("❌ WIA扫描失败: {}", e);
                    let error_msg = format!("扫描操作失败:\n\n{}\n\n请检查:\n1. 扫描仪是否已连接并开机\n2. 扫描仪驱动是否已正确安装\n3. WIA服务是否正常运行", e);
                    let error_msg_utf16: Vec<u16> = error_msg.encode_utf16().chain(std::iter::once(0)).collect();
                    let title_utf16: Vec<u16> = "扫描失败".encode_utf16().chain(std::iter::once(0)).collect();
                    MessageBoxW(hwnd, error_msg_utf16.as_ptr(), title_utf16.as_ptr(), MB_OK | MB_ICONERROR);
                }
            }
        }
        MENU_EXIF_PANEL => {
            toggle_exif_panel(hwnd);
        }
        MENU_SAVE_FILE => {
            if let Some(ref mat) = state.current_mat {
                // 确保加载完整分辨率以保存最佳质量
                if let Err(e) = state.ensure_full_resolution() {
                    println!("⚠️ 无法加载完整分辨率: {}，将保存当前版本", e);
                }
                
                if let Some(path) = show_save_file_dialog(hwnd) {
                    // 重新获取mat（可能已更新为完整分辨率）
                    let mat_to_save = state.current_mat.as_ref().unwrap();
                    match save_mat_with_unicode_path(mat_to_save, &path) {
                        Ok(final_path) => {
                            println!("✅ 已保存: {}", final_path);
                        }
                        Err(e) => {
                            println!("❌ 保存失败: {}", e);
                            let error_msg = format!("无法保存到:\n{}\n\n错误信息: {}", path, e);
                            let error_msg_utf16: Vec<u16> = error_msg.encode_utf16().chain(std::iter::once(0)).collect();
                            let title_utf16: Vec<u16> = "保存错误".encode_utf16().chain(std::iter::once(0)).collect();
                            MessageBoxW(hwnd, error_msg_utf16.as_ptr(), title_utf16.as_ptr(), MB_OK | MB_ICONERROR);
                        }
                    }
                }
            } else {
                let msg = "没有可保存的图像，请先进行处理或打开图像";
                let msg_utf16: Vec<u16> = msg.encode_utf16().chain(std::iter::once(0)).collect();
                let title_utf16: Vec<u16> = "提示".encode_utf16().chain(std::iter::once(0)).collect();
                MessageBoxW(hwnd, msg_utf16.as_ptr(), title_utf16.as_ptr(), MB_OK | MB_ICONINFORMATION);
            }
        }
        MENU_CLOSE_FILE => {
            if state.has_image() {
                state.close_current_file();
                // 更新窗口标题为初始状态
                update_window_title_with_sam_status(hwnd, state, false);
                // 重绘窗口以显示黑屏状态
                InvalidateRect(hwnd, null_mut(), 1);
            } else {
                let msg = "当前没有打开的文件";
                let msg_utf16: Vec<u16> = msg.encode_utf16().chain(std::iter::once(0)).collect();
                let title_utf16: Vec<u16> = "提示".encode_utf16().chain(std::iter::once(0)).collect();
                MessageBoxW(hwnd, msg_utf16.as_ptr(), title_utf16.as_ptr(), MB_OK | MB_ICONINFORMATION);
            }
        }
        MENU_STRAIGHTEN_DRAW_LINE => {
            if state.has_image() {
                // 激活画直线模式
                state.straighten_mode = true;
                state.straighten_start = None;
                state.straighten_end = None;
                // 关闭其他模式
                state.selection_mode = false;
                // 更新窗口标题显示拉直提示
                update_window_title_with_straighten_mode(hwnd, state);
                println!("🔧 画直线模式已激活，请在图像上拖拽绘制直线");
            } else {
                let msg = "没有可用的图像，请先加载图像";
                let msg_utf16: Vec<u16> = msg.encode_utf16().chain(std::iter::once(0)).collect();
                let title_utf16: Vec<u16> = "提示".encode_utf16().chain(std::iter::once(0)).collect();
                MessageBoxW(hwnd, msg_utf16.as_ptr(), title_utf16.as_ptr(), MB_OK | MB_ICONINFORMATION);
            }
        }
        MENU_STRAIGHTEN_HORIZONTAL => {
            if state.has_image() {
                // 检查是否已经画了直线
                if let (Some((x1, y1)), Some((x2, y2))) = (state.straighten_start, state.straighten_end) {
                    // 计算直线与水平线的夹角
                    let angle = calculate_horizontal_angle(x1, y1, x2, y2);
                    println!("🔧 计算出的角度: {:.2}度", angle);
                    
                    // 旋转图像
                    if let Some(ref current) = state.current_mat.clone() {
                        match rotate_image_angle(current, angle) {
                            Ok(rotated) => {
                                if let Ok(()) = state.update_display_from_mat(&rotated) {
                                    println!("✅ 水平校正完成，图像已旋转{:.2}度", angle);
                                    // 清除拉直线状态
                                    state.straighten_mode = false;
                                    state.straighten_start = None;
                                    state.straighten_end = None;
                                    // 更新窗口标题
                                    let sam_loaded = state.sam_processor.as_ref().map_or(false, |sam| sam.is_encoded());
                                    update_window_title_with_sam_status(hwnd, state, sam_loaded);
                                    InvalidateRect(hwnd, null_mut(), 1);
                                } else {
                                    println!("❌ 更新显示失败");
                                }
                            }
                            Err(e) => {
                                println!("❌ 图像旋转失败: {}", e);
                                let error_msg = format!("图像旋转失败：\n{}", e);
                                let error_msg_utf16: Vec<u16> = error_msg.encode_utf16().chain(std::iter::once(0)).collect();
                                let title_utf16: Vec<u16> = "错误".encode_utf16().chain(std::iter::once(0)).collect();
                                MessageBoxW(hwnd, error_msg_utf16.as_ptr(), title_utf16.as_ptr(), MB_OK | MB_ICONERROR);
                            }
                        }
                    }
                } else {
                    let msg = "请先使用\"画直线\"功能画一条直线";
                    let msg_utf16: Vec<u16> = msg.encode_utf16().chain(std::iter::once(0)).collect();
                    let title_utf16: Vec<u16> = "提示".encode_utf16().chain(std::iter::once(0)).collect();
                    MessageBoxW(hwnd, msg_utf16.as_ptr(), title_utf16.as_ptr(), MB_OK | MB_ICONINFORMATION);
                }
            } else {
                let msg = "没有可用的图像，请先加载图像";
                let msg_utf16: Vec<u16> = msg.encode_utf16().chain(std::iter::once(0)).collect();
                let title_utf16: Vec<u16> = "提示".encode_utf16().chain(std::iter::once(0)).collect();
                MessageBoxW(hwnd, msg_utf16.as_ptr(), title_utf16.as_ptr(), MB_OK | MB_ICONINFORMATION);
            }
        }
        MENU_EXIT => {
            // 发送WM_CLOSE消息来关闭程序，等同于点击窗口右上角的X按钮
            PostMessageW(hwnd, WM_CLOSE, 0, 0);
        }
        MENU_TO_GRAY => {
            if let Some(ref current) = state.current_mat.clone() {
                match cv_ops::to_gray(&current) {
                    Ok(gray) => {
                        if let Ok(()) = state.update_display_from_mat(&gray) {
                            state.is_grayscale = true; // 设置为灰度图状态
                            state.update_sam_menu_state(hwnd); // 更新SAM菜单状态
                            println!("✅ 灰度转换完成");
                            InvalidateRect(hwnd, null_mut(), 1);
                        }
                    }
                    Err(e) => println!("❌ 灰度转换失败: {}", e),
                }
            }
        }
        MENU_GAUSSIAN => {
            if let Some(ref current) = state.current_mat.clone() {
                match cv_ops::to_gray(&current).and_then(|gray| cv_ops::gaussian(&gray, 5, 2.0)) {
                    Ok(blurred) => {
                        if let Ok(()) = state.update_display_from_mat(&blurred) {
                            println!("✅ 高斯模糊完成");
                            InvalidateRect(hwnd, null_mut(), 1);
                        }
                    }
                    Err(e) => println!("❌ 高斯模糊失败: {}", e),
                }
            }
        }
        MENU_CANNY => {
            if let Some(ref current) = state.current_mat.clone() {
                match cv_ops::to_gray(&current)
                    .and_then(|gray| cv_ops::gaussian(&gray, 5, 0.0))
                    .and_then(|blurred| cv_ops::canny(&blurred, 50.0, 150.0)) {
                    Ok(edges) => {
                        if let Ok(()) = state.update_display_from_mat(&edges) {
                            state.is_grayscale = true; // 设置为灰度图状态
                            state.update_sam_menu_state(hwnd); // 更新SAM菜单状态
                            println!("✅ Canny边缘检测完成");
                            InvalidateRect(hwnd, null_mut(), 1);
                        }
                    }
                    Err(e) => println!("❌ Canny边缘检测失败: {}", e),
                }
            }
        }
        MENU_ADAPTIVE_THRESHOLD => {
            if let Some(ref current) = state.current_mat.clone() {
                match cv_ops::to_gray(&current)
                    .and_then(|gray| cv_ops::adaptive_threshold(&gray, 255.0, 11, 2.0)) {
                    Ok(binary) => {
                        if let Ok(()) = state.update_display_from_mat(&binary) {
                            state.is_grayscale = true; // 设置为灰度图状态
                            state.update_sam_menu_state(hwnd); // 更新SAM菜单状态
                            println!("✅ 自适应阈值处理完成");
                            InvalidateRect(hwnd, null_mut(), 1);
                        }
                    }
                    Err(e) => println!("❌ 自适应阈值处理失败: {}", e),
                }
            }
        }
        MENU_MORPH_CLOSE => {
            if let Some(ref current) = state.current_mat.clone() {
                match cv_ops::to_gray(&current)
                    .and_then(|gray| cv_ops::adaptive_threshold(&gray, 255.0, 11, 2.0))
                    .and_then(|binary| cv_ops::morph_close(&binary, 3)) {
                    Ok(closed) => {
                        if let Ok(()) = state.update_display_from_mat(&closed) {
                            println!("✅ 形态学闭运算完成");
                            InvalidateRect(hwnd, null_mut(), 1);
                        }
                    }
                    Err(e) => println!("❌ 形态学闭运算失败: {}", e),
                }
            }
        }
        MENU_DILATE => {
            if let Some(ref current) = state.current_mat.clone() {
                match cv_ops::to_gray(&current)
                    .and_then(|gray| cv_ops::adaptive_threshold(&gray, 255.0, 11, 2.0))
                    .and_then(|binary| cv_ops::dilate(&binary, 3, 1)) {
                    Ok(dilated) => {
                        if let Ok(()) = state.update_display_from_mat(&dilated) {
                            println!("✅ 膨胀操作完成");
                            InvalidateRect(hwnd, null_mut(), 1);
                        }
                    }
                    Err(e) => println!("❌ 膨胀操作失败: {}", e),
                }
            }
        }
        MENU_FIND_CONTOURS => {
            if let Some(ref current) = state.current_mat.clone() {
                match cv_ops::to_gray(&current)
                    .and_then(|gray| cv_ops::adaptive_threshold(&gray, 255.0, 11, 2.0))
                    .and_then(|binary| cv_ops::find_and_draw_contours(&binary)) {
                    Ok(contours) => {
                        println!("✅ 轮廓检测完成，找到 {} 个轮廓", contours.len());
                        // 这里可以在原图上绘制轮廓，暂时只输出结果
                    }
                    Err(e) => println!("❌ 轮廓检测失败: {}", e),
                }
            }
        }
        MENU_PERSPECTIVE_TRANSFORM => {
            if let Some(ref current) = state.current_mat.clone() {
                // 确保加载完整分辨率（透视变换需要高质量）
                if let Err(e) = state.ensure_full_resolution() {
                    println!("❌ 加载完整分辨率失败: {}", e);
                    let error_msg = format!("无法加载完整分辨率图像：{}", e);
                    let error_msg_utf16: Vec<u16> = error_msg.encode_utf16().chain(std::iter::once(0)).collect();
                    let title_utf16: Vec<u16> = "错误".encode_utf16().chain(std::iter::once(0)).collect();
                    MessageBoxW(hwnd, error_msg_utf16.as_ptr(), title_utf16.as_ptr(), MB_OK | MB_ICONERROR);
                    return;
                }
                
                println!("开始简化透视变换...");
                // 重新获取mat（可能已更新为完整分辨率）
                let current_mat = state.current_mat.as_ref().unwrap();
                match cv_ops::perspective_transform_simplified(current_mat) {
                    Ok(Some(transformed)) => {
                        if let Ok(()) = state.update_display_from_mat(&transformed) {
                            println!("✅ 简化透视变换完成");
                            InvalidateRect(hwnd, null_mut(), 1);
                        }
                    }
                    Ok(None) => {
                        println!("❌ 透视变换失败：未能检测到色卡");
                        let msg = "未能检测到色卡，请确保图像中包含清晰的矩形色卡";
                        let msg_utf16: Vec<u16> = msg.encode_utf16().chain(std::iter::once(0)).collect();
                        let title_utf16: Vec<u16> = "透视变换失败".encode_utf16().chain(std::iter::once(0)).collect();
                        MessageBoxW(hwnd, msg_utf16.as_ptr(), title_utf16.as_ptr(), MB_OK | MB_ICONWARNING);
                    }
                    Err(e) => {
                        println!("❌ 透视变换出错: {}", e);
                    }
                }
            }
        }
        MENU_MANUAL_PERSPECTIVE => {
            if let Some(ref current) = state.current_mat.clone() {
                // 确保加载完整分辨率（透视变换需要高质量）
                if let Err(e) = state.ensure_full_resolution() {
                    println!("❌ 加载完整分辨率失败: {}", e);
                    let error_msg = format!("无法加载完整分辨率图像：{}", e);
                    let error_msg_utf16: Vec<u16> = error_msg.encode_utf16().chain(std::iter::once(0)).collect();
                    let title_utf16: Vec<u16> = "错误".encode_utf16().chain(std::iter::once(0)).collect();
                    MessageBoxW(hwnd, error_msg_utf16.as_ptr(), title_utf16.as_ptr(), MB_OK | MB_ICONERROR);
                    return;
                }
                
                // 进入手动透视变换模式
                state.manual_perspective_mode = true;
                state.manual_perspective_points.clear();
                
                let msg = "手动透视变换模式已启动！\n\n请用鼠标左键在图像上依次点击四个角点：\n1. 左上角\n2. 右上角\n3. 右下角\n4. 左下角\n\n点击完成后，程序将自动执行透视变换。\n\n按ESC键可以取消操作。";
                let msg_utf16: Vec<u16> = msg.encode_utf16().chain(std::iter::once(0)).collect();
                let title_utf16: Vec<u16> = "手动透视变换".encode_utf16().chain(std::iter::once(0)).collect();
                MessageBoxW(hwnd, msg_utf16.as_ptr(), title_utf16.as_ptr(), MB_OK | MB_ICONINFORMATION);
                
                println!("✅ 手动透视变换模式已启动，等待用户选择四个角点...");
                InvalidateRect(hwnd, null_mut(), 1);
            } else {
                let msg = "没有可用的图像，请先加载图像";
                let msg_utf16: Vec<u16> = msg.encode_utf16().chain(std::iter::once(0)).collect();
                let title_utf16: Vec<u16> = "错误".encode_utf16().chain(std::iter::once(0)).collect();
                MessageBoxW(hwnd, msg_utf16.as_ptr(), title_utf16.as_ptr(), MB_OK | MB_ICONERROR);
            }
        }
        MENU_TEMPLATE_MATCHING => {
            if let Some(ref scene) = state.current_mat.clone() {
                println!("开始模板匹配定位色卡...");
                
                let msg = "模板匹配色卡检测：\n\n当前显示的图像将作为场景图像（包含色卡的照片）\n现在请选择标准色卡模板图像进行匹配";
                let msg_utf16: Vec<u16> = msg.encode_utf16().chain(std::iter::once(0)).collect();
                let title_utf16: Vec<u16> = "选择色卡模板".encode_utf16().chain(std::iter::once(0)).collect();
                
                if MessageBoxW(hwnd, msg_utf16.as_ptr(), title_utf16.as_ptr(), MB_OKCANCEL | MB_ICONINFORMATION) == IDOK {
                    if let Some(template_path) = show_open_file_dialog(hwnd) {
                        println!("模板图像: {}", template_path);
                        println!("场景图像: 当前显示的图像");
                        
                        match cv_ops::match_and_transform_color_card(&template_path, &scene) {
                            Ok(result) => {
                                // 第一步：显示检测结果（带绿色框的原图）
                                if let Some(ref detection_img) = result.scene_with_detection {
                                    if let Ok(()) = state.update_display_from_mat(detection_img) {
                                        println!("✅ 模板匹配定位色卡完成，匹配度: {:.1}%", result.match_confidence * 100.0);
                                        InvalidateRect(hwnd, null_mut(), 1);
                                        
                                        // 显示确认对话框，询问是否要裁剪色卡
                                        let confirm_msg = format!(
                                            "色卡检测成功！\n\n匹配置信度: {:.1}%\n\n当前显示的是标注了色卡位置的原图像。\n绿色框和彩色角点标出了检测到的色卡位置。\n\n是否要裁剪出色卡区域进行颜色分析？",
                                            result.match_confidence * 100.0
                                        );
                                        let confirm_msg_utf16: Vec<u16> = confirm_msg.encode_utf16().chain(std::iter::once(0)).collect();
                                        let confirm_title_utf16: Vec<u16> = "色卡检测成功".encode_utf16().chain(std::iter::once(0)).collect();
                                        
                                        let user_choice = MessageBoxW(
                                            hwnd, 
                                            confirm_msg_utf16.as_ptr(), 
                                            confirm_title_utf16.as_ptr(), 
                                            MB_YESNO | MB_ICONQUESTION
                                        );
                                        
                                        if user_choice == IDYES {
                                            // 第二步：用户选择"是"，显示裁剪后色卡并更新current_mat
                                            if let Some(ref warped) = result.warped_image {
                                                // 更新显示并同时更新current_mat为裁剪后的图像
                                                state.current_mat = Some(warped.clone());
                                                if let Ok(()) = state.update_display_from_mat(warped) {
                                                    InvalidateRect(hwnd, null_mut(), 1);
                                                    
                                                    let success_msg = "色卡裁剪完成！\n\n当前显示的是模板匹配透视变换后的色卡图像。\n调试图像已保存到debug_out目录。\n\n✅ 重要提示：当前工作图像已切换为裁剪后的色卡\n后续的所有图像处理操作都将基于这张裁剪后的色卡进行。\n\n您现在可以使用\"读取色卡色块颜色\"功能提取颜色信息。";
                                                    let success_msg_utf16: Vec<u16> = success_msg.encode_utf16().chain(std::iter::once(0)).collect();
                                                    let success_title_utf16: Vec<u16> = "裁剪成功".encode_utf16().chain(std::iter::once(0)).collect();
                                                    MessageBoxW(hwnd, success_msg_utf16.as_ptr(), success_title_utf16.as_ptr(), MB_OK | MB_ICONINFORMATION);
                                                    
                                                    println!("✅ 工作图像已更新为裁剪后的色卡 ({}x{})", warped.cols(), warped.rows());
                                                } else {
                                                    println!("❌ 更新裁剪后的色卡图像显示失败");
                                                }
                                            } else {
                                                println!("❌ 模板匹配结果中没有裁剪后的图像");
                                                let error_msg = "内部错误：模板匹配成功但没有生成裁剪后的图像。";
                                                let error_msg_utf16: Vec<u16> = error_msg.encode_utf16().chain(std::iter::once(0)).collect();
                                                let error_title_utf16: Vec<u16> = "内部错误".encode_utf16().chain(std::iter::once(0)).collect();
                                                MessageBoxW(hwnd, error_msg_utf16.as_ptr(), error_title_utf16.as_ptr(), MB_OK | MB_ICONERROR);
                                            }
                                        } else {
                                            // 用户选择"否"，保持当前的检测结果显示
                                            let keep_msg = "保持当前显示。\n\n如果需要裁剪色卡，请重新运行\"模板匹配定位色卡\"功能。";
                                            let keep_msg_utf16: Vec<u16> = keep_msg.encode_utf16().chain(std::iter::once(0)).collect();
                                            let keep_title_utf16: Vec<u16> = "已取消裁剪".encode_utf16().chain(std::iter::once(0)).collect();
                                            MessageBoxW(hwnd, keep_msg_utf16.as_ptr(), keep_title_utf16.as_ptr(), MB_OK | MB_ICONINFORMATION);
                                        }
                                    }
                                } else {
                                    println!("❌ 检测结果图像为空");
                                }
                            }
                            Err(e) => {
                                println!("❌ 模板匹配出错: {}", e);
                                let error_msg = format!("模板匹配过程中发生错误：\n{}\n\n请检查：\n1. 模板图像文件是否可读\n2. 图像格式是否支持\n3. OpenCV功能是否正常", e);
                                let error_msg_utf16: Vec<u16> = error_msg.encode_utf16().chain(std::iter::once(0)).collect();
                                let error_title_utf16: Vec<u16> = "处理错误".encode_utf16().chain(std::iter::once(0)).collect();
                                MessageBoxW(hwnd, error_msg_utf16.as_ptr(), error_title_utf16.as_ptr(), MB_OK | MB_ICONERROR);
                            }
                        }
                    }
                }
            } else {
                let msg = "没有可用的场景图像。\n\n请先打开一张包含色卡的照片作为场景图像，然后再使用模板匹配功能。";
                let msg_utf16: Vec<u16> = msg.encode_utf16().chain(std::iter::once(0)).collect();
                let title_utf16: Vec<u16> = "提示".encode_utf16().chain(std::iter::once(0)).collect();
                MessageBoxW(hwnd, msg_utf16.as_ptr(), title_utf16.as_ptr(), MB_OK | MB_ICONINFORMATION);
            }
        }
        MENU_EXTRACT_COLOR_BLOCKS => {
            if let Some(ref current) = state.current_mat.clone() {
                println!("开始从当前显示的图像中提取色块颜色...");
                match cv_ops::extract_color_blocks_grid(&current) {
                    Ok(result) => {
                        println!("✅ 成功提取了 {} 个色块", result.color_blocks.len());
                        
                        // 第一步：显示网格划分图像
                        if let Ok(()) = state.update_display_from_mat(&result.grid_image) {
                            InvalidateRect(hwnd, null_mut(), 1);
                            
                            // 显示"下一步"对话框
                            if show_next_step_dialog(hwnd, "网格划分完成", 
                                "已完成网格划分，当前显示的是网格线和色块编号。\n\n点击\"确定\"继续查看色卡参考图，或点击\"取消\"停止。") {
                                
                                // 保存网格图像
                                if let Err(e) = ocv_bridge::save_debug_image(&result.grid_image, "grid_04_grid_lines.png") {
                                    println!("保存网格图像失败: {}", e);
                                }
                                
                                // 第二步：显示色卡参考图
                                if let Ok(()) = state.update_display_from_mat(&result.color_palette) {
                                    InvalidateRect(hwnd, null_mut(), 1);
                                    
                                    // 保存色卡参考图
                                    if let Err(e) = ocv_bridge::save_debug_image(&result.color_palette, "grid_05_color_palette.png") {
                                        println!("保存色卡参考图失败: {}", e);
                                    }
                                    
                                    // 显示提取结果的详细信息
                                    let mut result_msg = String::from("成功提取了24个色块的RGB值：\n\n");
                                    for (i, block) in result.color_blocks.iter().enumerate() {
                                        if i < 8 {  // 只显示前8个，避免消息框过长
                                            result_msg.push_str(&format!(
                                                "色块{}: RGB({},{},{}) {}\n",
                                                block.block_number, block.r, block.g, block.b, block.hex
                                            ));
                                        }
                                    }
                                    if result.color_blocks.len() > 8 {
                                        result_msg.push_str("...\n");
                                    }
                                    result_msg.push_str("\n详细信息已保存到调试输出目录\n当前显示的是24个色块的色卡参考图");
                                    
                                    let msg_utf16: Vec<u16> = result_msg.encode_utf16().chain(std::iter::once(0)).collect();
                                    let title_utf16: Vec<u16> = "色块提取成功".encode_utf16().chain(std::iter::once(0)).collect();
                                    MessageBoxW(hwnd, msg_utf16.as_ptr(), title_utf16.as_ptr(), MB_OK | MB_ICONINFORMATION);
                                } else {
                                    println!("❌ 更新色卡参考图显示失败");
                                }
                            } else {
                                println!("用户取消了色块提取过程");
                                // 保存网格图像（即使用户取消也保存）
                                if let Err(e) = ocv_bridge::save_debug_image(&result.grid_image, "grid_04_grid_lines.png") {
                                    println!("保存网格图像失败: {}", e);
                                }
                            }
                        } else {
                            println!("❌ 更新网格图像显示失败");
                        }
                    }
                    Err(e) => {
                        println!("❌ 色块提取失败: {}", e);
                        let error_msg = format!("色块提取失败：\n{}\n\n请确保当前显示的是经过透视变换的色卡图像", e);
                        let error_msg_utf16: Vec<u16> = error_msg.encode_utf16().chain(std::iter::once(0)).collect();
                        let title_utf16: Vec<u16> = "色块提取失败".encode_utf16().chain(std::iter::once(0)).collect();
                        MessageBoxW(hwnd, error_msg_utf16.as_ptr(), title_utf16.as_ptr(), MB_OK | MB_ICONERROR);
                    }
                }
            } else {
                let msg = "没有可用的图像，请先加载图像并进行透视变换";
                let msg_utf16: Vec<u16> = msg.encode_utf16().chain(std::iter::once(0)).collect();
                let title_utf16: Vec<u16> = "提示".encode_utf16().chain(std::iter::once(0)).collect();
                MessageBoxW(hwnd, msg_utf16.as_ptr(), title_utf16.as_ptr(), MB_OK | MB_ICONINFORMATION);
            }
        }
        MENU_RESTORE => {
            if let Some(ref original) = state.original_bgr.clone() {
                if let Ok(()) = state.update_display_from_mat(&original) {
                    state.is_grayscale = false; // 重置为彩色图状态
                    state.reset_zoom(); // 重置缩放和平移状态，让图像居中显示
                    state.update_sam_menu_state(hwnd); // 更新SAM菜单状态
                    println!("✅ 恢复原图完成");
                    InvalidateRect(hwnd, null_mut(), 1);
                }
            }
        }
        MENU_RESIZE_IMAGE => {
            println!("用户点击了调整图像大小菜单");
            if let Some(ref current) = state.current_mat.clone() {
                println!("当前图像存在，尺寸: {}x{}", state.current_width, state.current_height);
                println!("准备调用 show_resize_dialog 函数");
                if let Some(result) = show_resize_dialog(hwnd, state.current_width, state.current_height) {
                    println!("对话框返回结果: {:?}", result);
                    println!("开始调用 resize_image 函数");
                    match resize_image(current, result.new_width, result.new_height, result.quality) {
                        Ok(resized) => {
                            println!("图像调整成功，开始更新显示");
                            if let Ok(()) = state.update_display_from_mat(&resized) {
                                println!("✅ 图像大小调整完成: {}x{} -> {}x{}", 
                                    state.current_width, state.current_height, 
                                    result.new_width, result.new_height);
                                InvalidateRect(hwnd, null_mut(), 1);
                            }
                        }
                        Err(e) => {
                            println!("❌ 图像大小调整失败: {}", e);
                            let error_msg = format!("图像大小调整失败：\n{}", e);
                            let error_msg_utf16: Vec<u16> = error_msg.encode_utf16().chain(std::iter::once(0)).collect();
                            let title_utf16: Vec<u16> = "调整失败".encode_utf16().chain(std::iter::once(0)).collect();
                            MessageBoxW(hwnd, error_msg_utf16.as_ptr(), title_utf16.as_ptr(), MB_OK | MB_ICONERROR);
                        }
                    }
                } else {
                    println!("对话框返回 None，用户可能取消了操作");
                }
            } else {
                println!("没有可用的图像");
                let msg = "没有可用的图像，请先加载图像";
                let msg_utf16: Vec<u16> = msg.encode_utf16().chain(std::iter::once(0)).collect();
                let title_utf16: Vec<u16> = "提示".encode_utf16().chain(std::iter::once(0)).collect();
                MessageBoxW(hwnd, msg_utf16.as_ptr(), title_utf16.as_ptr(), MB_OK | MB_ICONINFORMATION);
            }
        }
        // 图像旋转命令处理
        MENU_ROTATE_180 => {
            if let Some(ref current) = state.current_mat.clone() {
                match rotate_image_180(current) {
                    Ok(rotated) => {
                        if let Ok(()) = state.update_display_from_mat(&rotated) {
                            println!("✅ 图像180度旋转完成");
                            InvalidateRect(hwnd, null_mut(), 1);
                        }
                    }
                    Err(e) => {
                        println!("❌ 图像旋转失败: {}", e);
                        let error_msg = format!("图像旋转失败：\n{}", e);
                        let error_msg_utf16: Vec<u16> = error_msg.encode_utf16().chain(std::iter::once(0)).collect();
                        let title_utf16: Vec<u16> = "旋转失败".encode_utf16().chain(std::iter::once(0)).collect();
                        MessageBoxW(hwnd, error_msg_utf16.as_ptr(), title_utf16.as_ptr(), MB_OK | MB_ICONERROR);
                    }
                }
            } else {
                let msg = "没有可用的图像，请先加载图像";
                let msg_utf16: Vec<u16> = msg.encode_utf16().chain(std::iter::once(0)).collect();
                let title_utf16: Vec<u16> = "提示".encode_utf16().chain(std::iter::once(0)).collect();
                MessageBoxW(hwnd, msg_utf16.as_ptr(), title_utf16.as_ptr(), MB_OK | MB_ICONINFORMATION);
            }
        }
        MENU_ROTATE_90_CW => {
            if let Some(ref current) = state.current_mat.clone() {
                match rotate_image_90_cw(current) {
                    Ok(rotated) => {
                        if let Ok(()) = state.update_display_from_mat(&rotated) {
                            println!("✅ 图像顺时针90度旋转完成");
                            InvalidateRect(hwnd, null_mut(), 1);
                        }
                    }
                    Err(e) => {
                        println!("❌ 图像旋转失败: {}", e);
                        let error_msg = format!("图像旋转失败：\n{}", e);
                        let error_msg_utf16: Vec<u16> = error_msg.encode_utf16().chain(std::iter::once(0)).collect();
                        let title_utf16: Vec<u16> = "旋转失败".encode_utf16().chain(std::iter::once(0)).collect();
                        MessageBoxW(hwnd, error_msg_utf16.as_ptr(), title_utf16.as_ptr(), MB_OK | MB_ICONERROR);
                    }
                }
            } else {
                let msg = "没有可用的图像，请先加载图像";
                let msg_utf16: Vec<u16> = msg.encode_utf16().chain(std::iter::once(0)).collect();
                let title_utf16: Vec<u16> = "提示".encode_utf16().chain(std::iter::once(0)).collect();
                MessageBoxW(hwnd, msg_utf16.as_ptr(), title_utf16.as_ptr(), MB_OK | MB_ICONINFORMATION);
            }
        }
        MENU_ROTATE_90_CCW => {
            if let Some(ref current) = state.current_mat.clone() {
                match rotate_image_90_ccw(current) {
                    Ok(rotated) => {
                        if let Ok(()) = state.update_display_from_mat(&rotated) {
                            println!("✅ 图像逆时针90度旋转完成");
                            InvalidateRect(hwnd, null_mut(), 1);
                        }
                    }
                    Err(e) => {
                        println!("❌ 图像旋转失败: {}", e);
                        let error_msg = format!("图像旋转失败：\n{}", e);
                        let error_msg_utf16: Vec<u16> = error_msg.encode_utf16().chain(std::iter::once(0)).collect();
                        let title_utf16: Vec<u16> = "旋转失败".encode_utf16().chain(std::iter::once(0)).collect();
                        MessageBoxW(hwnd, error_msg_utf16.as_ptr(), title_utf16.as_ptr(), MB_OK | MB_ICONERROR);
                    }
                }
            } else {
                let msg = "没有可用的图像，请先加载图像";
                let msg_utf16: Vec<u16> = msg.encode_utf16().chain(std::iter::once(0)).collect();
                let title_utf16: Vec<u16> = "提示".encode_utf16().chain(std::iter::once(0)).collect();
                MessageBoxW(hwnd, msg_utf16.as_ptr(), title_utf16.as_ptr(), MB_OK | MB_ICONINFORMATION);
            }
        }
        MENU_ROTATE_ANGLE => {
            if let Some(ref current) = state.current_mat.clone() {
                // 显示角度输入对话框
                if let Some(angle) = show_angle_input_dialog(hwnd) {
                    match rotate_image_angle(current, angle) {
                        Ok(rotated) => {
                            if let Ok(()) = state.update_display_from_mat(&rotated) {
                                println!("✅ 图像旋转{}度完成", angle);
                                InvalidateRect(hwnd, null_mut(), 1);
                            }
                        }
                        Err(e) => {
                            println!("❌ 图像旋转失败: {}", e);
                            let error_msg = format!("图像旋转失败：\n{}", e);
                            let error_msg_utf16: Vec<u16> = error_msg.encode_utf16().chain(std::iter::once(0)).collect();
                            let title_utf16: Vec<u16> = "旋转失败".encode_utf16().chain(std::iter::once(0)).collect();
                            MessageBoxW(hwnd, error_msg_utf16.as_ptr(), title_utf16.as_ptr(), MB_OK | MB_ICONERROR);
                        }
                    }
                }
            } else {
                let msg = "没有可用的图像，请先加载图像";
                let msg_utf16: Vec<u16> = msg.encode_utf16().chain(std::iter::once(0)).collect();
                let title_utf16: Vec<u16> = "提示".encode_utf16().chain(std::iter::once(0)).collect();
                MessageBoxW(hwnd, msg_utf16.as_ptr(), title_utf16.as_ptr(), MB_OK | MB_ICONINFORMATION);
            }
        }
        MENU_SAM_INTERACTIVE => {
            if state.is_grayscale {
                let msg = "SAM分割功能不支持灰度图或黑白图，请使用彩色图像或点击'恢复原图'恢复彩色图像";
                let msg_utf16: Vec<u16> = msg.encode_utf16().chain(std::iter::once(0)).collect();
                let title_utf16: Vec<u16> = "SAM分割限制".encode_utf16().chain(std::iter::once(0)).collect();
                MessageBoxW(hwnd, msg_utf16.as_ptr(), title_utf16.as_ptr(), MB_OK | MB_ICONWARNING);
                return;
            }
            
            if let Some(ref current) = state.current_mat.clone() {
                // 首先初始化SAM处理器并编码图像
                match state.init_sam_processor() {
                    Ok(()) => {
                        if let Some(ref mut sam) = state.sam_processor {
                            match sam.encode_image(current) {
                                Ok(()) => {
                                    println!("✅ SAM图像编码完成");
                                    
                                    // 更新窗口标题显示SAM模型已加载
                                    update_window_title_with_sam_status(hwnd, state, true);
                                    
                                    // 编码成功后立即进入交互分割模式
                                    state.selecting = true;
                                    state.select_start = None;
                                    state.select_end = None;
                                    println!("🖱️ SAM交互分割：请在图像上按下并拖拽以框选目标，释放鼠标完成框选");
                                }
                                Err(e) => {
                                    println!("❌ SAM编码失败: {}", e);
                                    let error_msg = format!("SAM编码失败：\n{}", e);
                                    let error_msg_utf16: Vec<u16> = error_msg.encode_utf16().chain(std::iter::once(0)).collect();
                                    let title_utf16: Vec<u16> = "编码失败".encode_utf16().chain(std::iter::once(0)).collect();
                                    MessageBoxW(hwnd, error_msg_utf16.as_ptr(), title_utf16.as_ptr(), MB_OK | MB_ICONERROR);
                                }
                            }
                        }
                    }
                    Err(e) => {
                        println!("❌ SAM处理器初始化失败: {}", e);
                        let error_msg = format!("SAM处理器初始化失败：\n{}\n\n请确保ONNX模型文件存在", e);
                        let error_msg_utf16: Vec<u16> = error_msg.encode_utf16().chain(std::iter::once(0)).collect();
                        let title_utf16: Vec<u16> = "初始化失败".encode_utf16().chain(std::iter::once(0)).collect();
                        MessageBoxW(hwnd, error_msg_utf16.as_ptr(), title_utf16.as_ptr(), MB_OK | MB_ICONERROR);
                    }
                }
            } else {
                let msg = "没有可用的图像，请先加载图像";
                let msg_utf16: Vec<u16> = msg.encode_utf16().chain(std::iter::once(0)).collect();
                let title_utf16: Vec<u16> = "提示".encode_utf16().chain(std::iter::once(0)).collect();
                MessageBoxW(hwnd, msg_utf16.as_ptr(), title_utf16.as_ptr(), MB_OK | MB_ICONINFORMATION);
            }
        }
        
        MENU_SAM_EXTRACT_MASK => {
            if state.is_grayscale {
                let msg = "SAM分割功能不支持灰度图或黑白图，请使用彩色图像或点击'恢复原图'恢复彩色图像";
                let msg_utf16: Vec<u16> = msg.encode_utf16().chain(std::iter::once(0)).collect();
                let title_utf16: Vec<u16> = "SAM分割限制".encode_utf16().chain(std::iter::once(0)).collect();
                MessageBoxW(hwnd, msg_utf16.as_ptr(), title_utf16.as_ptr(), MB_OK | MB_ICONWARNING);
                return;
            }
            
            if !state.has_mask() {
                let msg = "没有可用的mask，请先进行SAM分割！\n\n使用\"SAM交互分割抠像\"功能在图像上拖拽框选目标，生成mask后再使用此功能。";
                let msg_utf16: Vec<u16> = msg.encode_utf16().chain(std::iter::once(0)).collect();
                let title_utf16: Vec<u16> = "提示".encode_utf16().chain(std::iter::once(0)).collect();
                MessageBoxW(hwnd, msg_utf16.as_ptr(), title_utf16.as_ptr(), MB_OK | MB_ICONINFORMATION);
                return;
            }
            
            match extract_mask_and_display(state) {
                Ok(()) => {
                    println!("✅ 抠图并显示完成");
                    InvalidateRect(hwnd, null_mut(), 1);
                    
                    let msg = "抠图完成！\n\n当前显示的是从原图中裁剪出的mask区域。\n黑色区域表示非mask部分，彩色区域是mask内的原图像素。\n\n您可以使用\"保存文件\"功能保存这个抠图结果。\n\n注意：SAM模型已在分割完成后重置，下次分割时需要重新编码图像。";
                    let msg_utf16: Vec<u16> = msg.encode_utf16().chain(std::iter::once(0)).collect();
                    let title_utf16: Vec<u16> = "抠图成功".encode_utf16().chain(std::iter::once(0)).collect();
                    MessageBoxW(hwnd, msg_utf16.as_ptr(), title_utf16.as_ptr(), MB_OK | MB_ICONINFORMATION);
                }
                Err(e) => {
                    println!("❌ 抠图失败: {}", e);
                    let error_msg = format!("抠图处理失败：\n{}", e);
                    let error_msg_utf16: Vec<u16> = error_msg.encode_utf16().chain(std::iter::once(0)).collect();
                    let title_utf16: Vec<u16> = "抠图失败".encode_utf16().chain(std::iter::once(0)).collect();
                    MessageBoxW(hwnd, error_msg_utf16.as_ptr(), title_utf16.as_ptr(), MB_OK | MB_ICONERROR);
                }
            }
        }
        
        // 新增：选框工具菜单处理
        MENU_SELECTION_TOOL => {
            match start_selection_tool(state) {
                Ok(()) => {
                    println!("✅ 选框工具已激活");
                    InvalidateRect(hwnd, null_mut(), 1);
                }
                Err(e) => {
                    println!("❌ 启动选框工具失败: {}", e);
                    let error_msg = format!("启动选框工具失败：\n{}", e);
                    let error_msg_utf16: Vec<u16> = error_msg.encode_utf16().chain(std::iter::once(0)).collect();
                    let title_utf16: Vec<u16> = "选框工具失败".encode_utf16().chain(std::iter::once(0)).collect();
                    MessageBoxW(hwnd, error_msg_utf16.as_ptr(), title_utf16.as_ptr(), MB_OK | MB_ICONERROR);
                }
            }
        }
        
        // 新增：裁剪选框区域菜单处理
        MENU_CROP_SELECTION => {
            match crop_selection_area(state) {
                Ok(()) => {
                    println!("✅ 选框区域裁剪完成");
                    InvalidateRect(hwnd, null_mut(), 1);
                }
                Err(e) => {
                    println!("❌ 选框区域裁剪失败: {}", e);
                    let error_msg = format!("选框区域裁剪失败：\n{}", e);
                    let error_msg_utf16: Vec<u16> = error_msg.encode_utf16().chain(std::iter::once(0)).collect();
                    let title_utf16: Vec<u16> = "裁剪失败".encode_utf16().chain(std::iter::once(0)).collect();
                    MessageBoxW(hwnd, error_msg_utf16.as_ptr(), title_utf16.as_ptr(), MB_OK | MB_ICONERROR);
                }
            }
        }
        
        // 新增：涂抹选框区域菜单处理
        MENU_BLUR_SELECTION => {
            match blur_selection_area(state) {
                Ok(()) => {
                    println!("✅ 选框区域涂抹完成");
                    InvalidateRect(hwnd, null_mut(), 1);
                }
                Err(e) => {
                    println!("❌ 选框区域涂抹失败: {}", e);
                    let error_msg = format!("选框区域涂抹失败：\n{}", e);
                    let error_msg_utf16: Vec<u16> = error_msg.encode_utf16().chain(std::iter::once(0)).collect();
                    let title_utf16: Vec<u16> = "涂抹失败".encode_utf16().chain(std::iter::once(0)).collect();
                    MessageBoxW(hwnd, error_msg_utf16.as_ptr(), title_utf16.as_ptr(), MB_OK | MB_ICONERROR);
                }
            }
        }
        
        // 新增：批量缩图菜单处理
        MENU_BATCH_RESIZE => {
            show_batch_resize_dialog(hwnd);
        }
        
        _ => {}
    }
}

// 窗口过程函数
unsafe extern "system" fn window_proc(
    hwnd: HWND,
    msg: UINT,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    match msg {
        WM_DESTROY => {
            PostQuitMessage(0);
            0
        }
        WM_COMMAND => {
            let command_id = (wparam & 0xFFFF) as u32;
            handle_menu_command(hwnd, command_id);
            0
        }
        // 添加拖拽文件处理
        WM_DROPFILES => {
            handle_drop_files(hwnd, wparam as HDROP);
            0
        }
		WM_ERASEBKGND => {
			// 我们自己进行背景清除与双缓冲绘制，阻止系统擦除背景以减少闪烁
			1
		}
        WM_KEYDOWN => {
            let key_code = (wparam & 0xFFFF) as u32;
            if key_code == 0x1B { // VK_ESCAPE
                let state = IMAGE_STATE.as_mut().unwrap();
                if state.manual_perspective_mode {
                    println!("🔄 用户按ESC键，退出手动透视变换模式");
                    state.manual_perspective_mode = false;
                    state.manual_perspective_points.clear();
                    InvalidateRect(hwnd, null_mut(), 1);
                    
                    let msg = "已退出手动透视变换模式";
                    let msg_utf16: Vec<u16> = msg.encode_utf16().chain(std::iter::once(0)).collect();
                    let title_utf16: Vec<u16> = "操作取消".encode_utf16().chain(std::iter::once(0)).collect();
                    MessageBoxW(hwnd, msg_utf16.as_ptr(), title_utf16.as_ptr(), MB_OK | MB_ICONINFORMATION);
                }
            }
            0
        }
        WM_LBUTTONDOWN => {
            let state = IMAGE_STATE.as_mut().unwrap();
            if state.has_image() {
                let mx = (lparam & 0xFFFF) as u16 as i16 as i32;
                let my = ((lparam >> 16) & 0xFFFF) as u16 as i16 as i32;
                
                // 检查是否点击了导航箭头
                if state.mouse_in_left_arrow {
                    if let Err(e) = state.load_previous_image() {
                        println!("❌ 加载上一张图片失败: {}", e);
                    }
                    InvalidateRect(hwnd, null_mut(), 0);
                    return 0;
                } else if state.mouse_in_right_arrow {
                    if let Err(e) = state.load_next_image() {
                        println!("❌ 加载下一张图片失败: {}", e);
                    }
                    InvalidateRect(hwnd, null_mut(), 0);
                    return 0;
                }
                
                // 获取窗口尺寸
                // 检查是否可以平移（不在其他模式）
                if !state.selecting && !state.selection_mode && !state.straighten_mode && !state.manual_perspective_mode {
                    if state.can_pan() {
                        // 开始平移模式
                        state.start_pan(mx, my);
                        state.set_hand_cursor(hwnd);
                        InvalidateRect(hwnd, null_mut(), 1);
                        return 0;
                    }
                }
                
                if let Some((ix, iy)) = window_to_image_coords_with_zoom(hwnd, state, mx, my) {
                    if state.selecting {
                        // SAM交互分割模式
                        state.select_start = Some((ix, iy));
                        state.select_end = Some((ix, iy));
                    } else if state.selection_mode {
                        // 选框工具模式
                        state.selection_start = Some((ix, iy));
                        state.selection_end = Some((ix, iy));
                    } else if state.straighten_mode {
                        // 拉直工具模式
                        state.straighten_start = Some((ix, iy));
                        state.straighten_end = Some((ix, iy));
                    } else if state.manual_perspective_mode {
                        // 手动透视变换模式
                        state.manual_perspective_points.push((ix, iy));
                        println!("✅ 手动透视变换：已选择第 {} 个角点 ({}, {})", 
                            state.manual_perspective_points.len(), ix, iy);
                        
                        // 如果已经选择了4个点，执行透视变换
                        if state.manual_perspective_points.len() == 4 {
                            if let Some(ref current) = state.current_mat.clone() {
                                // 将整数坐标转换为浮点坐标
                                let points: [core::Point2f; 4] = [
                                    core::Point2f::new(state.manual_perspective_points[0].0 as f32, state.manual_perspective_points[0].1 as f32),
                                    core::Point2f::new(state.manual_perspective_points[1].0 as f32, state.manual_perspective_points[1].1 as f32),
                                    core::Point2f::new(state.manual_perspective_points[2].0 as f32, state.manual_perspective_points[2].1 as f32),
                                    core::Point2f::new(state.manual_perspective_points[3].0 as f32, state.manual_perspective_points[3].1 as f32),
                                ];
                                
                                match cv_ops::perspective_transform_from_points(&current, points) {
                                    Ok(transformed) => {
                                        if let Ok(()) = state.update_display_from_mat(&transformed) {
                                            println!("✅ 手动透视变换完成");
                                            // 退出手动透视变换模式
                                            state.manual_perspective_mode = false;
                                            state.manual_perspective_points.clear();
                                            InvalidateRect(hwnd, null_mut(), 1);
                                            
                                            let success_msg = "手动透视变换完成！\n\n程序已根据您选择的四个角点完成了透视变换。\n当前显示的是变换后的图像。";
                                            let success_msg_utf16: Vec<u16> = success_msg.encode_utf16().chain(std::iter::once(0)).collect();
                                            let success_title_utf16: Vec<u16> = "透视变换成功".encode_utf16().chain(std::iter::once(0)).collect();
                                            MessageBoxW(hwnd, success_msg_utf16.as_ptr(), success_title_utf16.as_ptr(), MB_OK | MB_ICONINFORMATION);
                                        }
                                    }
                                    Err(e) => {
                                        println!("❌ 手动透视变换失败: {}", e);
                                        let error_msg = format!("透视变换失败：\n{}", e);
                                        let error_msg_utf16: Vec<u16> = error_msg.encode_utf16().chain(std::iter::once(0)).collect();
                                        let error_title_utf16: Vec<u16> = "透视变换失败".encode_utf16().chain(std::iter::once(0)).collect();
                                        MessageBoxW(hwnd, error_msg_utf16.as_ptr(), error_title_utf16.as_ptr(), MB_OK | MB_ICONERROR);
                                        
                                        // 退出手动透视变换模式
                                        state.manual_perspective_mode = false;
                                        state.manual_perspective_points.clear();
                                    }
                                }
                            }
                        }
                    }
                    InvalidateRect(hwnd, null_mut(), 1);
                }
            }
            0
        }
        WM_MOUSEMOVE => {
            let state = IMAGE_STATE.as_mut().unwrap();
            if state.has_image() {
                let mx = (lparam & 0xFFFF) as u16 as i16 as i32;
                let my = ((lparam >> 16) & 0xFFFF) as u16 as i16 as i32;
                
                // 获取窗口尺寸用于更新导航箭头
                let mut rect: RECT = std::mem::zeroed();
                GetClientRect(hwnd, &mut rect);
                let window_width = rect.right - rect.left;
                let window_height = rect.bottom - rect.top;
                
                // 更新导航箭头状态
                let old_show_arrows = state.show_nav_arrows;
                state.update_nav_arrows(mx, my, window_width, window_height);
                
                // 如果箭头显示状态改变，触发重绘
                if old_show_arrows != state.show_nav_arrows {
                    InvalidateRect(hwnd, null_mut(), 0);
                }
                
                // 更新光标（只在没有按下鼠标按钮时）
                if (wparam & MK_LBUTTON as usize) == 0 && !state.pan_mode {
                    // 检查是否可以平移（不在其他模式）
                    if !state.selecting && !state.selection_mode && !state.straighten_mode && !state.manual_perspective_mode {
                        if state.can_pan() {
                            state.set_hand_cursor(hwnd);
                        } else {
                            state.set_arrow_cursor(hwnd);
                        }
                    } else {
                        state.set_arrow_cursor(hwnd);
                    }
                }
                
                if (wparam & MK_LBUTTON as usize) != 0 {
                    // 检查是否在平移模式
                    if state.pan_mode {
                        state.update_pan(mx, my);
                        InvalidateRect(hwnd, null_mut(), 0);
                        return 0;
                    }
                    
                    if let Some((ix, iy)) = window_to_image_coords_with_zoom(hwnd, state, mx, my) {
                        if state.selecting {
                            // SAM交互分割模式
                            state.select_end = Some((ix, iy));
                        } else if state.selection_mode {
                            // 选框工具模式
                            state.selection_end = Some((ix, iy));
                        } else if state.straighten_mode {
                            // 拉直工具模式
                            state.straighten_end = Some((ix, iy));
                        }
                        InvalidateRect(hwnd, null_mut(), 0);
                    }
                }
            }
            0
        }
        WM_LBUTTONUP => {
            let state = IMAGE_STATE.as_mut().unwrap();
            if state.has_image() {
                let mx = (lparam & 0xFFFF) as u16 as i16 as i32;
                let my = ((lparam >> 16) & 0xFFFF) as u16 as i16 as i32;
                
                // 检查是否在平移模式
                if state.pan_mode {
                    state.end_pan();
                    state.set_arrow_cursor(hwnd);
                    InvalidateRect(hwnd, null_mut(), 1);
                    return 0;
                }
                
                if let Some((ix, iy)) = window_to_image_coords_with_zoom(hwnd, state, mx, my) {
                    if state.selecting {
                        // SAM交互分割模式
                        state.select_end = Some((ix, iy));
                        state.selecting = false;
                        if let (Some((x1,y1)), Some((x2,y2))) = (state.select_start, state.select_end) {
                            if let Some(ref sam) = state.sam_processor {
                                if sam.is_encoded() {
                                    let (fx1, fy1) = (x1.min(x2) as f32, y1.min(y2) as f32);
                                    let (fx2, fy2) = (x1.max(x2) as f32, y1.max(y2) as f32);
                                    match sam.decode_prompt(fx1, fy1, fx2, fy2) {
                                        Ok((mask, mw, mh)) => {
                                            // 保存mask到状态中
                                            state.set_current_mask(mask.clone(), mw as i32, mh as i32);
                                            
                                            if let Err(e) = overlay_mask_on_current(state, &mask, mw as i32, mh as i32) {
                                                println!("❌ 叠加分割结果失败: {}", e);
                                            } else {
                                                InvalidateRect(hwnd, null_mut(), 1);
                                            }
                                            
                                            // SAM分割成功后立即重置处理器状态，为下次分割做准备
                                            if let Some(ref mut sam) = state.sam_processor {
                                                sam.reset();
                                                println!("🔄 SAM处理器已重置，下次分割时需要重新编码图像");
                                            }
                                        }
                                        Err(e) => {
                                            println!("❌ SAM分割失败: {}", e);
                                            let msg = format!("SAM分割失败：\n{}", e);
                                            let msg_utf16: Vec<u16> = msg.encode_utf16().chain(std::iter::once(0)).collect();
                                            let title_utf16: Vec<u16> = "分割失败".encode_utf16().chain(std::iter::once(0)).collect();
                                            MessageBoxW(hwnd, msg_utf16.as_ptr(), title_utf16.as_ptr(), MB_OK | MB_ICONERROR);
                                        }
                                    }
                                }
                            }
                        }
                    } else if state.selection_mode {
                        // 选框工具模式
                        state.selection_end = Some((ix, iy));
                        // 选框工具模式下不自动退出，保持选框显示
                        println!("✅ 选框完成: ({},{}) -> ({},{})", 
                            state.selection_start.unwrap().0, state.selection_start.unwrap().1,
                            ix, iy);
                    } else if state.straighten_mode {
                        // 拉直工具模式
                        state.straighten_end = Some((ix, iy));
                        // 拉直模式下不自动退出，保持拉直线显示
                        if let (Some((x1,y1)), Some((x2,y2))) = (state.straighten_start, state.straighten_end) {
                            println!("✅ 拉直线完成: ({},{}) -> ({},{})", x1, y1, x2, y2);
                        }
                    }
                }
                InvalidateRect(hwnd, null_mut(), 1);
            }
            0
        }
        WM_SIZE => {
            // 窗口大小改变时重绘
            InvalidateRect(hwnd, null_mut(), 1);
            0
        }
        WM_MOUSEWHEEL => {
            let state = IMAGE_STATE.as_mut().unwrap();
            if state.has_image() {
                // 获取滚轮滚动方向和鼠标位置
                let wheel_delta = ((wparam >> 16) & 0xFFFF) as u16 as i16;
                let mouse_x = (lparam & 0xFFFF) as u16 as i16 as i32;
                let mouse_y = ((lparam >> 16) & 0xFFFF) as u16 as i16 as i32;
                
                // 获取窗口客户区大小
                let mut rect: RECT = std::mem::zeroed();
                GetClientRect(hwnd, &mut rect);
                let window_width = rect.right - rect.left;
                let window_height = rect.bottom - rect.top;
                
                // 将屏幕坐标转换为窗口客户区坐标
                let mut point = POINT { x: mouse_x, y: mouse_y };
                ScreenToClient(hwnd, &mut point);
                
                // 应用缩放
                let zoom_delta = if wheel_delta > 0 { 1.0 } else { -1.0 };
                state.apply_zoom(zoom_delta, point.x as f32, point.y as f32, window_width, window_height);
                
                // 重绘窗口
                InvalidateRect(hwnd, null_mut(), 1);
                
                println!("🔍 缩放: {:.2}x, 偏移: ({:.1}, {:.1})", 
                    state.zoom_factor, state.zoom_offset_x, state.zoom_offset_y);
            }
            0
        }
        WM_PAINT => {
            let mut ps: PAINTSTRUCT = std::mem::zeroed();
            let hdc = BeginPaint(hwnd, &mut ps);
            
            // 获取窗口客户区大小
            let mut rect: RECT = std::mem::zeroed();
            GetClientRect(hwnd, &mut rect);
            let window_width = rect.right - rect.left;
            let window_height = rect.bottom - rect.top;
            
            // 双缓冲：创建内存DC和位图，所有绘制先在内存DC完成
            let mem_dc = CreateCompatibleDC(hdc);
            let mem_bmp = CreateCompatibleBitmap(hdc, window_width, window_height);
            let old_bmp = SelectObject(mem_dc, mem_bmp as HGDIOBJ);

            // 背景清为黑色
            let black_brush = CreateSolidBrush(RGB(0, 0, 0));
            FillRect(mem_dc, &rect, black_brush);
            DeleteObject(black_brush as *mut _);

            // 绘制图片
            if let Some(ref state) = IMAGE_STATE {
                if state.has_image() && !state.display_pixels.is_empty() {
                    let (x, y, scaled_width, scaled_height) = calculate_scaled_size_with_zoom(
                        state,
                        window_width,
                        window_height,
                    );
                    
                    // 设置高质量拉伸模式 - HALFTONE提供最佳质量
                    SetStretchBltMode(mem_dc, HALFTONE);
                    
                    // 设置画刷原点以改善抖动效果
                    SetBrushOrgEx(mem_dc, 0, 0, std::ptr::null_mut());
                    
                    // 创建位图信息结构
                    let bmi = BITMAPINFO {
                        bmiHeader: BITMAPINFOHEADER {
                            biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
                            biWidth: state.current_width,
                            biHeight: -state.current_height, // 负值表示从上到下
                            biPlanes: 1,
                            biBitCount: 32, // BGRA格式
                            biCompression: BI_RGB,
                            biSizeImage: 0,
                            biXPelsPerMeter: 0,
                            biYPelsPerMeter: 0,
                            biClrUsed: 0,
                            biClrImportant: 0,
                        },
                        bmiColors: [RGBQUAD {
                            rgbBlue: 0,
                            rgbGreen: 0,
                            rgbRed: 0,
                            rgbReserved: 0,
                        }],
                    };
                    
                    // 使用StretchDIBits进行高质量图片绘制
                    StretchDIBits(
                        mem_dc,
                        x,                               // 目标x坐标
                        y,                               // 目标y坐标
                        scaled_width,                    // 目标宽度
                        scaled_height,                   // 目标高度
                        0,                               // 源x坐标
                        0,                               // 源y坐标
                        state.current_width,             // 源宽度
                        state.current_height,            // 源高度
                        state.display_pixels.as_ptr() as *const _,
                        &bmi,
                        DIB_RGB_COLORS,
                        SRCCOPY,
                    );
                    
                    // 正在框选时叠加绘制矩形
                    if state.selecting {
                        if let (Some((sx, sy)), Some((ex, ey))) = (state.select_start, state.select_end) {
                            // 将图像坐标转换回窗口坐标用于绘制
                            let to_win = |ix: i32, iy: i32| -> (i32,i32) {
                                let wx = x + (ix * scaled_width) / state.current_width;
                                let wy = y + (iy * scaled_height) / state.current_height;
                                (wx, wy)
                            };
                            let (wx1, wy1) = to_win(sx, sy);
                            let (wx2, wy2) = to_win(ex, ey);
                            let left = wx1.min(wx2);
                            let top = wy1.min(wy2);
                            let right = wx1.max(wx2);
                            let bottom = wy1.max(wy2);
                            let pen = CreatePen(PS_SOLID as i32, 2, RGB(0,255,0));
                            let old_pen = SelectObject(mem_dc, pen as HGDIOBJ);
                            let hollow_brush = GetStockObject(HOLLOW_BRUSH as i32) as HBRUSH;
                            let old_brush = SelectObject(mem_dc, hollow_brush as HGDIOBJ);
                            Rectangle(mem_dc, left, top, right, bottom);
                            SelectObject(mem_dc, old_pen);
                            SelectObject(mem_dc, old_brush);
                            DeleteObject(pen as HGDIOBJ);
                        }
                    }
                    
                    // 新增：选框工具选框绘制
                    if state.selection_mode && state.has_selection() {
                        if let (Some((sx, sy)), Some((ex, ey))) = (state.selection_start, state.selection_end) {
                            // 将图像坐标转换回窗口坐标用于绘制
                            let to_win = |ix: i32, iy: i32| -> (i32,i32) {
                                let wx = x + (ix * scaled_width) / state.current_width;
                                let wy = y + (iy * scaled_height) / state.current_height;
                                (wx, wy)
                            };
                            let (wx1, wy1) = to_win(sx, sy);
                            let (wx2, wy2) = to_win(ex, ey);
                            let left = wx1.min(wx2);
                            let top = wy1.min(wy2);
                            let right = wx1.max(wx2);
                            let bottom = wy1.max(wy2);
                            
                            // 绘制绿色选框（2像素宽）
                            let pen = CreatePen(PS_SOLID as i32, 2, RGB(0,255,0));
                            let old_pen = SelectObject(mem_dc, pen as HGDIOBJ);
                            let hollow_brush = GetStockObject(HOLLOW_BRUSH as i32) as HBRUSH;
                            let old_brush = SelectObject(mem_dc, hollow_brush as HGDIOBJ);
                            Rectangle(mem_dc, left, top, right, bottom);
                            SelectObject(mem_dc, old_pen);
                            SelectObject(mem_dc, old_brush);
                            DeleteObject(pen as HGDIOBJ);
                            
                            // 在选框上方显示尺寸信息
                            let size_text = format!("{} x {}", right - left, bottom - top);
                            let size_text_utf16: Vec<u16> = size_text.encode_utf16().chain(std::iter::once(0)).collect();
                            SetTextColor(mem_dc, RGB(0, 255, 0));
                            SetBkMode(mem_dc, TRANSPARENT as i32);
                            SetBkColor(mem_dc, RGB(0, 0, 0));
                            TextOutW(mem_dc, left, top - 20, size_text_utf16.as_ptr(), size_text.len() as i32);
                        }
                    }
                    
                    // 新增：拉直工具直线绘制
                    if state.straighten_mode {
                        if let (Some((sx, sy)), Some((ex, ey))) = (state.straighten_start, state.straighten_end) {
                            // 将图像坐标转换回窗口坐标用于绘制
                            let to_win = |ix: i32, iy: i32| -> (i32,i32) {
                                let wx = x + (ix * scaled_width) / state.current_width;
                                let wy = y + (iy * scaled_height) / state.current_height;
                                (wx, wy)
                            };
                            let (wx1, wy1) = to_win(sx, sy);
                            let (wx2, wy2) = to_win(ex, ey);
                            
                            // 绘制绿色直线（3像素宽）
                            let pen = CreatePen(PS_SOLID as i32, 3, RGB(0,255,0));
                            let old_pen = SelectObject(mem_dc, pen as HGDIOBJ);
                            
                            MoveToEx(mem_dc, wx1, wy1, null_mut());
                            LineTo(mem_dc, wx2, wy2);
                            
                            SelectObject(mem_dc, old_pen);
                            DeleteObject(pen as HGDIOBJ);
                        }
                    }
                    
                    // 新增：手动透视变换角点绘制
                    if state.manual_perspective_mode && !state.manual_perspective_points.is_empty() {
                        // 将图像坐标转换回窗口坐标用于绘制
                        let to_win = |ix: i32, iy: i32| -> (i32,i32) {
                            let wx = x + (ix * scaled_width) / state.current_width;
                            let wy = y + (iy * scaled_height) / state.current_height;
                            (wx, wy)
                        };
                        
                        // 绘制已选择的角点
                        let colors = [RGB(255, 0, 0), RGB(0, 255, 0), RGB(0, 0, 255), RGB(255, 255, 0)]; // 红、绿、蓝、黄
                        for (i, &(ix, iy)) in state.manual_perspective_points.iter().enumerate() {
                            let (wx, wy) = to_win(ix, iy);
                            
                            // 绘制彩色圆点，不同角点用不同颜色
                            let color = if i < colors.len() { colors[i] } else { RGB(255, 255, 255) };
                            
                            let brush = CreateSolidBrush(color);
                            let old_brush = SelectObject(mem_dc, brush as HGDIOBJ);
                            
                            // 绘制圆点（半径8像素）
                            Ellipse(mem_dc, wx - 8, wy - 8, wx + 8, wy + 8);
                            
                            SelectObject(mem_dc, old_brush);
                            DeleteObject(brush as HGDIOBJ);
                            
                            // 在角点旁边显示序号
                            let num_text = format!("{}", i + 1);
                            let num_text_utf16: Vec<u16> = num_text.encode_utf16().chain(std::iter::once(0)).collect();
                            SetTextColor(mem_dc, color);
                            SetBkMode(mem_dc, TRANSPARENT as i32);
                            SetBkColor(mem_dc, RGB(0, 0, 0));
                            TextOutW(mem_dc, wx + 12, wy - 8, num_text_utf16.as_ptr(), num_text.len() as i32);
                        }
                        
                        // 如果已经选择了4个点，绘制连接线
                        if state.manual_perspective_points.len() == 4 {
                            let pen = CreatePen(PS_SOLID as i32, 2, RGB(255, 255, 0)); // 黄色连接线
                            let old_pen = SelectObject(mem_dc, pen as HGDIOBJ);
                            
                            // 连接四个角点形成四边形
                            for i in 0..4 {
                                let (ix1, iy1) = state.manual_perspective_points[i];
                                let (ix2, iy2) = state.manual_perspective_points[(i + 1) % 4];
                                
                                let (wx1, wy1) = to_win(ix1, iy1);
                                let (wx2, wy2) = to_win(ix2, iy2);
                                
                                MoveToEx(mem_dc, wx1, wy1, null_mut());
                                LineTo(mem_dc, wx2, wy2);
                            }
                            
                            SelectObject(mem_dc, old_pen);
                            DeleteObject(pen as HGDIOBJ);
                        }
                    }
                    
                    // 更新窗口标题显示图像信息和SAM状态
                    let sam_loaded = state.sam_processor.as_ref().map_or(false, |sam| sam.is_encoded());
                    if state.straighten_mode {
                        update_window_title_with_straighten_mode(hwnd, state);
                    } else {
                        update_window_title_with_sam_status(hwnd, state, sam_loaded);
                    }
                } else {
                    // 显示提示信息
                    let hint_text = "请使用菜单 文件->打开文件 来加载图像";
                    let text_utf16: Vec<u16> = hint_text.encode_utf16().collect();
                    
                    SetTextColor(mem_dc, RGB(255, 255, 255));
                    SetBkMode(mem_dc, TRANSPARENT as i32);
                    
                    DrawTextW(
                        mem_dc,
                        text_utf16.as_ptr(),
                        text_utf16.len() as i32,
                        &mut rect,
                        DT_CENTER | DT_VCENTER | DT_SINGLELINE,
                    );
                }
            }

            // 绘制导航箭头
            if let Some(ref state) = IMAGE_STATE {
                if state.show_nav_arrows && state.has_image() {
                    const ARROW_SIZE: i32 = 40; // 箭头大小
                    const ARROW_AREA_WIDTH: i32 = 100;
                    
                    // 计算箭头位置（垂直居中）
                    let arrow_y = (window_height - ARROW_SIZE) / 2;
                    
                    // 左侧箭头（如果可以向前导航）
                    if state.can_go_previous() {
                        let arrow_x = (ARROW_AREA_WIDTH - ARROW_SIZE) / 2;
                        
                        // 绘制圆形背景
                        let bg_color = if state.mouse_in_left_arrow { RGB(80, 80, 80) } else { RGB(50, 50, 50) };
                        let brush = CreateSolidBrush(bg_color);
                        let old_brush = SelectObject(mem_dc, brush as HGDIOBJ);
                        let pen = CreatePen(PS_SOLID as i32, 1, RGB(150, 150, 150));
                        let old_pen = SelectObject(mem_dc, pen as HGDIOBJ);
                        
                        Ellipse(mem_dc, arrow_x, arrow_y, arrow_x + ARROW_SIZE, arrow_y + ARROW_SIZE);
                        
                        SelectObject(mem_dc, old_brush);
                        SelectObject(mem_dc, old_pen);
                        DeleteObject(brush as HGDIOBJ);
                        DeleteObject(pen as HGDIOBJ);
                        
                        // 绘制左箭头 <
                        let white_pen = CreatePen(PS_SOLID as i32, 3, RGB(255, 255, 255));
                        let old_pen = SelectObject(mem_dc, white_pen as HGDIOBJ);
                        
                        let cx = arrow_x + ARROW_SIZE / 2;
                        let cy = arrow_y + ARROW_SIZE / 2;
                        let offset = 10;
                        
                        MoveToEx(mem_dc, cx + offset / 2, cy - offset, null_mut());
                        LineTo(mem_dc, cx - offset / 2, cy);
                        LineTo(mem_dc, cx + offset / 2, cy + offset);
                        
                        SelectObject(mem_dc, old_pen);
                        DeleteObject(white_pen as HGDIOBJ);
                    }
                    
                    // 右侧箭头（如果可以向后导航）
                    if state.can_go_next() {
                        let arrow_x = window_width - ARROW_AREA_WIDTH + (ARROW_AREA_WIDTH - ARROW_SIZE) / 2;
                        
                        // 绘制圆形背景
                        let bg_color = if state.mouse_in_right_arrow { RGB(80, 80, 80) } else { RGB(50, 50, 50) };
                        let brush = CreateSolidBrush(bg_color);
                        let old_brush = SelectObject(mem_dc, brush as HGDIOBJ);
                        let pen = CreatePen(PS_SOLID as i32, 1, RGB(150, 150, 150));
                        let old_pen = SelectObject(mem_dc, pen as HGDIOBJ);
                        
                        Ellipse(mem_dc, arrow_x, arrow_y, arrow_x + ARROW_SIZE, arrow_y + ARROW_SIZE);
                        
                        SelectObject(mem_dc, old_brush);
                        SelectObject(mem_dc, old_pen);
                        DeleteObject(brush as HGDIOBJ);
                        DeleteObject(pen as HGDIOBJ);
                        
                        // 绘制右箭头 >
                        let white_pen = CreatePen(PS_SOLID as i32, 3, RGB(255, 255, 255));
                        let old_pen = SelectObject(mem_dc, white_pen as HGDIOBJ);
                        
                        let cx = arrow_x + ARROW_SIZE / 2;
                        let cy = arrow_y + ARROW_SIZE / 2;
                        let offset = 10;
                        
                        MoveToEx(mem_dc, cx - offset / 2, cy - offset, null_mut());
                        LineTo(mem_dc, cx + offset / 2, cy);
                        LineTo(mem_dc, cx - offset / 2, cy + offset);
                        
                        SelectObject(mem_dc, old_pen);
                        DeleteObject(white_pen as HGDIOBJ);
                    }
                }
            }

            // 将内存DC内容拷贝到前台，一次提交减少闪烁
            BitBlt(hdc, 0, 0, window_width, window_height, mem_dc, 0, 0, SRCCOPY);

            // 释放内存对象
            SelectObject(mem_dc, old_bmp);
            DeleteObject(mem_bmp as HGDIOBJ);
            DeleteDC(mem_dc);
            
            EndPaint(hwnd, &ps);
            0
        }
        WM_CLOSE => {
            println!("[dialog_proc] 收到 WM_CLOSE，销毁窗口");
            DestroyWindow(hwnd);
            0
        }
        WM_DESTROY => {
            println!("[dialog_proc] 收到 WM_DESTROY");
            0
        }
        _ => DefWindowProcW(hwnd, msg, wparam, lparam),
    }
}

fn setup_dpi_awareness() {
    unsafe {
        // 尝试使用Per-Monitor V2 DPI感知
        if SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2) == 0 {
            // 如果失败，尝试使用Per-Monitor DPI感知
            if SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE) != 0 {
                // 如果仍然失败，使用系统DPI感知
                SetProcessDPIAware();
            }
        }
    }
}

fn main() {
    println!("=== 图像处理查看器 ===");
    println!("基于 OpenCV + Win32 GDI");
    
    // 设置DPI感知
    setup_dpi_awareness();
    
    // 创建调试输出目录
    if let Err(e) = std::fs::create_dir_all("./debug_out") {
        println!("警告：无法创建调试输出目录: {}", e);
    }
    
    unsafe {
        // 初始化全局状态
        IMAGE_STATE = Some(ImageState::new());
        
        let instance = GetModuleHandleW(null_mut());
        let window_class = "ImageViewerOpenCV";
        let window_class_utf16: Vec<u16> = window_class.encode_utf16().chain(std::iter::once(0)).collect();

        // 注册窗口类
        let wc = WNDCLASSW {
            style: CS_HREDRAW | CS_VREDRAW,
            lpfnWndProc: Some(window_proc),
            hInstance: instance,
            lpszClassName: window_class_utf16.as_ptr(),
            hCursor: LoadCursorW(null_mut(), IDC_ARROW),
            hbrBackground: (COLOR_WINDOW + 1) as HBRUSH,
            cbClsExtra: 0,
            cbWndExtra: 0,
            hIcon: null_mut(),
            lpszMenuName: null_mut(),
        };

        if RegisterClassW(&wc) == 0 {
            panic!("Failed to register window class");
        }

        // 创建菜单
        let menu = create_menu();
        
        // 创建窗口
        let window_title = "图像查看器";
        let window_title_utf16: Vec<u16> = window_title.encode_utf16().chain(std::iter::once(0)).collect();

        let hwnd = CreateWindowExW(
            0,
            window_class_utf16.as_ptr(),
            window_title_utf16.as_ptr(),
            WS_OVERLAPPEDWINDOW,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            1200,
            800,
            null_mut(),
            menu,
            instance,
            null_mut(),
        );

        if hwnd.is_null() {
            panic!("Failed to create window");
        }
        
        // 保存主窗口句柄到全局变量
        MAIN_HWND = Some(hwnd);

        // 启用拖拽文件接受
        DragAcceptFiles(hwnd, 1); // 1表示接受拖拽文件

        // 显示和更新窗口
        ShowWindow(hwnd, SW_SHOW);
        UpdateWindow(hwnd);

        println!("✅ 程序启动成功！");
        println!("📁 使用菜单 文件->打开文件 来加载图像");
        println!("🖱️ 或者直接拖拽图像文件到窗口来加载图像");
        println!("🎨 支持的功能:");
        println!("   - 灰度转换");
        println!("   - 高斯模糊");
        println!("   - Canny边缘检测");
        println!("   - 自适应阈值");
        println!("   - 形态学闭运算");
        println!("   - 膨胀操作");
        println!("   - 轮廓检测");
        println!("   - 透视变换(简化色卡检测)");
        println!("   - 模板匹配定位色卡(基于特征匹配SIFT/AKAZE/ORB)");
        println!("   - 读取色卡色块颜色(网格法提取24个色块RGB值)");
        println!("🐛 调试图像会保存到 ./debug_out/ 目录");

        // 消息循环
        let mut msg: MSG = std::mem::zeroed();
        while GetMessageW(&mut msg, null_mut(), 0, 0) > 0 {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
    }
}

// 新增：选框工具功能
fn start_selection_tool(state: &mut ImageState) -> Result<()> {
    if !state.has_image() {
        return Err(anyhow::anyhow!("没有可用的图像，请先加载图像"));
    }
    
    state.start_selection_mode();
    Ok(())
}

// 新增：裁剪选框区域功能
fn crop_selection_area(state: &mut ImageState) -> Result<()> {
    use opencv::{imgproc, core as cvcore};
    
    if !state.has_selection() {
        return Err(anyhow::anyhow!("没有可用的选框，请先使用选框工具框选区域"));
    }
    
    // 确保加载完整分辨率（裁剪需要保持最佳质量）
    state.ensure_full_resolution()?;
    
    let current_img = match &state.current_mat {
        Some(img) => img,
        None => {
            return Err(anyhow::anyhow!("没有可用的图像"));
        }
    };
    
    let (left, top, right, bottom) = state.get_selection_rect().unwrap();
    
    println!("🔍 开始裁剪选框区域...");
    println!("  选框区域: ({},{}) -> ({},{})", left, top, right, bottom);
    println!("  选框尺寸: {}x{}", right - left, bottom - top);
    println!("  原图尺寸: {}x{}", current_img.cols(), current_img.rows());
    
    // 确保选框坐标在图像范围内
    let left = left.max(0).min(current_img.cols() - 1);
    let top = top.max(0).min(current_img.rows() - 1);
    let right = right.max(left + 1).min(current_img.cols());
    let bottom = bottom.max(top + 1).min(current_img.rows());
    
    println!("  调整后选框区域: ({},{}) -> ({},{})", left, top, right, bottom);
    println!("  调整后选框尺寸: {}x{}", right - left, bottom - top);
    
    // 创建ROI矩形
    let roi = cvcore::Rect::new(left, top, right - left, bottom - top);
    
    // 裁剪图像
    let cropped_img = match Mat::roi(current_img, roi) {
        Ok(roi_mat) => roi_mat.try_clone()?,
        Err(_) => {
            return Err(anyhow::anyhow!("裁剪图像失败，ROI超出边界"));
        }
    };
    
    println!("  裁剪后图像尺寸: {}x{}", cropped_img.cols(), cropped_img.rows());
    println!("  裁剪后图像通道数: {}", cropped_img.channels());
    
    // 更新显示
    state.update_display_from_mat(&cropped_img)?;
    
    // 退出选框模式
    state.end_selection_mode();
    
    println!("✅ 选框区域裁剪完成");
    Ok(())
}

// 新增：涂抹选框区域功能
fn blur_selection_area(state: &mut ImageState) -> Result<()> {
    use opencv::{imgproc, core as cvcore};
    
    if !state.has_selection() {
        return Err(anyhow::anyhow!("没有可用的选框，请先使用选框工具框选区域"));
    }
    
    let current_img = match &state.current_mat {
        Some(img) => img,
        None => {
            return Err(anyhow::anyhow!("没有可用的图像"));
        }
    };
    
    let (left, top, right, bottom) = state.get_selection_rect().unwrap();
    
    println!("🔍 开始涂抹选框区域...");
    println!("  选框区域: ({},{}) -> ({},{})", left, top, right, bottom);
    println!("  选框尺寸: {}x{}", right - left, bottom - top);
    println!("  原图尺寸: {}x{}", current_img.cols(), current_img.rows());
    
    // 确保选框坐标在图像范围内
    let left = left.max(0).min(current_img.cols() - 1);
    let top = top.max(0).min(current_img.rows() - 1);
    let right = right.max(left + 1).min(current_img.cols());
    let bottom = bottom.max(top + 1).min(current_img.rows());
    
    println!("  调整后选框区域: ({},{}) -> ({},{})", left, top, right, bottom);
    println!("  调整后选框尺寸: {}x{}", right - left, bottom - top);
    
    // 创建ROI矩形
    let roi = cvcore::Rect::new(left, top, right - left, bottom - top);
    
    // 提取选框区域（用于验证ROI是否有效）
    let _selection_roi = match Mat::roi(current_img, roi) {
        Ok(roi_mat) => roi_mat.try_clone()?,
        Err(_) => {
            return Err(anyhow::anyhow!("提取选框区域失败，ROI超出边界"));
        }
    };
    
    // 创建涂抹后的图像副本
    let mut blurred_img = current_img.try_clone()?;
    
    // 在选框区域应用50%灰色涂抹
    let gray_color = cvcore::Scalar::new(128.0, 128.0, 128.0, 0.0); // BGR格式，50%灰色
    
    // 用50%灰色填充选框区域
    imgproc::rectangle(&mut blurred_img, roi, gray_color, -1, imgproc::LINE_8, 0)?;
    
    // 更新显示
    state.update_display_from_mat(&blurred_img)?;
    
    // 退出选框模式
    state.end_selection_mode();
    
    println!("✅ 选框区域涂抹完成");
    Ok(())
}

// 显示调整图像大小对话框
unsafe fn show_resize_dialog(hwnd: HWND, current_width: i32, current_height: i32) -> Option<ResizeResult> {
    println!("开始创建调整图像大小对话框");
    println!("主窗口句柄: {:?}, 当前图像尺寸: {}x{}", hwnd, current_width, current_height);
    

    
    // 注册对话框窗口类
    let dialog_class = "ResizeDialogClass";
    let dialog_class_utf16: Vec<u16> = dialog_class.encode_utf16().chain(std::iter::once(0)).collect();
    
    let wc = WNDCLASSW {
        style: CS_HREDRAW | CS_VREDRAW,
        lpfnWndProc: Some(dialog_proc),
        cbClsExtra: 0,
        cbWndExtra: 0,
        hInstance: GetModuleHandleW(null_mut()),
        hIcon: null_mut(),
        hCursor: LoadCursorW(null_mut(), IDC_ARROW),
        hbrBackground: (COLOR_WINDOW + 1) as HBRUSH,
        lpszMenuName: null_mut(),
        lpszClassName: dialog_class_utf16.as_ptr(),
    };
    
    if RegisterClassW(&wc) == 0 {
        println!("错误：无法注册对话框窗口类");
        return None;
    }
    
    // 创建模态对话框窗口
    let dialog_hwnd = CreateWindowExW(
        WS_EX_DLGMODALFRAME | WS_EX_TOPMOST,
        dialog_class_utf16.as_ptr(),
        null_mut(),
        WS_POPUP | WS_CAPTION | WS_SYSMENU | WS_VISIBLE | DS_MODALFRAME,
        100, 100, 400, 500, // 增加高度以确保按钮完全可见
        hwnd,
        null_mut(),
        GetModuleHandleW(null_mut()),
        null_mut(),
    );
    
    if dialog_hwnd.is_null() {
        println!("错误：无法创建对话框窗口");
        return None;
    }
    
    println!("对话框窗口创建成功，句柄: {:?}", dialog_hwnd);
    
    // 设置对话框标题
    let title = "重设大小";
    let title_utf16: Vec<u16> = title.encode_utf16().chain(std::iter::once(0)).collect();
    SetWindowTextW(dialog_hwnd, title_utf16.as_ptr());
    println!("对话框标题设置完成");
    
    // 创建控件
    let mut result = ResizeResult {
        new_width: current_width,
        new_height: current_height,
        quality: 80,
        interpolation: INTER_LINEAR, // 新增：插值算法
    };
    println!("ResizeResult 结构体初始化完成");
    
    // 创建宽度输入框标签
    let static_class = "STATIC";
    let static_class_utf16: Vec<u16> = static_class.encode_utf16().chain(std::iter::once(0)).collect();
    let width_label = CreateWindowExW(
        0,
        static_class_utf16.as_ptr(),
        null_mut(),
        WS_VISIBLE | WS_CHILD | SS_LEFT,
        20, 50, 80, 20,
        dialog_hwnd,
        null_mut(),
        GetModuleHandleW(null_mut()),
        null_mut(),
    );
    
    if width_label.is_null() {
        println!("错误：无法创建宽度标签");
        return None;
    }
    
    let width_label_text = "宽度 (px):";
    let width_label_text_utf16: Vec<u16> = width_label_text.encode_utf16().chain(std::iter::once(0)).collect();
    SetWindowTextW(width_label, width_label_text_utf16.as_ptr());
    println!("宽度标签创建成功");
    
    // 创建宽度输入框
    let edit_class = "EDIT";
    let edit_class_utf16: Vec<u16> = edit_class.encode_utf16().chain(std::iter::once(0)).collect();
    let width_edit = CreateWindowExW(
        WS_EX_CLIENTEDGE,
        edit_class_utf16.as_ptr(),
        null_mut(),
        WS_VISIBLE | WS_CHILD | WS_BORDER | ES_NUMBER,
        110, 50, 100, 20,
        dialog_hwnd,
        IDC_WIDTH_EDIT as HMENU,
        GetModuleHandleW(null_mut()),
        null_mut(),
    );
    
    if width_edit.is_null() {
        println!("错误：无法创建宽度输入框");
        return None;
    }
    
    let width_text = format!("{}", current_width);
    let width_text_utf16: Vec<u16> = width_text.encode_utf16().chain(std::iter::once(0)).collect();
    SetWindowTextW(width_edit, width_text_utf16.as_ptr());
    println!("宽度输入框创建成功，初始值: {}", current_width);
    
    // 创建高度输入框标签
    let height_label = CreateWindowExW(
        0,
        static_class_utf16.as_ptr(),
        null_mut(),
        WS_VISIBLE | WS_CHILD | SS_LEFT,
        20, 80, 80, 20,
        dialog_hwnd,
        null_mut(),
        GetModuleHandleW(null_mut()),
        null_mut(),
    );
    
    if height_label.is_null() {
        println!("错误：无法创建高度标签");
        return None;
    }
    
    let height_label_text = "高度 (px):";
    let height_label_text_utf16: Vec<u16> = height_label_text.encode_utf16().chain(std::iter::once(0)).collect();
    SetWindowTextW(height_label, height_label_text_utf16.as_ptr());
    println!("高度标签创建成功");
    
    // 创建高度输入框
    let height_edit = CreateWindowExW(
        WS_EX_CLIENTEDGE,
        edit_class_utf16.as_ptr(),
        null_mut(),
        WS_VISIBLE | WS_CHILD | WS_BORDER | ES_NUMBER,
        110, 80, 100, 20,
        dialog_hwnd,
        IDC_HEIGHT_EDIT as HMENU,
        GetModuleHandleW(null_mut()),
        null_mut(),
    );
    
    if height_edit.is_null() {
        println!("错误：无法创建高度输入框");
        return None;
    }
    
    let height_text = format!("{}", current_height);
    let height_text_utf16: Vec<u16> = height_text.encode_utf16().chain(std::iter::once(0)).collect();
    SetWindowTextW(height_edit, height_text_utf16.as_ptr());
    println!("高度输入框创建成功，初始值: {}", current_height);
    
    // 创建锁定符号标签 (🔒)
    let lock_symbol_label = CreateWindowExW(
        0,
        static_class_utf16.as_ptr(),
        null_mut(),
        WS_VISIBLE | WS_CHILD | SS_LEFT,
        220, 80, 30, 20,
        dialog_hwnd,
        null_mut(),
        GetModuleHandleW(null_mut()),
        null_mut(),
    );
    
    if lock_symbol_label.is_null() {
        println!("错误：无法创建锁定符号标签");
        return None;
    }
    
    let lock_symbol_text = "🔒";
    let lock_symbol_text_utf16: Vec<u16> = lock_symbol_text.encode_utf16().chain(std::iter::once(0)).collect();
    SetWindowTextW(lock_symbol_label, lock_symbol_text_utf16.as_ptr());
    println!("锁定符号标签创建成功");
    
    // 创建质量标签
    let quality_label = CreateWindowExW(
        0,
        static_class_utf16.as_ptr(),
        null_mut(),
        WS_VISIBLE | WS_CHILD | SS_LEFT,
        20, 120, 100, 20,
        dialog_hwnd,
        null_mut(),
        GetModuleHandleW(null_mut()),
        null_mut(),
    );
    
    if quality_label.is_null() {
        println!("错误：无法创建质量标签");
        return None;
    }
    
    let quality_label_text = "质量: 80% (高)";
    let quality_label_text_utf16: Vec<u16> = quality_label_text.encode_utf16().chain(std::iter::once(0)).collect();
    SetWindowTextW(quality_label, quality_label_text_utf16.as_ptr());
    println!("质量标签创建成功");
    
    // 创建质量滑块
    let msctls_trackbar32_class = "msctls_trackbar32";
    let msctls_trackbar32_class_utf16: Vec<u16> = msctls_trackbar32_class.encode_utf16().chain(std::iter::once(0)).collect();
    let quality_trackbar = CreateWindowExW(
        0,
        msctls_trackbar32_class_utf16.as_ptr(),
        null_mut(),
        WS_VISIBLE | WS_CHILD | TBS_HORZ | TBS_AUTOTICKS | WS_TABSTOP,
        20, 150, 200, 30,
        dialog_hwnd,
        IDC_QUALITY_TRACKBAR as HMENU,
        GetModuleHandleW(null_mut()),
        null_mut(),
    );
    
    if quality_trackbar.is_null() {
        println!("错误：无法创建质量滑块");
        return None;
    }
    
    // 设置滑块范围 (1-100)
    SendMessageW(quality_trackbar, TBM_SETRANGE, 1, MAKELONG!(1, 100) as isize);
    SendMessageW(quality_trackbar, TBM_SETPOS, 1, 80);
    
    
    // 强制更新trackbar的显示
    InvalidateRect(quality_trackbar, null_mut(), 1);
    UpdateWindow(quality_trackbar);
    
    println!("质量滑块创建成功，范围: 1-100, 初始值: 80");
    
    // 创建插值算法标签
    let interpolation_label = CreateWindowExW(
        0,
        static_class_utf16.as_ptr(),
        null_mut(),
        WS_VISIBLE | WS_CHILD | SS_LEFT,
        20, 190, 100, 20,
        dialog_hwnd,
        null_mut(),
        GetModuleHandleW(null_mut()),
        null_mut(),
    );
    
    if interpolation_label.is_null() {
        println!("错误：无法创建插值算法标签");
        return None;
    }
    
    let interpolation_label_text = "插值算法:";
    let interpolation_label_text_utf16: Vec<u16> = interpolation_label_text.encode_utf16().chain(std::iter::once(0)).collect();
    SetWindowTextW(interpolation_label, interpolation_label_text_utf16.as_ptr());
    println!("插值算法标签创建成功");
    
    // 创建插值算法选择下拉框
    let combobox_class = "COMBOBOX";
    let combobox_class_utf16: Vec<u16> = combobox_class.encode_utf16().chain(std::iter::once(0)).collect();
    let interpolation_combo = CreateWindowExW(
        0,
        combobox_class_utf16.as_ptr(),
        null_mut(),
        WS_VISIBLE | WS_CHILD | CBS_DROPDOWNLIST | WS_VSCROLL,
        110, 190, 150, 100,
        dialog_hwnd,
        null_mut(),
        GetModuleHandleW(null_mut()),
        null_mut(),
    );
    
    if interpolation_combo.is_null() {
        println!("错误：无法创建插值算法选择下拉框");
        return None;
    }
    
    // 添加插值算法选项
    let nearest_text = "最近邻插值 (最快)";
    let nearest_text_utf16: Vec<u16> = nearest_text.encode_utf16().chain(std::iter::once(0)).collect();
    SendMessageW(interpolation_combo, CB_ADDSTRING, 0, nearest_text_utf16.as_ptr() as LPARAM);
    
    let linear_text = "双线性插值 (推荐)";
    let linear_text_utf16: Vec<u16> = linear_text.encode_utf16().chain(std::iter::once(0)).collect();
    SendMessageW(interpolation_combo, CB_ADDSTRING, 0, linear_text_utf16.as_ptr() as LPARAM);
    
    let cubic_text = "双三次插值 (高质量)";
    let cubic_text_utf16: Vec<u16> = cubic_text.encode_utf16().chain(std::iter::once(0)).collect();
    SendMessageW(interpolation_combo, CB_ADDSTRING, 0, cubic_text_utf16.as_ptr() as LPARAM);
    
    let area_text = "区域插值 (适合缩小)";
    let area_text_utf16: Vec<u16> = area_text.encode_utf16().chain(std::iter::once(0)).collect();
    SendMessageW(interpolation_combo, CB_ADDSTRING, 0, area_text_utf16.as_ptr() as LPARAM);
    
    let lanczos_text = "Lanczos插值 (最高质量)";
    let lanczos_text_utf16: Vec<u16> = lanczos_text.encode_utf16().chain(std::iter::once(0)).collect();
    SendMessageW(interpolation_combo, CB_ADDSTRING, 0, lanczos_text_utf16.as_ptr() as LPARAM);
    
    // 设置默认选择为双线性插值（索引1）
    SendMessageW(interpolation_combo, CB_SETCURSEL, 1, 0);
    println!("插值算法选择下拉框创建成功，默认选择双线性插值");
    
    // 创建文件类型标签
    let filetype_label = CreateWindowExW(
        0,
        static_class_utf16.as_ptr(),
        null_mut(),
        WS_VISIBLE | WS_CHILD | SS_LEFT,
        20, 230, 80, 20, // 调整Y坐标
        dialog_hwnd,
        null_mut(),
        GetModuleHandleW(null_mut()),
        null_mut(),
    );
    
    if filetype_label.is_null() {
        println!("错误：无法创建文件类型标签");
        return None;
    }
    
    let filetype_label_text = "文件类型:";
    let filetype_label_text_utf16: Vec<u16> = filetype_label_text.encode_utf16().chain(std::iter::once(0)).collect();
    SetWindowTextW(filetype_label, filetype_label_text_utf16.as_ptr());
    println!("文件类型标签创建成功");
    
    // 创建文件类型下拉框
    let combobox_class = "COMBOBOX";
    let combobox_class_utf16: Vec<u16> = combobox_class.encode_utf16().chain(std::iter::once(0)).collect();
    let filetype_combo = CreateWindowExW(
        0,
        combobox_class_utf16.as_ptr(),
        null_mut(),
        WS_VISIBLE | WS_CHILD | CBS_DROPDOWNLIST | WS_VSCROLL,
        110, 230, 100, 100, // 调整Y坐标
        dialog_hwnd,
        null_mut(),
        GetModuleHandleW(null_mut()),
        null_mut(),
    );
    
    if filetype_combo.is_null() {
        println!("错误：无法创建文件类型下拉框");
        return None;
    }
    
    // 添加文件类型选项
    let jpg_text = "JPG";
    let jpg_text_utf16: Vec<u16> = jpg_text.encode_utf16().chain(std::iter::once(0)).collect();
    SendMessageW(filetype_combo, CB_ADDSTRING, 0, jpg_text_utf16.as_ptr() as LPARAM);
    SendMessageW(filetype_combo, CB_SETCURSEL, 0, 0);
    println!("文件类型下拉框创建成功，添加了JPG选项");
    
    // 创建当前图像信息标签
    let current_info_label = CreateWindowExW(
        0,
        static_class_utf16.as_ptr(),
        null_mut(),
        WS_VISIBLE | WS_CHILD | SS_LEFT,
        20, 280, 350, 40, // 调整Y坐标
        dialog_hwnd,
        null_mut(),
        GetModuleHandleW(null_mut()),
        null_mut(),
    );
    
    if current_info_label.is_null() {
        println!("错误：无法创建当前图像信息标签");
        return None;
    }
    
    let current_info_text = format!("当前: {} x {} 像素", current_width, current_height);
    let current_info_text_utf16: Vec<u16> = current_info_text.encode_utf16().chain(std::iter::once(0)).collect();
    SetWindowTextW(current_info_label, current_info_text_utf16.as_ptr());
    println!("当前图像信息标签创建成功");
    
    // 创建新图像信息标签
    let new_info_label = CreateWindowExW(
        0,
        static_class_utf16.as_ptr(),
        null_mut(),
        WS_VISIBLE | WS_CHILD | SS_LEFT,
        20, 320, 350, 40, // 调整Y坐标
        dialog_hwnd,
        null_mut(),
        GetModuleHandleW(null_mut()),
        null_mut(),
    );
    
    if new_info_label.is_null() {
        println!("错误：无法创建新图像信息标签");
        return None;
    }
    
    let new_info_text = format!("新建: {} x {} 像素", current_width, current_height);
    let new_info_text_utf16: Vec<u16> = new_info_text.encode_utf16().chain(std::iter::once(0)).collect();
    SetWindowTextW(new_info_label, new_info_text_utf16.as_ptr());
    println!("新图像信息标签创建成功");
    
    // 创建保存按钮
    let button_class = "BUTTON";
    let button_class_utf16: Vec<u16> = button_class.encode_utf16().chain(std::iter::once(0)).collect();
    let save_button = CreateWindowExW(
        0,
        button_class_utf16.as_ptr(),
        null_mut(),
        WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
        20, 380, 80, 30, // 调整Y坐标
        dialog_hwnd,
        IDC_SAVE_BUTTON as HMENU,
        GetModuleHandleW(null_mut()),
        null_mut(),
    );
    
    if save_button.is_null() {
        println!("错误：无法创建保存按钮");
        return None;
    }
    
    let save_text = "保存";
    let save_text_utf16: Vec<u16> = save_text.encode_utf16().chain(std::iter::once(0)).collect();
    SetWindowTextW(save_button, save_text_utf16.as_ptr());
    println!("保存按钮创建成功");
    
    // 创建取消按钮
    let cancel_button = CreateWindowExW(
        0,
        button_class_utf16.as_ptr(),
        null_mut(),
        WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
        120, 380, 80, 30, // 调整Y坐标
        dialog_hwnd,
        IDC_CANCEL_BUTTON as HMENU,
        GetModuleHandleW(null_mut()),
        null_mut(),
    );
    
    if cancel_button.is_null() {
        println!("错误：无法创建取消按钮");
        return None;
    }
    
    let cancel_text = "取消";
    let cancel_text_utf16: Vec<u16> = cancel_text.encode_utf16().chain(std::iter::once(0)).collect();
    SetWindowTextW(cancel_button, cancel_text_utf16.as_ptr());
    println!("取消按钮创建成功");
    
    // 移除锁定按钮相关代码，因为默认就是锁定状态
    println!("默认锁定宽高比，无需额外按钮");
    
    // 存储控件句柄到全局状态
    unsafe {
        DIALOG_STATE = Some(DialogState {
            hwnd: dialog_hwnd,
            width_edit,
            height_edit,

            quality_trackbar,
            quality_label,
            new_info_label,
            interpolation_combo, // 新增：插值算法选择下拉框
            is_active: true,
            parent_hwnd: hwnd,
            original_aspect_ratio: current_width as f64 / current_height as f64, // 新增：原始宽高比
            aspect_ratio_locked: true, // 新增：默认锁定宽高比
            dialog_result: None, // 新增：存储对话框结果
            should_exit: false, // 新增：标记是否应该退出对话框
        });
        
        // 验证存储的状态
        if let Some(ref dialog_state) = DIALOG_STATE {
            println!("DIALOG_STATE 初始化完成:");
            println!("  - new_info_label 句柄: {:?}", dialog_state.new_info_label);
            println!("  - 原始宽高比: {}", dialog_state.original_aspect_ratio);
            println!("  - 宽高比锁定: {}", dialog_state.aspect_ratio_locked);
            
            // 立即更新质量标签显示初始值
            update_quality_label(80, false);
        }
    }
    
    println!("所有控件创建完成，开始设置对话框为模态");
    
    // 设置对话框为模态
    println!("禁用主窗口");
    EnableWindow(hwnd, 0);
    println!("设置对话框为前台窗口");
    SetForegroundWindow(dialog_hwnd);
    
    println!("开始进入消息循环");
    let mut message_count = 0;
    
    // 消息循环处理对话框
    let mut msg: MSG = std::mem::zeroed();
    let mut dialog_result = None;
    
    while unsafe { !DIALOG_STATE.as_ref().unwrap().should_exit } && GetMessageW(&mut msg, null_mut(), 0, 0) > 0 {
        message_count += 1;
        if message_count % 100 == 0 {
            println!("已处理 {} 条消息", message_count);
        }
        
        println!("收到消息: 类型={}, 窗口={:?}, 参数={:?}", msg.message, msg.hwnd, msg.wParam);
        
        // 转发所有消息，确保主窗口能正常响应
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
        
        // 只在对话框相关消息时处理宽高比锁定和实时更新逻辑
        if msg.hwnd == dialog_hwnd || IsChild(dialog_hwnd, msg.hwnd) != 0 {
            println!("处理对话框相关消息");
            
            // 处理编辑框的实时输入和宽高比锁定
            if msg.message == WM_CHAR {
                println!("收到 WM_CHAR 消息，窗口: {:?}", msg.hwnd);
                // 延迟处理，等待文本更新完成
                std::thread::sleep(std::time::Duration::from_millis(100));
                
                if let Some(ref mut dialog_state) = DIALOG_STATE {
                    if msg.hwnd == dialog_state.height_edit {
                        println!("高度输入框收到字符输入");
                        // 高度发生变化，如果锁定宽高比，则自动调整宽度
                        if dialog_state.aspect_ratio_locked {
                            println!("宽高比已锁定，开始调整宽度");
                            // 获取新的高度值
                            let text_len = SendMessageW(dialog_state.height_edit, WM_GETTEXTLENGTH, 0, 0);
                            if text_len > 0 {
                                let mut buffer = vec![0u16; text_len as usize + 1];
                                SendMessageW(dialog_state.height_edit, WM_GETTEXT, buffer.len(), buffer.as_mut_ptr() as LPARAM);
                                if let Ok(height_str) = String::from_utf16(&buffer[..text_len as usize]) {
                                    if let Ok(new_height) = height_str.parse::<i32>() {
                                        if new_height > 0 {
                                            // 根据宽高比计算新的宽度
                                            let new_width = (new_height as f64 * dialog_state.original_aspect_ratio) as i32;
                                            // 更新宽度输入框
                                            let width_text = format!("{}", new_width);
                                            let width_text_utf16: Vec<u16> = width_text.encode_utf16().chain(std::iter::once(0)).collect();
                                            SetWindowTextW(dialog_state.width_edit, width_text_utf16.as_ptr());
                                            println!("高度变化: {} -> 宽度自动调整为: {}", new_height, new_width);
                                            
                                            // 更新信息标签显示新的尺寸
                                            let new_info_text = format!("新建: {} x {} 像素", new_width, new_height);
                                            let new_info_text_utf16: Vec<u16> = new_info_text.encode_utf16().chain(std::iter::once(0)).collect();
                                            println!("[WM_CHAR高度] 准备更新信息标签，新文本: '{}', 标签句柄: {:?}", new_info_text, dialog_state.new_info_label);
                                            let result = SetWindowTextW(dialog_state.new_info_label, new_info_text_utf16.as_ptr());
                                            println!("[WM_CHAR高度] SetWindowTextW 结果: {}", if result == 0 { "失败" } else { "成功" });
                                            
                                            // 强制重绘标签
                                            InvalidateRect(dialog_state.new_info_label, null_mut(), 1);
                                            UpdateWindow(dialog_state.new_info_label);
                                            println!("[WM_CHAR高度] 已强制重绘标签");
                                        }
                                    }
                                }
                            }
                        }
                    } else if msg.hwnd == dialog_state.width_edit {
                        println!("宽度输入框收到字符输入");
                        // 宽度发生变化，如果锁定宽高比，则自动调整高度
                        if dialog_state.aspect_ratio_locked {
                            println!("宽高比已锁定，开始调整高度");
                            // 获取新的宽度值
                            let text_len = SendMessageW(dialog_state.width_edit, WM_GETTEXTLENGTH, 0, 0);
                            if text_len > 0 {
                                let mut buffer = vec![0u16; text_len as usize + 1];
                                SendMessageW(dialog_state.width_edit, WM_GETTEXT, buffer.len(), buffer.as_mut_ptr() as LPARAM);
                                if let Ok(width_str) = String::from_utf16(&buffer[..text_len as usize]) {
                                    if let Ok(new_width) = width_str.parse::<i32>() {
                                        if new_width > 0 {
                                            // 根据宽高比计算新的高度
                                            let new_height = (new_width as f64 / dialog_state.original_aspect_ratio) as i32;
                                            // 更新高度输入框
                                            let height_text = format!("{}", new_height);
                                            let height_text_utf16: Vec<u16> = height_text.encode_utf16().chain(std::iter::once(0)).collect();
                                            SetWindowTextW(dialog_state.height_edit, height_text_utf16.as_ptr());
                                            println!("宽度变化: {} -> 高度自动调整为: {}", new_width, new_height);
                                            
                                            // 更新信息标签显示新的尺寸
                                            let new_info_text = format!("新建: {} x {} 像素", new_width, new_height);
                                            let new_info_text_utf16: Vec<u16> = new_info_text.encode_utf16().chain(std::iter::once(0)).collect();
                                            println!("[WM_CHAR宽度] 准备更新信息标签，新文本: '{}', 标签句柄: {:?}", new_info_text, dialog_state.new_info_label);
                                            let result = SetWindowTextW(dialog_state.new_info_label, new_info_text_utf16.as_ptr());
                                            println!("[WM_CHAR宽度] SetWindowTextW 结果: {}", if result == 0 { "失败" } else { "成功" });
                                            
                                            // 强制重绘标签
                                            InvalidateRect(dialog_state.new_info_label, null_mut(), 1);
                                            UpdateWindow(dialog_state.new_info_label);
                                            println!("[WM_CHAR宽度] 已强制重绘标签");
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    println!("消息循环结束，should_exit: {}", unsafe { DIALOG_STATE.as_ref().unwrap().should_exit });
    
    // 恢复主窗口
    println!("恢复主窗口状态");
    EnableWindow(hwnd, 1);
    
    // 确保主窗口重新获得焦点并显示在前台
    SetForegroundWindow(hwnd);
    ShowWindow(hwnd, SW_RESTORE);
    
    // 清理全局状态
    unsafe {
        let final_result = DIALOG_STATE.as_ref().and_then(|state| state.dialog_result.clone());
        DIALOG_STATE = None;
        dialog_result = final_result;
    }
    
    // 销毁对话框
    println!("销毁对话框窗口");
    DestroyWindow(dialog_hwnd);
    
    // 注销窗口类
    UnregisterClassW(dialog_class_utf16.as_ptr(), GetModuleHandleW(null_mut()));
    
    println!("对话框销毁完成，返回结果: {:?}", dialog_result);
    dialog_result
}

// 更新质量标签显示
unsafe fn update_quality_label(quality: i32, force_repaint: bool) {
    if let Some(ref dialog_state) = DIALOG_STATE {
        let quality_text = match quality {
            1..=20 => format!("质量: {}% (很低)", quality),
            21..=40 => format!("质量: {}% (低)", quality),
            41..=60 => format!("质量: {}% (中等)", quality),
            61..=80 => format!("质量: {}% (高)", quality),
            81..=100 => format!("质量: {}% (很高)", quality),
            _ => format!("质量: {}%", quality),
        };
        let quality_text_utf16: Vec<u16> = quality_text.encode_utf16().chain(std::iter::once(0)).collect();
        
        println!("[update_quality_label] 更新质量标签，质量值: {}, 文本: '{}'", quality, quality_text);
        println!("[update_quality_label] quality_label句柄: 0x{:X}", dialog_state.quality_label as usize);
        
        let result = SetWindowTextW(dialog_state.quality_label, quality_text_utf16.as_ptr());
        println!("[update_quality_label] SetWindowTextW结果: {}", if result != 0 { "成功" } else { "失败" });
        
        // 只有在需要强制重绘时才调用重绘函数
        if force_repaint {
            // 只重绘标签，不重绘整个对话框
            InvalidateRect(dialog_state.quality_label, null_mut(), 1);
            UpdateWindow(dialog_state.quality_label);
            println!("[update_quality_label] 已调用InvalidateRect和UpdateWindow强制重绘质量标签");
        }
    } else {
        println!("[update_quality_label] 错误: DIALOG_STATE为None");
    }
}

// 对话框窗口过程函数
unsafe extern "system" fn dialog_proc(
    hwnd: HWND,
    msg: UINT,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    match msg {
        WM_COMMAND => {
            let control_id = (wparam & 0xFFFF) as i32;
            println!("[dialog_proc] 收到 WM_COMMAND，控件ID: {}", control_id);
            
            if let Some(ref mut dialog_state) = DIALOG_STATE {
                if control_id == IDC_SAVE_BUTTON {
                    println!("[dialog_proc] 保存按钮被点击，开始处理保存逻辑");
                    
                    // 获取输入框的值
                    let mut text_len = SendMessageW(dialog_state.width_edit, WM_GETTEXTLENGTH, 0, 0);
                    let mut new_width = 0;
                    if text_len > 0 {
                        let mut buffer = vec![0u16; text_len as usize + 1];
                        SendMessageW(dialog_state.width_edit, WM_GETTEXT, buffer.len(), buffer.as_mut_ptr() as LPARAM);
                        if let Ok(width_str) = String::from_utf16(&buffer[..text_len as usize]) {
                            if let Ok(width) = width_str.parse::<i32>() {
                                new_width = width;
                                println!("[dialog_proc] 设置新宽度: {}", width);
                            }
                        }
                    }
                    
                    text_len = SendMessageW(dialog_state.height_edit, WM_GETTEXTLENGTH, 0, 0);
                    let mut new_height = 0;
                    if text_len > 0 {
                        let mut buffer = vec![0u16; text_len as usize + 1];
                        SendMessageW(dialog_state.height_edit, WM_GETTEXT, buffer.len(), buffer.as_mut_ptr() as LPARAM);
                        if let Ok(height_str) = String::from_utf16(&buffer[..text_len as usize]) {
                            if let Ok(height) = height_str.parse::<i32>() {
                                new_height = height;
                                println!("[dialog_proc] 设置新高度: {}", height);
                            }
                        }
                    }
                    
                    // 获取滑块值
                    let quality = SendMessageW(dialog_state.quality_trackbar, TBM_GETPOS, 0, 0) as i32;
                    println!("[dialog_proc] 设置新质量: {}", quality);
                    
                    // 获取选择的插值算法
                    let interpolation_index = SendMessageW(dialog_state.interpolation_combo, CB_GETCURSEL, 0, 0) as i32;
                    let interpolation = match interpolation_index {
                        0 => INTER_NEAREST,
                        1 => INTER_LINEAR,
                        2 => INTER_CUBIC,
                        3 => INTER_AREA,
                        4 => INTER_LANCZOS4,
                        _ => INTER_LINEAR, // 默认使用双线性插值
                    };
                    println!("[dialog_proc] 选择的插值算法: {} (索引: {})", 
                        match interpolation {
                            INTER_NEAREST => "最近邻插值",
                            INTER_LINEAR => "双线性插值",
                            INTER_CUBIC => "双三次插值",
                            INTER_AREA => "区域插值",
                            INTER_LANCZOS4 => "Lanczos插值",
                            _ => "未知算法",
                        }, interpolation_index);
                    
                    // 创建结果对象
                    let result = ResizeResult {
                        new_width,
                        new_height,
                        quality,
                        interpolation, // 使用选择的插值算法
                    };
                    
                    // 显示保存文件对话框，预填原图像文件名
                    if let Some(ref state) = IMAGE_STATE {
                        if let Some(ref file_path) = state.current_file_path {
                            // 构建新的文件名：原文件名 + "_resized" + 原扩展名
                            let path = std::path::Path::new(file_path);
                            let stem = path.file_stem().unwrap_or_default().to_string_lossy();
                            let extension = path.extension().unwrap_or_default().to_string_lossy();
                            let new_filename = format!("{}_resized.{}", stem, extension);
                            
                            println!("[dialog_proc] 准备显示保存对话框，预填文件名: {}", new_filename);
                            
                            // 创建保存文件对话框
                            let mut file_name = [0u16; 260]; // MAX_PATH
                            let new_filename_utf16: Vec<u16> = new_filename.encode_utf16().chain(std::iter::once(0)).collect();
                            
                            // 复制文件名到缓冲区
                            let mut i = 0;
                            for &ch in &new_filename_utf16 {
                                if i < file_name.len() {
                                    file_name[i] = ch;
                                    i += 1;
                                }
                            }
                            
                            let filter = "PNG 文件\0*.png\0JPEG 文件\0*.jpg;*.jpeg\0BMP 文件\0*.bmp\0TIFF 文件\0*.tiff\0所有文件\0*.*\0\0";
                            let filter_utf16: Vec<u16> = filter.encode_utf16().collect();
                            let title_utf16: Vec<u16> = "保存调整后的图像\0".encode_utf16().collect();
                            let def_ext_utf16: Vec<u16> = "png\0".encode_utf16().collect();
                            
                            let mut ofn = OPENFILENAMEW {
                                lStructSize: std::mem::size_of::<OPENFILENAMEW>() as u32,
                                hwndOwner: hwnd, // 使用dialog_proc的hwnd作为所有者
                                hInstance: null_mut(),
                                lpstrFilter: filter_utf16.as_ptr(),
                                lpstrCustomFilter: null_mut(),
                                nMaxCustFilter: 0,
                                nFilterIndex: 1,
                                lpstrFile: file_name.as_mut_ptr(),
                                nMaxFile: file_name.len() as u32,
                                lpstrFileTitle: null_mut(),
                                nMaxFileTitle: 0,
                                lpstrInitialDir: null_mut(),
                                lpstrTitle: title_utf16.as_ptr(),
                                Flags: OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST,
                                nFileOffset: 0,
                                nFileExtension: 0,
                                lpstrDefExt: def_ext_utf16.as_ptr(),
                                lCustData: 0,
                                lpfnHook: None,
                                lpTemplateName: null_mut(),
                                pvReserved: null_mut(),
                                dwReserved: 0,
                                FlagsEx: 0,
                            };
                            
                            println!("[dialog_proc] 准备调用 GetSaveFileNameW...");
                            let save_result_api = GetSaveFileNameW(&mut ofn);
                            println!("[dialog_proc] GetSaveFileNameW 返回结果: {}", save_result_api);
                            
                            if save_result_api != 0 {
                                let len = file_name.iter().position(|&c| c == 0).unwrap_or(file_name.len());
                                let save_path = String::from_utf16_lossy(&file_name[..len]);
                                println!("[dialog_proc] 用户选择保存路径: {}", save_path);
                                
                                // 实际保存图像
                                if let Some(ref state) = IMAGE_STATE {
                                    if let Some(ref current_mat) = state.current_mat {
                                        match ocv_bridge::save_resized_image(
                                            current_mat,
                                            result.new_width,
                                            result.new_height,
                                            &save_path,
                                            result.quality,
                                            result.interpolation,
                                        ) {
                                            Ok(()) => {
                                                println!("[dialog_proc] 图像保存成功");
                                                // 设置对话框结果并退出
                                                dialog_state.dialog_result = Some(result.clone());
                                                dialog_state.should_exit = true;
                                                println!("[dialog_proc] 保存操作完成，设置退出标志");
                                                PostMessageW(hwnd, WM_CLOSE, 0, 0);
                                            },
                                            Err(e) => {
                                                println!("[dialog_proc] 图像保存失败: {}", e);
                                                // 显示错误消息，但不退出对话框
                                                let error_msg = format!("保存图像失败：\n{}", e);
                                                let error_msg_utf16: Vec<u16> = error_msg.encode_utf16().chain(std::iter::once(0)).collect();
                                                let title_utf16: Vec<u16> = "保存失败".encode_utf16().chain(std::iter::once(0)).collect();
                                                MessageBoxW(
                                                    hwnd, 
                                                    error_msg_utf16.as_ptr(), 
                                                    title_utf16.as_ptr(), 
                                                    MB_OK | MB_ICONERROR
                                                );
                                            }
                                        }
                                    } else {
                                        println!("[dialog_proc] 无法获取当前图像");
                                    }
                                } else {
                                    println!("[dialog_proc] 无法获取图像状态");
                                }
                            } else {
                                println!("[dialog_proc] GetSaveFileNameW 失败或用户取消");
                                // 不退出对话框，让用户继续操作
                            }
                        } else {
                            println!("[dialog_proc] 没有原图像文件路径，使用默认保存对话框 (未实现)");
                        }
                    } else {
                        println!("[dialog_proc] 无法获取图像状态，使用默认保存对话框 (未实现)");
                    }
                } else if control_id == IDC_CANCEL_BUTTON {
                    println!("[dialog_proc] 取消按钮被点击，设置退出标志");
                    dialog_state.should_exit = true;
                    PostMessageW(hwnd, WM_CLOSE, 0, 0); // 发送WM_CLOSE退出对话框
                }
            }
            0
        }
        WM_NOTIFY => {
            // 处理trackbar的通知消息
            println!("[dialog_proc] 收到 WM_NOTIFY - wparam: 0x{:X}, lparam: 0x{:X}", wparam, lparam);
            
            if let Some(ref dialog_state) = DIALOG_STATE {
                // 检查是否来自trackbar的通知
                let control_id = wparam as i32;
                println!("[dialog_proc] WM_NOTIFY控件ID: {}", control_id);
                
                if control_id == IDC_QUALITY_TRACKBAR {
                    println!("[dialog_proc] 收到trackbar通知");
                    // 获取当前trackbar位置并更新标签
                    let current_pos = SendMessageW(dialog_state.quality_trackbar, TBM_GETPOS, 0, 0) as i32;
                    println!("[dialog_proc] trackbar当前位置: {}", current_pos);
                    update_quality_label(current_pos, false);
                }
            }
            0
        }
        WM_HSCROLL => {
            // 处理水平滚动条消息（包括trackbar）
            println!("[dialog_proc] 收到 WM_HSCROLL - wparam: 0x{:X}, lparam: 0x{:X}", wparam, lparam);
            
            if let Some(ref dialog_state) = DIALOG_STATE {
                // 检查是否来自质量trackbar
                if lparam == dialog_state.quality_trackbar as isize {
                    println!("[dialog_proc] 收到质量trackbar的WM_HSCROLL消息");
                    let scroll_code = (wparam & 0xFFFF) as u16;
                    println!("[dialog_proc] 滚动代码: {}", scroll_code);
                    
                    // 获取当前trackbar位置并更新标签
                    let current_pos = SendMessageW(dialog_state.quality_trackbar, TBM_GETPOS, 0, 0) as i32;
                    println!("[dialog_proc] trackbar当前位置: {}", current_pos);
                    update_quality_label(current_pos, false);
                }
            }
            0
        }
        WM_CLOSE => {
            println!("[dialog_proc] 收到 WM_CLOSE，销毁窗口");
            DestroyWindow(hwnd);
            0
        }
        WM_DESTROY => {
            println!("[dialog_proc] 收到 WM_DESTROY");
            0
        }
        _ => DefWindowProcW(hwnd, msg, wparam, lparam),
    }
}

// 调整图像大小
fn resize_image(mat: &core::Mat, new_width: i32, new_height: i32, quality: i32) -> Result<core::Mat> {
    use opencv::imgproc;
    
    if new_width <= 0 || new_height <= 0 {
        return Err(anyhow::anyhow!("无效的尺寸: {}x{}", new_width, new_height));
    }
    
    if quality < 1 || quality > 100 {
        return Err(anyhow::anyhow!("无效的质量值: {}", quality));
    }
    
    // 使用OpenCV的resize函数调整图像大小
    let mut resized = core::Mat::default();
    let size = core::Size::new(new_width, new_height);
    
    imgproc::resize(
        mat,
        &mut resized,
        size,
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )?;
    
    Ok(resized)
}

// 图像旋转函数
// 180度旋转
fn rotate_image_180(mat: &core::Mat) -> Result<core::Mat> {
    let mut rotated = core::Mat::default();
    
    // 使用ROTATE_180标志进行180度旋转
    core::rotate(mat, &mut rotated, core::ROTATE_180)?;
    
    Ok(rotated)
}

// 顺时针90度旋转
fn rotate_image_90_cw(mat: &core::Mat) -> Result<core::Mat> {
    let mut rotated = core::Mat::default();
    
    // 使用ROTATE_90_CLOCKWISE标志进行顺时针90度旋转
    core::rotate(mat, &mut rotated, core::ROTATE_90_CLOCKWISE)?;
    
    Ok(rotated)
}

// 逆时针90度旋转
fn rotate_image_90_ccw(mat: &core::Mat) -> Result<core::Mat> {
    let mut rotated = core::Mat::default();
    
    // 使用ROTATE_90_COUNTERCLOCKWISE标志进行逆时针90度旋转
    core::rotate(mat, &mut rotated, core::ROTATE_90_COUNTERCLOCKWISE)?;
    
    Ok(rotated)
}

// 计算直线与水平线的夹角（以度为单位）
fn calculate_horizontal_angle(x1: i32, y1: i32, x2: i32, y2: i32) -> f64 {
    // 计算直线的向量
    let dx = (x2 - x1) as f64;
    let dy = (y2 - y1) as f64;
    
    // 计算与水平线（x轴正方向）的夹角
    // 使用 atan2 计算角度，返回值范围为 [-π, π]
    let angle_rad = dy.atan2(dx);
    
    // 转换为度数
    let angle_deg = angle_rad.to_degrees();
    
    // 直接返回计算出的角度，用于校正图像使直线变水平
    angle_deg
}

// 任意角度旋转
fn rotate_image_angle(mat: &core::Mat, angle: f64) -> Result<core::Mat> {
    let center = core::Point2f::new(
        mat.cols() as f32 / 2.0,
        mat.rows() as f32 / 2.0,
    );
    
    // 创建旋转矩阵
    let rotation_matrix = imgproc::get_rotation_matrix_2d(center, angle, 1.0)?;
    
    let mut rotated = core::Mat::default();
    let size = core::Size::new(mat.cols(), mat.rows());
    
    // 应用仿射变换进行旋转
    imgproc::warp_affine(
        mat,
        &mut rotated,
        &rotation_matrix,
        size,
        imgproc::INTER_LINEAR,
        core::BORDER_CONSTANT,
        core::Scalar::all(0.0),
    )?;
    
    Ok(rotated)
}

// 显示角度输入对话框
unsafe fn show_angle_input_dialog(parent_hwnd: HWND) -> Option<f64> {
    // 使用简单的输入框获取角度
    let title = "旋转角度";
    
    // 将提示文本转换为UTF-16
    let title_utf16: Vec<u16> = title.encode_utf16().chain(std::iter::once(0)).collect();
    
    // 使用MessageBox询问用户输入（这是一个简化的实现）
    let msg = "请在控制台输入旋转角度（正数为逆时针，负数为顺时针），然后按Enter确认。";
    let msg_utf16: Vec<u16> = msg.encode_utf16().chain(std::iter::once(0)).collect();
    
    if MessageBoxW(parent_hwnd, msg_utf16.as_ptr(), title_utf16.as_ptr(), MB_OKCANCEL | MB_ICONQUESTION) == IDOK {
        // 在控制台获取用户输入
        print!("请输入旋转角度: ");
        use std::io::{self, Write};
        io::stdout().flush().unwrap();
        
        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_ok() {
            if let Ok(angle) = input.trim().parse::<f64>() {
                return Some(angle);
            }
        }
    }
    
    None
}

// 添加拖拽文件处理函数
unsafe fn handle_drop_files(hwnd: HWND, hdrop: HDROP) {
    // 获取拖拽的文件数量
    let file_count = DragQueryFileW(hdrop, 0xFFFFFFFF, null_mut(), 0);
    
    if file_count > 0 {
        // 只处理第一个文件
        let buffer_size = DragQueryFileW(hdrop, 0, null_mut(), 0) + 1;
        let mut buffer: Vec<u16> = vec![0; buffer_size as usize];
        
        if DragQueryFileW(hdrop, 0, buffer.as_mut_ptr(), buffer_size) > 0 {
            // 将UTF-16转换为Rust字符串
            let file_path = String::from_utf16_lossy(&buffer[..buffer_size as usize - 1]);
            println!("📁 拖拽文件: {}", file_path);
            
            // 检查文件扩展名是否为支持的图像格式
            let supported_extensions = ["jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp"];
            let ext = std::path::Path::new(&file_path)
                .extension()
                .and_then(|s| s.to_str())
                .map(|s| s.to_lowercase());
            
            if let Some(extension) = ext {
                if supported_extensions.contains(&extension.as_str()) {
                    // 加载图像
                    match load_image_from_path(&file_path) {
                        Ok(_) => {
                            println!("✅ 成功加载拖拽的图像: {}", file_path);
                            InvalidateRect(hwnd, null_mut(), 1);
                        }
                        Err(e) => {
                            println!("❌ 加载拖拽图像失败: {}", e);
                            let msg = format!("无法加载拖拽的图像：\n{}\n\n错误：{}", file_path, e);
                            let msg_utf16: Vec<u16> = msg.encode_utf16().chain(std::iter::once(0)).collect();
                            let title_utf16: Vec<u16> = "加载图像失败".encode_utf16().chain(std::iter::once(0)).collect();
                            MessageBoxW(hwnd, msg_utf16.as_ptr(), title_utf16.as_ptr(), MB_OK | MB_ICONERROR);
                        }
                    }
                } else {
                    let msg = format!("不支持的文件格式：{}\n\n支持的格式：JPG, PNG, BMP, TIFF, WebP", extension.to_uppercase());
                    let msg_utf16: Vec<u16> = msg.encode_utf16().chain(std::iter::once(0)).collect();
                    let title_utf16: Vec<u16> = "不支持的文件格式".encode_utf16().chain(std::iter::once(0)).collect();
                            MessageBoxW(hwnd, msg_utf16.as_ptr(), title_utf16.as_ptr(), MB_OK | MB_ICONWARNING);
                }
            }
        }
    }
    
    // 释放拖拽资源
    DragFinish(hdrop);
}

// 从文件路径加载图像的函数
fn load_image_from_path(file_path: &str) -> Result<()> {
    let state = unsafe { IMAGE_STATE.as_mut().unwrap() };
    
    // 使用image库快速获取图片尺寸（不解码全图）
    let (img_w, img_h) = match image::io::Reader::open(file_path) {
        Ok(reader) => {
            match reader.into_dimensions() {
                Ok(dimensions) => dimensions,
                Err(_) => {
                    println!("无法获取图片尺寸，尝试直接加载");
                    (0, 0) // 标记为未知尺寸
                }
            }
        }
        Err(_) => {
            println!("无法打开图片文件，尝试直接加载");
            (0, 0) // 标记为未知尺寸
        }
    };
    
    let max_dim = img_w.max(img_h);
    let is_large_image = max_dim > 0 && max_dim > state.display_max_size as u32;
    
    if is_large_image {
        println!("🔍 检测到大图 ({}x{})，加载缩小版本以提升速度", img_w, img_h);
    }
    
    // 使用OpenCV加载图像
    let mut mat = imgcodecs::imread(file_path, imgcodecs::IMREAD_COLOR)
        .context("无法读取图像文件")?;
    
    if mat.empty() {
        return Err(anyhow::anyhow!("图像文件为空或格式不受支持"));
    }
    
    // 如果是大图，缩小到display_max_size
    if is_large_image {
        let scale = state.display_max_size as f32 / max_dim as f32;
        let new_w = (img_w as f32 * scale) as i32;
        let new_h = (img_h as f32 * scale) as i32;
        
        let mut resized = core::Mat::default();
        imgproc::resize(&mat, &mut resized, 
            core::Size::new(new_w, new_h), 
            0.0, 0.0, 
            imgproc::INTER_AREA)
            .context("缩放图像失败")?;
        
        mat = resized;
        println!("✅ 已缩放至: {}x{}", new_w, new_h);
    }
    
    // 转换为显示格式
    match ocv_bridge::mat_to_bgra(&mat) {
        Ok(image_buf) => {
            state.display_pixels = image_buf.pixels;
            state.current_width = image_buf.w;
            state.current_height = image_buf.h;
            state.current_file_path = Some(file_path.to_string());
            
            // 更新Mat数据
            state.original_bgr = Some(mat.clone());
            state.current_mat = Some(mat);
            
            // 更新智能加载状态
            state.full_resolution_path = Some(file_path.to_string());
            state.is_full_resolution_loaded = !is_large_image;
            
            // 重置缩放状态
            state.reset_zoom();
            
            // 更新图片列表
            state.update_image_list();
            
            if is_large_image {
                println!("⚡ 图像快速加载完成（拖拽，使用缩小版本）");
            } else {
                println!("✅ 图像加载成功：{}x{}", image_buf.w, image_buf.h);
            }
            
            // 更新EXIF面板（如果打开）
            unsafe { update_exif_panel_if_visible(); }
            
            Ok(())
        }
        Err(e) => {
            Err(anyhow::anyhow!("转换图像格式失败: {}", e))
        }
    }
}

// 新增：批量缩图对话框结构体
#[derive(Debug, Clone)]
struct BatchResizeSettings {
    input_folder: String,
    output_folder: String,
    long_edge_limit: i32,
    quality: i32,
    interpolation: i32,
}

// 新增：显示批量缩图对话框
unsafe fn show_batch_resize_dialog(hwnd: HWND) {
    println!("开始创建批量缩图对话框");
    
    // 注册对话框窗口类
    let dialog_class = "BatchResizeDialogClass";
    let dialog_class_utf16: Vec<u16> = dialog_class.encode_utf16().chain(std::iter::once(0)).collect();
    
    let wc = WNDCLASSW {
        style: CS_HREDRAW | CS_VREDRAW,
        lpfnWndProc: Some(batch_resize_dialog_proc),
        cbClsExtra: 0,
        cbWndExtra: 0,
        hInstance: GetModuleHandleW(null_mut()),
        hIcon: null_mut(),
        hCursor: LoadCursorW(null_mut(), IDC_ARROW),
        hbrBackground: (COLOR_WINDOW + 1) as HBRUSH,
        lpszMenuName: null_mut(),
        lpszClassName: dialog_class_utf16.as_ptr(),
    };
    
    if RegisterClassW(&wc) == 0 {
        println!("错误：无法注册批量缩图对话框窗口类");
        return;
    }
    
    // 创建模态对话框窗口
    let dialog_hwnd = CreateWindowExW(
        WS_EX_DLGMODALFRAME | WS_EX_TOPMOST,
        dialog_class_utf16.as_ptr(),
        null_mut(),
        WS_POPUP | WS_CAPTION | WS_SYSMENU | WS_VISIBLE | DS_MODALFRAME,
        100, 100, 500, 600,
        hwnd,
        null_mut(),
        GetModuleHandleW(null_mut()),
        null_mut(),
    );
    
    if dialog_hwnd.is_null() {
        println!("错误：无法创建批量缩图对话框窗口");
        return;
    }
    
    println!("批量缩图对话框窗口创建成功，句柄: {:?}", dialog_hwnd);
    
    // 设置对话框标题
    let title = "Resize";
    let title_utf16: Vec<u16> = title.encode_utf16().chain(std::iter::once(0)).collect();
    SetWindowTextW(dialog_hwnd, title_utf16.as_ptr());
    
    // 创建控件
    let static_class = "STATIC";
    let static_class_utf16: Vec<u16> = static_class.encode_utf16().chain(std::iter::once(0)).collect();
    let edit_class = "EDIT";
    let edit_class_utf16: Vec<u16> = edit_class.encode_utf16().chain(std::iter::once(0)).collect();
    let button_class = "BUTTON";
    let button_class_utf16: Vec<u16> = button_class.encode_utf16().chain(std::iter::once(0)).collect();
    
    // 创建 Input folder 标签
    let input_label = CreateWindowExW(
        0,
        static_class_utf16.as_ptr(),
        null_mut(),
        WS_VISIBLE | WS_CHILD | SS_LEFT,
        20, 30, 80, 20,
        dialog_hwnd,
        null_mut(),
        GetModuleHandleW(null_mut()),
        null_mut(),
    );
    
    let input_label_text = "Input folder:";
    let input_label_text_utf16: Vec<u16> = input_label_text.encode_utf16().chain(std::iter::once(0)).collect();
    SetWindowTextW(input_label, input_label_text_utf16.as_ptr());
    
    // 创建 Input folder 输入框
    let input_edit = CreateWindowExW(
        WS_EX_CLIENTEDGE,
        edit_class_utf16.as_ptr(),
        null_mut(),
        WS_VISIBLE | WS_CHILD | WS_BORDER | ES_AUTOHSCROLL,
        110, 30, 250, 20,
        dialog_hwnd,
        IDC_INPUT_FOLDER_EDIT as HMENU,
        GetModuleHandleW(null_mut()),
        null_mut(),
    );
    
    // 创建 Input folder Open 按钮
    let input_button = CreateWindowExW(
        0,
        button_class_utf16.as_ptr(),
        null_mut(),
        WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
        370, 30, 60, 20,
        dialog_hwnd,
        IDC_INPUT_FOLDER_BUTTON as HMENU,
        GetModuleHandleW(null_mut()),
        null_mut(),
    );
    
    let input_button_text = "Open";
    let input_button_text_utf16: Vec<u16> = input_button_text.encode_utf16().chain(std::iter::once(0)).collect();
    SetWindowTextW(input_button, input_button_text_utf16.as_ptr());
    
    // 创建 Output folder 标签
    let output_label = CreateWindowExW(
        0,
        static_class_utf16.as_ptr(),
        null_mut(),
        WS_VISIBLE | WS_CHILD | SS_LEFT,
        20, 70, 80, 20,
        dialog_hwnd,
        null_mut(),
        GetModuleHandleW(null_mut()),
        null_mut(),
    );
    
    let output_label_text = "Output folder:";
    let output_label_text_utf16: Vec<u16> = output_label_text.encode_utf16().chain(std::iter::once(0)).collect();
    SetWindowTextW(output_label, output_label_text_utf16.as_ptr());
    
    // 创建 Output folder 输入框
    let output_edit = CreateWindowExW(
        WS_EX_CLIENTEDGE,
        edit_class_utf16.as_ptr(),
        null_mut(),
        WS_VISIBLE | WS_CHILD | WS_BORDER | ES_AUTOHSCROLL,
        110, 70, 250, 20,
        dialog_hwnd,
        IDC_OUTPUT_FOLDER_EDIT as HMENU,
        GetModuleHandleW(null_mut()),
        null_mut(),
    );
    
    // 创建 Output folder Open 按钮
    let output_button = CreateWindowExW(
        0,
        button_class_utf16.as_ptr(),
        null_mut(),
        WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
        370, 70, 60, 20,
        dialog_hwnd,
        IDC_OUTPUT_FOLDER_BUTTON as HMENU,
        GetModuleHandleW(null_mut()),
        null_mut(),
    );
    
    let output_button_text = "Open";
    let output_button_text_utf16: Vec<u16> = output_button_text.encode_utf16().chain(std::iter::once(0)).collect();
    SetWindowTextW(output_button, output_button_text_utf16.as_ptr());
    
    // 创建 长边限制 标签
    let long_edge_label = CreateWindowExW(
        0,
        static_class_utf16.as_ptr(),
        null_mut(),
        WS_VISIBLE | WS_CHILD | SS_LEFT,
        20, 110, 80, 20,
        dialog_hwnd,
        null_mut(),
        GetModuleHandleW(null_mut()),
        null_mut(),
    );
    
    let long_edge_label_text = "长边限制:";
    let long_edge_label_text_utf16: Vec<u16> = long_edge_label_text.encode_utf16().chain(std::iter::once(0)).collect();
    SetWindowTextW(long_edge_label, long_edge_label_text_utf16.as_ptr());
    
    // 创建 长边限制 输入框
    let long_edge_edit = CreateWindowExW(
        WS_EX_CLIENTEDGE,
        edit_class_utf16.as_ptr(),
        null_mut(),
        WS_VISIBLE | WS_CHILD | WS_BORDER | ES_NUMBER,
        110, 110, 100, 20,
        dialog_hwnd,
        IDC_LONG_EDGE_EDIT as HMENU,
        GetModuleHandleW(null_mut()),
        null_mut(),
    );
    
    // 设置默认值 1024
    let long_edge_default = "1024";
    let long_edge_default_utf16: Vec<u16> = long_edge_default.encode_utf16().chain(std::iter::once(0)).collect();
    SetWindowTextW(long_edge_edit, long_edge_default_utf16.as_ptr());
    
    // 创建质量标签
    let quality_label = CreateWindowExW(
        0,
        static_class_utf16.as_ptr(),
        null_mut(),
        WS_VISIBLE | WS_CHILD | SS_LEFT,
        20, 150, 100, 20,
        dialog_hwnd,
        null_mut(),
        GetModuleHandleW(null_mut()),
        null_mut(),
    );
    
    let quality_label_text = "质量: 80% (高)";
    let quality_label_text_utf16: Vec<u16> = quality_label_text.encode_utf16().chain(std::iter::once(0)).collect();
    SetWindowTextW(quality_label, quality_label_text_utf16.as_ptr());
    
    // 创建质量滑块
    let msctls_trackbar32_class = "msctls_trackbar32";
    let msctls_trackbar32_class_utf16: Vec<u16> = msctls_trackbar32_class.encode_utf16().chain(std::iter::once(0)).collect();
    let quality_trackbar = CreateWindowExW(
        0,
        msctls_trackbar32_class_utf16.as_ptr(),
        null_mut(),
        WS_VISIBLE | WS_CHILD | TBS_HORZ | TBS_AUTOTICKS | WS_TABSTOP,
        20, 180, 200, 30,
        dialog_hwnd,
        IDC_BATCH_QUALITY_TRACKBAR as HMENU,
        GetModuleHandleW(null_mut()),
        null_mut(),
    );
    
    // 设置滑块范围 (1-100)
    SendMessageW(quality_trackbar, TBM_SETRANGE, 1, MAKELONG!(1, 100) as isize);
    SendMessageW(quality_trackbar, TBM_SETPOS, 1, 80);
    
    // 创建插值算法标签
    let interpolation_label = CreateWindowExW(
        0,
        static_class_utf16.as_ptr(),
        null_mut(),
        WS_VISIBLE | WS_CHILD | SS_LEFT,
        20, 220, 100, 20,
        dialog_hwnd,
        null_mut(),
        GetModuleHandleW(null_mut()),
        null_mut(),
    );
    
    let interpolation_label_text = "插值算法:";
    let interpolation_label_text_utf16: Vec<u16> = interpolation_label_text.encode_utf16().chain(std::iter::once(0)).collect();
    SetWindowTextW(interpolation_label, interpolation_label_text_utf16.as_ptr());
    
    // 创建插值算法选择下拉框
    let combobox_class = "COMBOBOX";
    let combobox_class_utf16: Vec<u16> = combobox_class.encode_utf16().chain(std::iter::once(0)).collect();
    let interpolation_combo = CreateWindowExW(
        0,
        combobox_class_utf16.as_ptr(),
        null_mut(),
        WS_VISIBLE | WS_CHILD | CBS_DROPDOWNLIST | WS_VSCROLL,
        110, 220, 150, 100,
        dialog_hwnd,
        IDC_BATCH_INTERPOLATION_COMBO as HMENU,
        GetModuleHandleW(null_mut()),
        null_mut(),
    );
    
    // 添加插值算法选项
    let nearest_text = "最近邻插值 (最快)";
    let nearest_text_utf16: Vec<u16> = nearest_text.encode_utf16().chain(std::iter::once(0)).collect();
    SendMessageW(interpolation_combo, CB_ADDSTRING, 0, nearest_text_utf16.as_ptr() as LPARAM);
    
    let linear_text = "双线性插值 (推荐)";
    let linear_text_utf16: Vec<u16> = linear_text.encode_utf16().chain(std::iter::once(0)).collect();
    SendMessageW(interpolation_combo, CB_ADDSTRING, 0, linear_text_utf16.as_ptr() as LPARAM);
    
    let cubic_text = "双三次插值 (高质量)";
    let cubic_text_utf16: Vec<u16> = cubic_text.encode_utf16().chain(std::iter::once(0)).collect();
    SendMessageW(interpolation_combo, CB_ADDSTRING, 0, cubic_text_utf16.as_ptr() as LPARAM);
    
    let area_text = "区域插值 (适合缩小)";
    let area_text_utf16: Vec<u16> = area_text.encode_utf16().chain(std::iter::once(0)).collect();
    SendMessageW(interpolation_combo, CB_ADDSTRING, 0, area_text_utf16.as_ptr() as LPARAM);
    
    let lanczos_text = "Lanczos插值 (最高质量)";
    let lanczos_text_utf16: Vec<u16> = lanczos_text.encode_utf16().chain(std::iter::once(0)).collect();
    SendMessageW(interpolation_combo, CB_ADDSTRING, 0, lanczos_text_utf16.as_ptr() as LPARAM);
    
    // 设置默认选择为双线性插值（索引1）
    SendMessageW(interpolation_combo, CB_SETCURSEL, 1, 0);
    
    // 创建开始转换按钮
    let start_button = CreateWindowExW(
        0,
        button_class_utf16.as_ptr(),
        null_mut(),
        WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
        20, 270, 100, 30,
        dialog_hwnd,
        IDC_START_CONVERT_BUTTON as HMENU,
        GetModuleHandleW(null_mut()),
        null_mut(),
    );
    
    let start_button_text = "开始转换";
    let start_button_text_utf16: Vec<u16> = start_button_text.encode_utf16().chain(std::iter::once(0)).collect();
    SetWindowTextW(start_button, start_button_text_utf16.as_ptr());
    
    // 创建取消按钮
    let cancel_button = CreateWindowExW(
        0,
        button_class_utf16.as_ptr(),
        null_mut(),
        WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
        140, 270, 80, 30,
        dialog_hwnd,
        IDC_CANCEL_BUTTON as HMENU,
        GetModuleHandleW(null_mut()),
        null_mut(),
    );
    
    let cancel_button_text = "取消";
    let cancel_button_text_utf16: Vec<u16> = cancel_button_text.encode_utf16().chain(std::iter::once(0)).collect();
    SetWindowTextW(cancel_button, cancel_button_text_utf16.as_ptr());
    
    // 设置对话框为模态
    EnableWindow(hwnd, 0);
    SetForegroundWindow(dialog_hwnd);
    
    // 消息循环处理对话框
    let mut msg: MSG = std::mem::zeroed();
    
    while GetMessageW(&mut msg, null_mut(), 0, 0) > 0 {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
        
        // 检查是否应该退出对话框
        if msg.message == WM_CLOSE {
            break;
        }
    }
    
    // 恢复主窗口
    EnableWindow(hwnd, 1);
    SetForegroundWindow(hwnd);
    ShowWindow(hwnd, SW_RESTORE);
    
    // 销毁对话框
    DestroyWindow(dialog_hwnd);
    
    // 注销窗口类
    UnregisterClassW(dialog_class_utf16.as_ptr(), GetModuleHandleW(null_mut()));
}

// 新增：批量缩图对话框窗口过程函数
unsafe extern "system" fn batch_resize_dialog_proc(
    hwnd: HWND,
    msg: UINT,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    match msg {
        WM_COMMAND => {
            let control_id = (wparam & 0xFFFF) as i32;
            println!("[batch_resize_dialog_proc] 收到 WM_COMMAND，控件ID: {}", control_id);
            
            match control_id {
                IDC_INPUT_FOLDER_BUTTON => {
                    println!("[batch_resize_dialog_proc] Input folder Open 按钮被点击");
                    if let Some(folder_path) = show_folder_dialog(hwnd) {
                        // 更新输入文件夹输入框
                        let input_edit = GetDlgItem(hwnd, IDC_INPUT_FOLDER_EDIT);
                        if !input_edit.is_null() {
                            let folder_path_utf16: Vec<u16> = folder_path.encode_utf16().chain(std::iter::once(0)).collect();
                            SetWindowTextW(input_edit, folder_path_utf16.as_ptr());
                            
                            // 自动设置输出文件夹为 [input_folder]/resized
                            let output_path = format!("{}/resized", folder_path);
                            let output_edit = GetDlgItem(hwnd, IDC_OUTPUT_FOLDER_EDIT);
                            if !output_edit.is_null() {
                                let output_path_utf16: Vec<u16> = output_path.encode_utf16().chain(std::iter::once(0)).collect();
                                SetWindowTextW(output_edit, output_path_utf16.as_ptr());
                            }
                        }
                    }
                }
                IDC_OUTPUT_FOLDER_BUTTON => {
                    println!("[batch_resize_dialog_proc] Output folder Open 按钮被点击");
                    if let Some(folder_path) = show_folder_dialog(hwnd) {
                        // 更新输出文件夹输入框
                        let output_edit = GetDlgItem(hwnd, IDC_OUTPUT_FOLDER_EDIT);
                        if !output_edit.is_null() {
                            let folder_path_utf16: Vec<u16> = folder_path.encode_utf16().chain(std::iter::once(0)).collect();
                            SetWindowTextW(output_edit, folder_path_utf16.as_ptr());
                        }
                    }
                }
                IDC_START_CONVERT_BUTTON => {
                    println!("[batch_resize_dialog_proc] 开始转换按钮被点击");
                    
                    // 获取所有设置
                    let mut settings = BatchResizeSettings {
                        input_folder: String::new(),
                        output_folder: String::new(),
                        long_edge_limit: 1024,
                        quality: 80,
                        interpolation: INTER_LINEAR,
                    };
                    
                    // 获取输入文件夹
                    let input_edit = GetDlgItem(hwnd, IDC_INPUT_FOLDER_EDIT);
                    if !input_edit.is_null() {
                        let text_len = SendMessageW(input_edit, WM_GETTEXTLENGTH, 0, 0);
                        if text_len > 0 {
                            let mut buffer = vec![0u16; text_len as usize + 1];
                            SendMessageW(input_edit, WM_GETTEXT, buffer.len(), buffer.as_mut_ptr() as LPARAM);
                            if let Ok(input_path) = String::from_utf16(&buffer[..text_len as usize]) {
                                settings.input_folder = input_path;
                            }
                        }
                    }
                    
                    // 获取输出文件夹
                    let output_edit = GetDlgItem(hwnd, IDC_OUTPUT_FOLDER_EDIT);
                    if !output_edit.is_null() {
                        let text_len = SendMessageW(output_edit, WM_GETTEXTLENGTH, 0, 0);
                        if text_len > 0 {
                            let mut buffer = vec![0u16; text_len as usize + 1];
                            SendMessageW(output_edit, WM_GETTEXT, buffer.len(), buffer.as_mut_ptr() as LPARAM);
                            if let Ok(output_path) = String::from_utf16(&buffer[..text_len as usize]) {
                                settings.output_folder = output_path;
                            }
                        }
                    }
                    
                    // 获取长边限制
                    let long_edge_edit = GetDlgItem(hwnd, IDC_LONG_EDGE_EDIT);
                    if !long_edge_edit.is_null() {
                        let text_len = SendMessageW(long_edge_edit, WM_GETTEXTLENGTH, 0, 0);
                        if text_len > 0 {
                            let mut buffer = vec![0u16; text_len as usize + 1];
                            SendMessageW(long_edge_edit, WM_GETTEXT, buffer.len(), buffer.as_mut_ptr() as LPARAM);
                            if let Ok(long_edge_str) = String::from_utf16(&buffer[..text_len as usize]) {
                                if let Ok(long_edge) = long_edge_str.parse::<i32>() {
                                    settings.long_edge_limit = long_edge;
                                }
                            }
                        }
                    }
                    
                    // 获取质量
                    let quality_trackbar = GetDlgItem(hwnd, IDC_BATCH_QUALITY_TRACKBAR);
                    if !quality_trackbar.is_null() {
                        settings.quality = SendMessageW(quality_trackbar, TBM_GETPOS, 0, 0) as i32;
                    }
                    
                    // 获取插值算法
                    let interpolation_combo = GetDlgItem(hwnd, IDC_BATCH_INTERPOLATION_COMBO);
                    if !interpolation_combo.is_null() {
                        let interpolation_index = SendMessageW(interpolation_combo, CB_GETCURSEL, 0, 0) as i32;
                        settings.interpolation = match interpolation_index {
                            0 => INTER_NEAREST,
                            1 => INTER_LINEAR,
                            2 => INTER_CUBIC,
                            3 => INTER_AREA,
                            4 => INTER_LANCZOS4,
                            _ => INTER_LINEAR,
                        };
                    }
                    
                    println!("[batch_resize_dialog_proc] 批量缩图设置: {:?}", settings);
                    
                    // 验证设置
                    if settings.input_folder.is_empty() {
                        let msg = "请选择输入文件夹！";
                        let msg_utf16: Vec<u16> = msg.encode_utf16().chain(std::iter::once(0)).collect();
                        let title_utf16: Vec<u16> = "设置错误".encode_utf16().chain(std::iter::once(0)).collect();
                        MessageBoxW(hwnd, msg_utf16.as_ptr(), title_utf16.as_ptr(), MB_OK | MB_ICONERROR);
                        return 0;
                    }
                    
                    if settings.output_folder.is_empty() {
                        let msg = "请选择输出文件夹！";
                        let msg_utf16: Vec<u16> = msg.encode_utf16().chain(std::iter::once(0)).collect();
                        let title_utf16: Vec<u16> = "设置错误".encode_utf16().chain(std::iter::once(0)).collect();
                        MessageBoxW(hwnd, msg_utf16.as_ptr(), title_utf16.as_ptr(), MB_OK | MB_ICONERROR);
                        return 0;
                    }
                    
                    // 关闭对话框
                    PostMessageW(hwnd, WM_CLOSE, 0, 0);
                    
                    // 在新线程中执行批量处理
                    let settings_clone = settings.clone();
                    std::thread::spawn(move || {
                        match batch_resize_images_with_settings(&settings_clone) {
                            Ok(()) => {
                                println!("✅ 批量缩图完成！");
                            }
                            Err(e) => {
                                println!("❌ 批量缩图失败：{}", e);
                            }
                        }
                    });
                }
                IDC_CANCEL_BUTTON => {
                    println!("[batch_resize_dialog_proc] 取消按钮被点击");
                    PostMessageW(hwnd, WM_CLOSE, 0, 0);
                }
                _ => {}
            }
            0
        }
        WM_NOTIFY => {
            // 处理质量滑块的通知消息
            let control_id = (wparam & 0xFFFF) as i32;
            if control_id == IDC_BATCH_QUALITY_TRACKBAR {
                let quality_trackbar = GetDlgItem(hwnd, IDC_BATCH_QUALITY_TRACKBAR);
                if !quality_trackbar.is_null() {
                    let current_pos = SendMessageW(quality_trackbar, TBM_GETPOS, 0, 0) as i32;
                    let quality_label = GetDlgItem(hwnd, IDC_BATCH_QUALITY_LABEL);
                    if !quality_label.is_null() {
                        let quality_text = match current_pos {
                            1..=20 => format!("质量: {}% (很低)", current_pos),
                            21..=40 => format!("质量: {}% (低)", current_pos),
                            41..=60 => format!("质量: {}% (中等)", current_pos),
                            61..=80 => format!("质量: {}% (高)", current_pos),
                            81..=100 => format!("质量: {}% (很高)", current_pos),
                            _ => format!("质量: {}%", current_pos),
                        };
                        let quality_text_utf16: Vec<u16> = quality_text.encode_utf16().chain(std::iter::once(0)).collect();
                        SetWindowTextW(quality_label, quality_text_utf16.as_ptr());
                    }
                }
            }
            0
        }
        WM_HSCROLL => {
            // 处理质量滑块的滚动消息
            let control_id = (wparam & 0xFFFF) as i32;
            if control_id == IDC_BATCH_QUALITY_TRACKBAR {
                let quality_trackbar = GetDlgItem(hwnd, IDC_BATCH_QUALITY_TRACKBAR);
                if !quality_trackbar.is_null() {
                    let current_pos = SendMessageW(quality_trackbar, TBM_GETPOS, 0, 0) as i32;
                    let quality_label = GetDlgItem(hwnd, IDC_BATCH_QUALITY_LABEL);
                    if !quality_label.is_null() {
                        let quality_text = match current_pos {
                            1..=20 => format!("质量: {}% (很低)", current_pos),
                            21..=40 => format!("质量: {}% (低)", current_pos),
                            41..=60 => format!("质量: {}% (中等)", current_pos),
                            61..=80 => format!("质量: {}% (高)", current_pos),
                            81..=100 => format!("质量: {}% (很高)", current_pos),
                            _ => format!("质量: {}%", current_pos),
                        };
                        let quality_text_utf16: Vec<u16> = quality_text.encode_utf16().chain(std::iter::once(0)).collect();
                        SetWindowTextW(quality_label, quality_text_utf16.as_ptr());
                    }
                }
            }
            0
        }
        WM_CLOSE => {
            println!("[batch_resize_dialog_proc] 收到 WM_CLOSE，销毁窗口");
            DestroyWindow(hwnd);
            0
        }
        WM_DESTROY => {
            println!("[batch_resize_dialog_proc] 收到 WM_DESTROY");
            0
        }
        _ => DefWindowProcW(hwnd, msg, wparam, lparam),
    }
}

// 新增：显示文件夹选择对话框
unsafe fn show_folder_dialog(hwnd: HWND) -> Option<String> {
    // 使用文件对话框来选择文件夹，通过选择文件然后获取其目录
    let mut file_name = [0u16; 260]; // MAX_PATH
    let filter = "所有文件\0*.*\0\0";
    let filter_utf16: Vec<u16> = filter.encode_utf16().collect();
    let title_utf16: Vec<u16> = "选择包含图像的文件夹\0".encode_utf16().collect();

    let mut ofn = OPENFILENAMEW {
        lStructSize: std::mem::size_of::<OPENFILENAMEW>() as u32,
        hwndOwner: hwnd,
        hInstance: null_mut(),
        lpstrFilter: filter_utf16.as_ptr(),
        lpstrCustomFilter: null_mut(),
        nMaxCustFilter: 0,
        nFilterIndex: 1,
        lpstrFile: file_name.as_mut_ptr(),
        nMaxFile: file_name.len() as u32,
        lpstrFileTitle: null_mut(),
        nMaxFileTitle: 0,
        lpstrInitialDir: null_mut(),
        lpstrTitle: title_utf16.as_ptr(),
        Flags: OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST,
        nFileOffset: 0,
        nFileExtension: 0,
        lpstrDefExt: null_mut(),
        lCustData: 0,
        lpfnHook: None,
        lpTemplateName: null_mut(),
        pvReserved: null_mut(),
        dwReserved: 0,
        FlagsEx: 0,
    };

    if GetOpenFileNameW(&mut ofn) != 0 {
        // 转换UTF-16路径为String
        let len = file_name.iter().position(|&c| c == 0).unwrap_or(file_name.len());
        let path = String::from_utf16_lossy(&file_name[..len]);

        // 获取文件所在目录
        if let Some(parent) = std::path::Path::new(&path).parent() {
            if let Some(parent_str) = parent.to_str() {
                return Some(parent_str.to_string());
            }
        }
        None
    } else {
        None
    }
}

// 新增：使用设置进行批量缩图处理
fn batch_resize_images_with_settings(settings: &BatchResizeSettings) -> Result<()> {
    use std::fs;
    use std::path::Path;
    
    println!("开始批量缩图处理...");
    println!("输入文件夹: {}", settings.input_folder);
    println!("输出文件夹: {}", settings.output_folder);
    println!("长边限制: {}", settings.long_edge_limit);
    println!("质量: {}%", settings.quality);
    println!("插值算法: {}", match settings.interpolation {
        INTER_NEAREST => "最近邻插值",
        INTER_LINEAR => "双线性插值",
        INTER_CUBIC => "双三次插值",
        INTER_AREA => "区域插值",
        INTER_LANCZOS4 => "Lanczos插值",
        _ => "未知算法",
    });
    
    // 支持的图像扩展名
    let supported_extensions = ["jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp"];

    // 读取文件夹中的所有文件
    let entries = fs::read_dir(&settings.input_folder)
        .context("无法读取输入文件夹")?;

    let mut processed_count = 0;
    let mut error_count = 0;

    // 创建输出文件夹
    let output_folder = Path::new(&settings.output_folder);
    if !output_folder.exists() {
        fs::create_dir_all(output_folder)
            .context("无法创建输出文件夹")?;
    }

    for entry in entries {
        if let Ok(entry) = entry {
            let path = entry.path();

            // 检查是否为文件
            if !path.is_file() {
                continue;
            }

            // 检查文件扩展名
            if let Some(extension) = path.extension() {
                if let Some(ext_str) = extension.to_str() {
                    if !supported_extensions.contains(&ext_str.to_lowercase().as_str()) {
                        continue;
                    }
                } else {
                    continue;
                }
            } else {
                continue;
            }

            // 处理图像文件
            match process_single_image_for_batch_with_settings(&path, output_folder, settings) {
                Ok(()) => {
                    processed_count += 1;
                    println!("✅ 处理完成: {}", path.file_name().unwrap().to_string_lossy());
                }
                Err(e) => {
                    error_count += 1;
                    println!("❌ 处理失败: {} - {}", path.file_name().unwrap().to_string_lossy(), e);
                }
            }
        }
    }

    println!("批量处理完成！成功: {}, 失败: {}", processed_count, error_count);
    
    // 显示完成消息
    let success_msg = format!("批量缩图完成！\n\n处理文件夹: {}\n输出文件夹: {}\n成功: {} 个文件\n失败: {} 个文件", 
        settings.input_folder, settings.output_folder, processed_count, error_count);
    let success_msg_utf16: Vec<u16> = success_msg.encode_utf16().chain(std::iter::once(0)).collect();
    let title_utf16: Vec<u16> = "批量处理完成".encode_utf16().chain(std::iter::once(0)).collect();
    
    unsafe {
        MessageBoxW(null_mut(), success_msg_utf16.as_ptr(), title_utf16.as_ptr(), MB_OK | MB_ICONINFORMATION);
    }
    
    Ok(())
}

// 新增：使用设置处理单个图像文件
fn process_single_image_for_batch_with_settings(
    input_path: &std::path::Path, 
    output_folder: &std::path::Path,
    settings: &BatchResizeSettings
) -> Result<()> {
    // 加载图像
    let mat = imgcodecs::imread(input_path.to_str().unwrap(), imgcodecs::IMREAD_COLOR)
        .context("无法读取图像文件")?;

    if mat.empty() {
        return Err(anyhow::anyhow!("图像文件为空"));
    }

    // 获取原始尺寸
    let original_width = mat.cols();
    let original_height = mat.rows();

    // 计算新的尺寸（保持宽高比，长边限制为指定值）
    let (new_width, new_height) = if original_width > original_height {
        if original_width > settings.long_edge_limit {
            let ratio = settings.long_edge_limit as f64 / original_width as f64;
            (settings.long_edge_limit, (original_height as f64 * ratio) as i32)
        } else {
            (original_width, original_height)
        }
    } else {
        if original_height > settings.long_edge_limit {
            let ratio = settings.long_edge_limit as f64 / original_height as f64;
            ((original_width as f64 * ratio) as i32, settings.long_edge_limit)
        } else {
            (original_width, original_height)
        }
    };

    // 调整图像大小
    let resized = if new_width != original_width || new_height != original_height {
        let mut resized = core::Mat::default();
        imgproc::resize(&mat, &mut resized, core::Size::new(new_width, new_height), 0.0, 0.0, settings.interpolation);
        resized
    } else {
        mat.clone()
    };

    // 生成输出文件名
    let file_name = input_path.file_name().unwrap();
    let output_path = output_folder.join(file_name);

    // 保存调整后的图像
    let output_path_str = output_path.to_str().unwrap();
    if !imgcodecs::imwrite(output_path_str, &resized, &core::Vector::new()).unwrap_or(false) {
        return Err(anyhow::anyhow!("无法保存调整后的图像"));
    }

    Ok(())
}

// ============== EXIF面板相关函数 ==============

/// 从文件读取EXIF数据
fn read_exif_from_file(file_path: &str) -> Result<ExifData> {
    use std::fs::File;
    use exif::{In, Reader, Tag};
    
    let file = File::open(file_path)
        .context("无法打开图像文件")?;
    let mut bufreader = std::io::BufReader::new(&file);
    let exifreader = Reader::new();
    let exif = exifreader.read_from_container(&mut bufreader)
        .context("无法读取EXIF数据")?;
    
    let mut data = ExifData::default();
    
    // 读取相机型号
    if let Some(field) = exif.get_field(Tag::Model, In::PRIMARY) {
        data.camera_model = field.display_value().to_string().trim().to_string();
    }
    
    // 读取光圈值 (FNumber)
    if let Some(field) = exif.get_field(Tag::FNumber, In::PRIMARY) {
        let value = field.display_value().to_string();
        data.aperture = format!("f/{}", value);
    }
    
    // 读取快门速度 (ExposureTime)
    if let Some(field) = exif.get_field(Tag::ExposureTime, In::PRIMARY) {
        data.shutter_speed = field.display_value().to_string();
    }
    
    // 读取ISO
    if let Some(field) = exif.get_field(Tag::PhotographicSensitivity, In::PRIMARY) {
        data.iso = format!("ISO-{}", field.display_value());
    }
    
    // 读取焦距
    if let Some(field) = exif.get_field(Tag::FocalLength, In::PRIMARY) {
        let value = field.display_value().to_string();
        data.focal_length = format!("{} mm", value.replace(" mm", ""));
    }
    
    // 读取色彩空间
    if let Some(field) = exif.get_field(Tag::ColorSpace, In::PRIMARY) {
        let value = field.value.get_uint(0).unwrap_or(0);
        data.color_space = match value {
            1 => "sRGB".to_string(),
            2 => "Adobe RGB".to_string(),
            65535 => "未校准".to_string(),
            _ => format!("未知({})", value),
        };
    }
    
    Ok(data)
}

// EXIF面板控件ID
const IDC_EXIF_CAMERA: i32 = 3001;
const IDC_EXIF_APERTURE: i32 = 3002;
const IDC_EXIF_SHUTTER: i32 = 3003;
const IDC_EXIF_ISO: i32 = 3004;
const IDC_EXIF_FOCAL: i32 = 3005;
const IDC_EXIF_COLORSPACE: i32 = 3006;

/// EXIF面板窗口过程
unsafe extern "system" fn exif_panel_wnd_proc(
    hwnd: HWND,
    msg: UINT,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    match msg {
        WM_CREATE => {
            // 创建Static控件显示EXIF信息
            let font = CreateFontW(
                22, 0, 0, 0, FW_NORMAL, 0, 0, 0,
                DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
                DEFAULT_QUALITY, DEFAULT_PITCH | FF_DONTCARE,
                "Microsoft YaHei\0".encode_utf16().collect::<Vec<u16>>().as_ptr(),
            );
            
            let mut y_pos = 15;
            let label_width = 90;
            let value_width = 240;
            let height = 30;
            let spacing = 38;
            
            // 相机型号标签和值
            let camera_label_text = "相机型号:\0".encode_utf16().collect::<Vec<u16>>();
            CreateWindowExW(
                0,
                "STATIC\0".encode_utf16().collect::<Vec<u16>>().as_ptr(),
                camera_label_text.as_ptr(),
                WS_CHILD | WS_VISIBLE,
                10, y_pos, label_width, height,
                hwnd,
                0 as HMENU,
                GetModuleHandleW(null_mut()),
                null_mut(),
            );
            
            let camera_value = CreateWindowExW(
                0,
                "STATIC\0".encode_utf16().collect::<Vec<u16>>().as_ptr(),
                "-\0".encode_utf16().collect::<Vec<u16>>().as_ptr(),
                WS_CHILD | WS_VISIBLE,
                100, y_pos, value_width, height,
                hwnd,
                IDC_EXIF_CAMERA as HMENU,
                GetModuleHandleW(null_mut()),
                null_mut(),
            );
            SendMessageW(camera_value, WM_SETFONT, font as WPARAM, 1);
            y_pos += spacing;
            
            // 光圈值
            let aperture_label_text = "光圈值:\0".encode_utf16().collect::<Vec<u16>>();
            CreateWindowExW(
                0,
                "STATIC\0".encode_utf16().collect::<Vec<u16>>().as_ptr(),
                aperture_label_text.as_ptr(),
                WS_CHILD | WS_VISIBLE,
                10, y_pos, label_width, height,
                hwnd,
                0 as HMENU,
                GetModuleHandleW(null_mut()),
                null_mut(),
            );
            
            let aperture_value = CreateWindowExW(
                0,
                "STATIC\0".encode_utf16().collect::<Vec<u16>>().as_ptr(),
                "-\0".encode_utf16().collect::<Vec<u16>>().as_ptr(),
                WS_CHILD | WS_VISIBLE,
                100, y_pos, value_width, height,
                hwnd,
                IDC_EXIF_APERTURE as HMENU,
                GetModuleHandleW(null_mut()),
                null_mut(),
            );
            SendMessageW(aperture_value, WM_SETFONT, font as WPARAM, 1);
            y_pos += spacing;
            
            // 曝光时间
            let shutter_label_text = "曝光时间:\0".encode_utf16().collect::<Vec<u16>>();
            CreateWindowExW(
                0,
                "STATIC\0".encode_utf16().collect::<Vec<u16>>().as_ptr(),
                shutter_label_text.as_ptr(),
                WS_CHILD | WS_VISIBLE,
                10, y_pos, label_width, height,
                hwnd,
                0 as HMENU,
                GetModuleHandleW(null_mut()),
                null_mut(),
            );
            
            let shutter_value = CreateWindowExW(
                0,
                "STATIC\0".encode_utf16().collect::<Vec<u16>>().as_ptr(),
                "-\0".encode_utf16().collect::<Vec<u16>>().as_ptr(),
                WS_CHILD | WS_VISIBLE,
                100, y_pos, value_width, height,
                hwnd,
                IDC_EXIF_SHUTTER as HMENU,
                GetModuleHandleW(null_mut()),
                null_mut(),
            );
            SendMessageW(shutter_value, WM_SETFONT, font as WPARAM, 1);
            y_pos += spacing;
            
            // ISO速度
            let iso_label_text = "ISO速度:\0".encode_utf16().collect::<Vec<u16>>();
            CreateWindowExW(
                0,
                "STATIC\0".encode_utf16().collect::<Vec<u16>>().as_ptr(),
                iso_label_text.as_ptr(),
                WS_CHILD | WS_VISIBLE,
                10, y_pos, label_width, height,
                hwnd,
                0 as HMENU,
                GetModuleHandleW(null_mut()),
                null_mut(),
            );
            
            let iso_value = CreateWindowExW(
                0,
                "STATIC\0".encode_utf16().collect::<Vec<u16>>().as_ptr(),
                "-\0".encode_utf16().collect::<Vec<u16>>().as_ptr(),
                WS_CHILD | WS_VISIBLE,
                100, y_pos, value_width, height,
                hwnd,
                IDC_EXIF_ISO as HMENU,
                GetModuleHandleW(null_mut()),
                null_mut(),
            );
            SendMessageW(iso_value, WM_SETFONT, font as WPARAM, 1);
            y_pos += spacing;
            
            // 焦距
            let focal_label_text = "焦距:\0".encode_utf16().collect::<Vec<u16>>();
            CreateWindowExW(
                0,
                "STATIC\0".encode_utf16().collect::<Vec<u16>>().as_ptr(),
                focal_label_text.as_ptr(),
                WS_CHILD | WS_VISIBLE,
                10, y_pos, label_width, height,
                hwnd,
                0 as HMENU,
                GetModuleHandleW(null_mut()),
                null_mut(),
            );
            
            let focal_value = CreateWindowExW(
                0,
                "STATIC\0".encode_utf16().collect::<Vec<u16>>().as_ptr(),
                "-\0".encode_utf16().collect::<Vec<u16>>().as_ptr(),
                WS_CHILD | WS_VISIBLE,
                100, y_pos, value_width, height,
                hwnd,
                IDC_EXIF_FOCAL as HMENU,
                GetModuleHandleW(null_mut()),
                null_mut(),
            );
            SendMessageW(focal_value, WM_SETFONT, font as WPARAM, 1);
            y_pos += spacing;
            
            // 色彩空间
            let colorspace_label_text = "色彩空间:\0".encode_utf16().collect::<Vec<u16>>();
            CreateWindowExW(
                0,
                "STATIC\0".encode_utf16().collect::<Vec<u16>>().as_ptr(),
                colorspace_label_text.as_ptr(),
                WS_CHILD | WS_VISIBLE,
                10, y_pos, label_width, height,
                hwnd,
                0 as HMENU,
                GetModuleHandleW(null_mut()),
                null_mut(),
            );
            
            let colorspace_value = CreateWindowExW(
                0,
                "STATIC\0".encode_utf16().collect::<Vec<u16>>().as_ptr(),
                "-\0".encode_utf16().collect::<Vec<u16>>().as_ptr(),
                WS_CHILD | WS_VISIBLE,
                100, y_pos, value_width, height,
                hwnd,
                IDC_EXIF_COLORSPACE as HMENU,
                GetModuleHandleW(null_mut()),
                null_mut(),
            );
            SendMessageW(colorspace_value, WM_SETFONT, font as WPARAM, 1);
            
            0
        }
        WM_CLOSE => {
            // 隐藏窗口而不是销毁
            ShowWindow(hwnd, SW_HIDE);
            EXIF_PANEL_VISIBLE = false;
            // 更新主窗口菜单的勾选状态
            update_exif_menu_check();
            0
        }
        WM_DESTROY => {
            EXIF_PANEL_HWND = None;
            0
        }
        _ => DefWindowProcW(hwnd, msg, wparam, lparam),
    }
}

/// 创建EXIF面板窗口
unsafe fn create_exif_panel_window(main_hwnd: HWND) -> HWND {
    // 注册窗口类
    let class_name = "ExifPanelClass\0".encode_utf16().collect::<Vec<u16>>();
    
    let wc = WNDCLASSW {
        style: 0,
        lpfnWndProc: Some(exif_panel_wnd_proc),
        cbClsExtra: 0,
        cbWndExtra: 0,
        hInstance: GetModuleHandleW(null_mut()),
        hIcon: null_mut(),
        hCursor: LoadCursorW(null_mut(), IDC_ARROW),
        hbrBackground: (COLOR_WINDOW + 1) as HBRUSH,
        lpszMenuName: null_mut(),
        lpszClassName: class_name.as_ptr(),
    };
    
    RegisterClassW(&wc);
    
    // 创建窗口
    let window_title = "拍摄信息\0".encode_utf16().collect::<Vec<u16>>();
    let panel_width = 360;
    let panel_height = 290;
    
    // 获取主窗口位置
    let mut main_rect: RECT = std::mem::zeroed();
    GetWindowRect(main_hwnd, &mut main_rect);
    
    // 放在主窗口右侧
    let x = main_rect.right + 10;
    let y = main_rect.top;
    
    let hwnd = CreateWindowExW(
        0,
        class_name.as_ptr(),
        window_title.as_ptr(),
        WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_VISIBLE,
        x, y,
        panel_width, panel_height,
        main_hwnd,
        null_mut(),
        GetModuleHandleW(null_mut()),
        null_mut(),
    );
    
    hwnd
}

/// 更新EXIF面板内容
unsafe fn update_exif_panel_content(panel_hwnd: HWND) {
    let state = IMAGE_STATE.as_ref().unwrap();
    if let Some(ref path) = state.current_file_path {
        match read_exif_from_file(path) {
            Ok(exif_data) => {
                // 更新窗口标题显示文件名
                let filename = std::path::Path::new(path)
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("未知");
                let title = format!("拍摄信息 - {}\0", filename);
                let title_utf16: Vec<u16> = title.encode_utf16().collect();
                SetWindowTextW(panel_hwnd, title_utf16.as_ptr());
                
                // 更新各个Static控件的文本
                let camera_text = if exif_data.camera_model.is_empty() {
                    "-\0".to_string()
                } else {
                    format!("{}\0", exif_data.camera_model)
                };
                let camera_utf16: Vec<u16> = camera_text.encode_utf16().collect();
                let camera_hwnd = GetDlgItem(panel_hwnd, IDC_EXIF_CAMERA);
                SetWindowTextW(camera_hwnd, camera_utf16.as_ptr());
                
                let aperture_text = if exif_data.aperture.is_empty() {
                    "-\0".to_string()
                } else {
                    format!("{}\0", exif_data.aperture)
                };
                let aperture_utf16: Vec<u16> = aperture_text.encode_utf16().collect();
                let aperture_hwnd = GetDlgItem(panel_hwnd, IDC_EXIF_APERTURE);
                SetWindowTextW(aperture_hwnd, aperture_utf16.as_ptr());
                
                let shutter_text = if exif_data.shutter_speed.is_empty() {
                    "-\0".to_string()
                } else {
                    format!("{}\0", exif_data.shutter_speed)
                };
                let shutter_utf16: Vec<u16> = shutter_text.encode_utf16().collect();
                let shutter_hwnd = GetDlgItem(panel_hwnd, IDC_EXIF_SHUTTER);
                SetWindowTextW(shutter_hwnd, shutter_utf16.as_ptr());
                
                let iso_text = if exif_data.iso.is_empty() {
                    "-\0".to_string()
                } else {
                    format!("{}\0", exif_data.iso)
                };
                let iso_utf16: Vec<u16> = iso_text.encode_utf16().collect();
                let iso_hwnd = GetDlgItem(panel_hwnd, IDC_EXIF_ISO);
                SetWindowTextW(iso_hwnd, iso_utf16.as_ptr());
                
                let focal_text = if exif_data.focal_length.is_empty() {
                    "-\0".to_string()
                } else {
                    format!("{}\0", exif_data.focal_length)
                };
                let focal_utf16: Vec<u16> = focal_text.encode_utf16().collect();
                let focal_hwnd = GetDlgItem(panel_hwnd, IDC_EXIF_FOCAL);
                SetWindowTextW(focal_hwnd, focal_utf16.as_ptr());
                
                // 更新色彩空间
                let colorspace_text = if exif_data.color_space.is_empty() {
                    "-\0".to_string()
                } else {
                    format!("{}\0", exif_data.color_space)
                };
                let colorspace_utf16: Vec<u16> = colorspace_text.encode_utf16().collect();
                let colorspace_hwnd = GetDlgItem(panel_hwnd, IDC_EXIF_COLORSPACE);
                SetWindowTextW(colorspace_hwnd, colorspace_utf16.as_ptr());
                
                println!("✅ EXIF信息已更新");
            }
            Err(e) => {
                // 显示"无EXIF数据"
                println!("⚠️ 无法读取EXIF: {}", e);
                let no_exif = "无EXIF数据\0".encode_utf16().collect::<Vec<u16>>();
                let camera_hwnd = GetDlgItem(panel_hwnd, IDC_EXIF_CAMERA);
                SetWindowTextW(camera_hwnd, no_exif.as_ptr());
            }
        }
    }
}

/// 如果EXIF面板可见，则更新内容
unsafe fn update_exif_panel_if_visible() {
    if EXIF_PANEL_VISIBLE {
        if let Some(panel_hwnd) = EXIF_PANEL_HWND {
            update_exif_panel_content(panel_hwnd);
        }
    }
}

/// 切换EXIF面板显示/隐藏
unsafe fn toggle_exif_panel(main_hwnd: HWND) {
    if EXIF_PANEL_VISIBLE {
        // 关闭面板
        if let Some(panel_hwnd) = EXIF_PANEL_HWND {
            ShowWindow(panel_hwnd, SW_HIDE);
        }
        EXIF_PANEL_VISIBLE = false;
        println!("⏹️ EXIF面板已隐藏");
    } else {
        // 显示面板
        if EXIF_PANEL_HWND.is_none() {
            // 首次创建窗口
            EXIF_PANEL_HWND = Some(create_exif_panel_window(main_hwnd));
        }
        
        if let Some(panel_hwnd) = EXIF_PANEL_HWND {
            // 更新EXIF数据
            update_exif_panel_content(panel_hwnd);
            ShowWindow(panel_hwnd, SW_SHOW);
            SetForegroundWindow(panel_hwnd);
        }
        EXIF_PANEL_VISIBLE = true;
        println!("▶️ EXIF面板已显示");
    }
    
    // 更新菜单勾选状态
    update_exif_menu_check();
}

/// 更新菜单的勾选状态
unsafe fn update_exif_menu_check() {
    if let Some(main_hwnd) = MAIN_HWND {
        let menu = GetMenu(main_hwnd);
        if !menu.is_null() {
            let check_flag = if EXIF_PANEL_VISIBLE { MF_CHECKED } else { MF_UNCHECKED };
            CheckMenuItem(menu, MENU_EXIF_PANEL, check_flag);
        }
    }
}