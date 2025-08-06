use std::ptr::null_mut;
use winapi::um::winuser::*;
use winapi::um::libloaderapi::GetModuleHandleW;
use winapi::um::wingdi::*;
use winapi::um::commdlg::*;
use winapi::shared::windef::*;
use winapi::shared::minwindef::*;
use winapi::um::shellscalingapi::*;
use opencv::{core, imgcodecs};
use opencv::prelude::*;
use anyhow::{Result, Context};

mod ocv_bridge;
mod cv_ops;

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
const MENU_RESTORE: u32 = 1009;

// 全局变量存储图像数据
static mut IMAGE_STATE: Option<ImageState> = None;

struct ImageState {
    original_bgr: Option<core::Mat>,
    display_pixels: Vec<u8>,
    current_width: i32,
    current_height: i32,
}

impl ImageState {
    fn new() -> Self {
        Self {
            original_bgr: None,
            display_pixels: Vec::new(),
            current_width: 0,
            current_height: 0,
        }
    }
    
    fn load_image(&mut self, path: &str) -> Result<()> {
        println!("加载图像: {}", path);
        
        // 检查文件是否存在
        if !std::path::Path::new(path).exists() {
            return Err(anyhow::anyhow!("文件不存在: {}", path));
        }
        
        // 尝试使用OpenCV读取图像为BGR格式
        let bgr = match imgcodecs::imread(path, imgcodecs::IMREAD_COLOR) {
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
        
        println!("图像尺寸: {}x{}, 通道数: {}", bgr.cols(), bgr.rows(), bgr.channels());
        
        // 转换为显示像素
        let image_buf = ocv_bridge::mat_to_bgra(&bgr)?;
        
        self.original_bgr = Some(bgr);
        self.display_pixels = image_buf.pixels;
        self.current_width = image_buf.w;
        self.current_height = image_buf.h;
        
        println!("图像加载成功");
        Ok(())
    }
    
    fn load_image_fallback(&mut self, path: &str) -> Result<()> {
        println!("使用备用方法加载图像...");
        
        // 使用标准库读取文件到内存
        let image_data = std::fs::read(path)
            .context("无法读取图像文件到内存")?;
        
        // 使用OpenCV从内存数据解码图像
        let data_vec = opencv::core::Vector::from_slice(&image_data);
        let bgr = imgcodecs::imdecode(&data_vec, imgcodecs::IMREAD_COLOR)
            .context("无法解码图像数据")?;
        
        if bgr.empty() {
            return Err(anyhow::anyhow!("解码后的图像为空，可能是不支持的格式"));
        }
        
        println!("图像尺寸: {}x{}, 通道数: {}", bgr.cols(), bgr.rows(), bgr.channels());
        
        // 转换为显示像素
        let image_buf = ocv_bridge::mat_to_bgra(&bgr)?;
        
        self.original_bgr = Some(bgr);
        self.display_pixels = image_buf.pixels;
        self.current_width = image_buf.w;
        self.current_height = image_buf.h;
        
        println!("图像加载成功（备用方法）");
        Ok(())
    }
    
    fn update_display_from_mat(&mut self, mat: &core::Mat) -> Result<()> {
        let image_buf = ocv_bridge::mat_to_bgra(mat)?;
        self.display_pixels = image_buf.pixels;
        self.current_width = image_buf.w;
        self.current_height = image_buf.h;
        Ok(())
    }
    
    fn has_image(&self) -> bool {
        self.original_bgr.is_some()
    }
}

// 创建菜单
unsafe fn create_menu() -> HMENU {
    let menu_bar = CreateMenu();
    
    // 创建文件菜单
    let file_menu = CreatePopupMenu();
    let open_file_text = "打开文件(&O)...";
    let open_file_utf16: Vec<u16> = open_file_text.encode_utf16().chain(std::iter::once(0)).collect();
    AppendMenuW(file_menu, MF_STRING, MENU_OPEN_FILE as usize, open_file_utf16.as_ptr());
    
    // 创建处理菜单
    let process_menu = CreatePopupMenu();
    
    // 基础处理
    let to_gray_text = "转换为灰度图(&G)";
    let to_gray_utf16: Vec<u16> = to_gray_text.encode_utf16().chain(std::iter::once(0)).collect();
    AppendMenuW(process_menu, MF_STRING, MENU_TO_GRAY as usize, to_gray_utf16.as_ptr());
    
    let gaussian_text = "高斯模糊(&B)";
    let gaussian_utf16: Vec<u16> = gaussian_text.encode_utf16().chain(std::iter::once(0)).collect();
    AppendMenuW(process_menu, MF_STRING, MENU_GAUSSIAN as usize, gaussian_utf16.as_ptr());
    
    AppendMenuW(process_menu, MF_SEPARATOR, 0, null_mut());
    
    // 边缘检测
    let canny_text = "Canny边缘检测(&C)";
    let canny_utf16: Vec<u16> = canny_text.encode_utf16().chain(std::iter::once(0)).collect();
    AppendMenuW(process_menu, MF_STRING, MENU_CANNY as usize, canny_utf16.as_ptr());
    
    let threshold_text = "自适应阈值(&T)";
    let threshold_utf16: Vec<u16> = threshold_text.encode_utf16().chain(std::iter::once(0)).collect();
    AppendMenuW(process_menu, MF_STRING, MENU_ADAPTIVE_THRESHOLD as usize, threshold_utf16.as_ptr());
    
    AppendMenuW(process_menu, MF_SEPARATOR, 0, null_mut());
    
    // 形态学操作
    let morph_text = "形态学闭运算(&M)";
    let morph_utf16: Vec<u16> = morph_text.encode_utf16().chain(std::iter::once(0)).collect();
    AppendMenuW(process_menu, MF_STRING, MENU_MORPH_CLOSE as usize, morph_utf16.as_ptr());
    
    let dilate_text = "膨胀操作(&D)";
    let dilate_utf16: Vec<u16> = dilate_text.encode_utf16().chain(std::iter::once(0)).collect();
    AppendMenuW(process_menu, MF_STRING, MENU_DILATE as usize, dilate_utf16.as_ptr());
    
    AppendMenuW(process_menu, MF_SEPARATOR, 0, null_mut());
    
    // 轮廓检测
    let contours_text = "轮廓检测(&F)";
    let contours_utf16: Vec<u16> = contours_text.encode_utf16().chain(std::iter::once(0)).collect();
    AppendMenuW(process_menu, MF_STRING, MENU_FIND_CONTOURS as usize, contours_utf16.as_ptr());
    
    // 透视变换
    let perspective_text = "透视变换(简化)(&P)";
    let perspective_utf16: Vec<u16> = perspective_text.encode_utf16().chain(std::iter::once(0)).collect();
    AppendMenuW(process_menu, MF_STRING, MENU_PERSPECTIVE_TRANSFORM as usize, perspective_utf16.as_ptr());
    
    AppendMenuW(process_menu, MF_SEPARATOR, 0, null_mut());
    
    // 恢复原图
    let restore_text = "恢复原图(&R)";
    let restore_utf16: Vec<u16> = restore_text.encode_utf16().chain(std::iter::once(0)).collect();
    AppendMenuW(process_menu, MF_STRING, MENU_RESTORE as usize, restore_utf16.as_ptr());
    
    // 添加菜单到菜单栏
    let file_text = "文件(&F)";
    let file_utf16: Vec<u16> = file_text.encode_utf16().chain(std::iter::once(0)).collect();
    AppendMenuW(menu_bar, MF_POPUP, file_menu as usize, file_utf16.as_ptr());
    
    let process_text = "处理(&P)";
    let process_utf16: Vec<u16> = process_text.encode_utf16().chain(std::iter::once(0)).collect();
    AppendMenuW(menu_bar, MF_POPUP, process_menu as usize, process_utf16.as_ptr());
    
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

// 处理菜单命令
unsafe fn handle_menu_command(hwnd: HWND, command_id: u32) {
    let state = IMAGE_STATE.as_mut().unwrap();
    
    match command_id {
        MENU_OPEN_FILE => {
            if let Some(file_path) = show_open_file_dialog(hwnd) {
                match state.load_image(&file_path) {
                    Ok(()) => {
                        println!("✅ 成功加载图像: {}", file_path);
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
        MENU_TO_GRAY => {
            if let Some(ref original) = state.original_bgr.clone() {
                match cv_ops::to_gray(&original) {
                    Ok(gray) => {
                        if let Ok(()) = state.update_display_from_mat(&gray) {
                            println!("✅ 灰度转换完成");
                            InvalidateRect(hwnd, null_mut(), 1);
                        }
                    }
                    Err(e) => println!("❌ 灰度转换失败: {}", e),
                }
            }
        }
        MENU_GAUSSIAN => {
            if let Some(ref original) = state.original_bgr.clone() {
                match cv_ops::to_gray(&original).and_then(|gray| cv_ops::gaussian(&gray, 5, 2.0)) {
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
            if let Some(ref original) = state.original_bgr.clone() {
                match cv_ops::to_gray(&original)
                    .and_then(|gray| cv_ops::gaussian(&gray, 5, 0.0))
                    .and_then(|blurred| cv_ops::canny(&blurred, 50.0, 150.0)) {
                    Ok(edges) => {
                        if let Ok(()) = state.update_display_from_mat(&edges) {
                            println!("✅ Canny边缘检测完成");
                            InvalidateRect(hwnd, null_mut(), 1);
                        }
                    }
                    Err(e) => println!("❌ Canny边缘检测失败: {}", e),
                }
            }
        }
        MENU_ADAPTIVE_THRESHOLD => {
            if let Some(ref original) = state.original_bgr.clone() {
                match cv_ops::to_gray(&original)
                    .and_then(|gray| cv_ops::adaptive_threshold(&gray, 255.0, 11, 2.0)) {
                    Ok(binary) => {
                        if let Ok(()) = state.update_display_from_mat(&binary) {
                            println!("✅ 自适应阈值处理完成");
                            InvalidateRect(hwnd, null_mut(), 1);
                        }
                    }
                    Err(e) => println!("❌ 自适应阈值处理失败: {}", e),
                }
            }
        }
        MENU_MORPH_CLOSE => {
            if let Some(ref original) = state.original_bgr.clone() {
                match cv_ops::to_gray(&original)
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
            if let Some(ref original) = state.original_bgr.clone() {
                match cv_ops::to_gray(&original)
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
            if let Some(ref original) = state.original_bgr.clone() {
                match cv_ops::to_gray(&original)
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
            if let Some(ref original) = state.original_bgr.clone() {
                println!("开始简化透视变换...");
                match cv_ops::perspective_transform_simplified(&original) {
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
        MENU_RESTORE => {
            if let Some(ref original) = state.original_bgr.clone() {
                if let Ok(()) = state.update_display_from_mat(&original) {
                    println!("✅ 恢复原图完成");
                    InvalidateRect(hwnd, null_mut(), 1);
                }
            }
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
        WM_SIZE => {
            // 窗口大小改变时重绘
            InvalidateRect(hwnd, null_mut(), 1);
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
            
            // 填充背景为黑色
            let black_brush = CreateSolidBrush(RGB(0, 0, 0));
            FillRect(hdc, &rect, black_brush);
            DeleteObject(black_brush as *mut _);
            
            // 绘制图片
            if let Some(ref state) = IMAGE_STATE {
                if state.has_image() && !state.display_pixels.is_empty() {
                    let (x, y, scaled_width, scaled_height) = calculate_scaled_size(
                        state.current_width,
                        state.current_height,
                        window_width,
                        window_height,
                    );
                    
                    // 设置高质量拉伸模式 - HALFTONE提供最佳质量
                    SetStretchBltMode(hdc, HALFTONE);
                    
                    // 设置画刷原点以改善抖动效果
                    SetBrushOrgEx(hdc, 0, 0, std::ptr::null_mut());
                    
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
                        hdc,
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
                    
                    // 更新窗口标题显示图像信息
                    let info_text = format!("图像查看器 - {}x{} - OpenCV处理", 
                        state.current_width, 
                        state.current_height
                    );
                    let title_utf16: Vec<u16> = info_text.encode_utf16().chain(std::iter::once(0)).collect();
                    SetWindowTextW(hwnd, title_utf16.as_ptr());
                } else {
                    // 显示提示信息
                    let hint_text = "请使用菜单 文件->打开文件 来加载图像";
                    let text_utf16: Vec<u16> = hint_text.encode_utf16().collect();
                    
                    SetTextColor(hdc, RGB(255, 255, 255));
                    SetBkMode(hdc, TRANSPARENT as i32);
                    
                    DrawTextW(
                        hdc,
                        text_utf16.as_ptr(),
                        text_utf16.len() as i32,
                        &mut rect,
                        DT_CENTER | DT_VCENTER | DT_SINGLELINE,
                    );
                }
            }
            
            EndPaint(hwnd, &ps);
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
        let window_title = "图像处理查看器 - OpenCV + Win32";
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

        // 显示和更新窗口
        ShowWindow(hwnd, SW_SHOW);
        UpdateWindow(hwnd);

        println!("✅ 程序启动成功！");
        println!("📁 使用菜单 文件->打开文件 来加载图像");
        println!("🎨 支持的功能:");
        println!("   - 灰度转换");
        println!("   - 高斯模糊");
        println!("   - Canny边缘检测");
        println!("   - 自适应阈值");
        println!("   - 形态学闭运算");
        println!("   - 膨胀操作");
        println!("   - 轮廓检测");
        println!("   - 透视变换(简化色卡检测)");
        println!("🐛 调试图像会保存到 ./debug_out/ 目录");

        // 消息循环
        let mut msg: MSG = std::mem::zeroed();
        while GetMessageW(&mut msg, null_mut(), 0, 0) > 0 {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
    }
}