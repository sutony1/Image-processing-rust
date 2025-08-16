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
mod sam_core;

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
const MENU_EXTRACT_COLOR_BLOCKS: u32 = 1009;
const MENU_TEMPLATE_MATCHING: u32 = 1010;
const MENU_RESTORE: u32 = 1011;
const MENU_SAVE_FILE: u32 = 1012;
// 添加SAM相关菜单项
const MENU_SAM_ENCODE: u32 = 1013;
const MENU_SAM_SEGMENT: u32 = 1014;
const MENU_SAM_EXTRACT_MASK: u32 = 1015; // 新增：抠图并显示
// 新增：选框工具和裁剪菜单
const MENU_SELECTION_TOOL: u32 = 1016; // 选框工具
const MENU_CROP_SELECTION: u32 = 1017; // 裁剪选框区域

// 全局变量存储图像数据
static mut IMAGE_STATE: Option<ImageState> = None;

struct ImageState {
    original_bgr: Option<core::Mat>,
    current_mat: Option<core::Mat>,
    display_pixels: Vec<u8>,
    current_width: i32,
    current_height: i32,
    sam_processor: Option<sam_core::SAMProcessor>, // 新增
    selecting: bool,
    select_start: Option<(i32,i32)>,
    select_end: Option<(i32,i32)>,
    current_mask: Option<(Vec<u8>, i32, i32)>, // 新增：存储当前mask信息
    // 新增：选框工具相关状态
    selection_mode: bool, // 是否处于选框模式
    selection_start: Option<(i32,i32)>, // 选框起始点
    selection_end: Option<(i32,i32)>, // 选框结束点
}

impl ImageState {
    fn new() -> Self {
        Self {
            original_bgr: None,
            current_mat: None,
            display_pixels: Vec::new(),
            current_width: 0,
            current_height: 0,
            sam_processor: None, // 新增
            selecting: false,
            select_start: None,
            select_end: None,
            current_mask: None, // 新增
            selection_mode: false, // 新增：选框模式
            selection_start: None, // 新增：选框起始点
            selection_end: None, // 新增：选框结束点
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
        
        let bgr_clone = bgr.try_clone()?;
        self.original_bgr = Some(bgr);
        self.current_mat = Some(bgr_clone);
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
        
        let bgr_clone = bgr.try_clone()?;
        self.original_bgr = Some(bgr);
        self.current_mat = Some(bgr_clone);
        self.display_pixels = image_buf.pixels;
        self.current_width = image_buf.w;
        self.current_height = image_buf.h;
        
        println!("图像加载成功（备用方法）");
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
    let save_file_text = "保存文件(&S)...";
    let save_file_utf16: Vec<u16> = save_file_text.encode_utf16().chain(std::iter::once(0)).collect();
    AppendMenuW(file_menu, MF_STRING, MENU_SAVE_FILE as usize, save_file_utf16.as_ptr());
    
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
    
    // 色卡分析
    let template_match_text = "模板匹配定位色卡(&T)";
    let template_match_utf16: Vec<u16> = template_match_text.encode_utf16().chain(std::iter::once(0)).collect();
    AppendMenuW(process_menu, MF_STRING, MENU_TEMPLATE_MATCHING as usize, template_match_utf16.as_ptr());
    
    let extract_color_text = "读取色卡色块颜色(&E)";
    let extract_color_utf16: Vec<u16> = extract_color_text.encode_utf16().chain(std::iter::once(0)).collect();
    AppendMenuW(process_menu, MF_STRING, MENU_EXTRACT_COLOR_BLOCKS as usize, extract_color_utf16.as_ptr());
    
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
    
    // 添加SAM菜单
    let sam_menu = CreatePopupMenu();
    let sam_encode_text = "SAM编码图像(&E)";
    let sam_encode_utf16: Vec<u16> = sam_encode_text.encode_utf16().chain(std::iter::once(0)).collect();
    AppendMenuW(sam_menu, MF_STRING, MENU_SAM_ENCODE as usize, sam_encode_utf16.as_ptr());
    
    let sam_segment_text = "SAM交互分割(&S)";
    let sam_segment_utf16: Vec<u16> = sam_segment_text.encode_utf16().chain(std::iter::once(0)).collect();
    AppendMenuW(sam_menu, MF_STRING, MENU_SAM_SEGMENT as usize, sam_segment_utf16.as_ptr());
    
    // 新增：抠图并显示命令
    let sam_extract_text = "抠图并显示(&E)";
    let sam_extract_utf16: Vec<u16> = sam_extract_text.encode_utf16().chain(std::iter::once(0)).collect();
    AppendMenuW(sam_menu, MF_STRING, MENU_SAM_EXTRACT_MASK as usize, sam_extract_utf16.as_ptr());
    
    let sam_text = "SAM分割(&S)";
    let sam_utf16: Vec<u16> = sam_text.encode_utf16().chain(std::iter::once(0)).collect();
    AppendMenuW(menu_bar, MF_POPUP, sam_menu as usize, sam_utf16.as_ptr());
    
    // 新增：选框工具和裁剪菜单
    let selection_menu = CreatePopupMenu();
    let selection_tool_text = "选框工具(&S)";
    let selection_tool_utf16: Vec<u16> = selection_tool_text.encode_utf16().chain(std::iter::once(0)).collect();
    AppendMenuW(selection_menu, MF_STRING, MENU_SELECTION_TOOL as usize, selection_tool_utf16.as_ptr());
    
    let crop_selection_text = "裁剪选框区域(&C)";
    let crop_selection_utf16: Vec<u16> = crop_selection_text.encode_utf16().chain(std::iter::once(0)).collect();
    AppendMenuW(selection_menu, MF_STRING, MENU_CROP_SELECTION as usize, crop_selection_utf16.as_ptr());
    
    let selection_text = "选框工具(&S)";
    let selection_utf16: Vec<u16> = selection_text.encode_utf16().chain(std::iter::once(0)).collect();
    AppendMenuW(menu_bar, MF_POPUP, selection_menu as usize, selection_utf16.as_ptr());
    
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
    let mut color_overlay = cvcore::Mat::new_rows_cols_with_default(state.current_height, state.current_width, cvcore::CV_8UC3, cvcore::Scalar::new(0.0, 255.0, 0.0, 0.0))?; // 绿色
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
            let pixel_ptr = unsafe { cropped_img.ptr_2d(y, x)? };
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
                let src_pixel_ptr = unsafe { cropped_img.ptr_2d(y, x)? };
                let dst_pixel_ptr = unsafe { result.ptr_2d_mut(y, x)? };
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
            let pixel_ptr = unsafe { result.ptr_2d(y, x)? };
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
        MENU_SAVE_FILE => {
            if let Some(ref mat) = state.current_mat {
                if let Some(path) = show_save_file_dialog(hwnd) {
                    match save_mat_with_unicode_path(mat, &path) {
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
        MENU_TO_GRAY => {
            if let Some(ref current) = state.current_mat.clone() {
                match cv_ops::to_gray(&current) {
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
                println!("开始简化透视变换...");
                match cv_ops::perspective_transform_simplified(&current) {
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
                    println!("✅ 恢复原图完成");
                    InvalidateRect(hwnd, null_mut(), 1);
                }
            }
        }
        MENU_SAM_ENCODE => {
            if let Some(ref current) = state.current_mat.clone() {
                match state.init_sam_processor() {
                    Ok(()) => {
                        if let Some(ref mut sam) = state.sam_processor {
                            match sam.encode_image(current) {
                                Ok(()) => {
                                    println!("✅ SAM图像编码完成");
                                    let msg = "SAM图像编码完成！\n\n现在可以使用\"SAM交互分割\"功能进行对象分割。";
                                    let msg_utf16: Vec<u16> = msg.encode_utf16().chain(std::iter::once(0)).collect();
                                    let title_utf16: Vec<u16> = "编码成功".encode_utf16().chain(std::iter::once(0)).collect();
                                    MessageBoxW(hwnd, msg_utf16.as_ptr(), title_utf16.as_ptr(), MB_OK | MB_ICONINFORMATION);
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
        MENU_SAM_SEGMENT => {
            if let Some(ref sam) = state.sam_processor {
                if !sam.is_encoded() {
                    let msg = "请先使用\"SAM编码图像\"功能对当前图像进行编码";
                    let msg_utf16: Vec<u16> = msg.encode_utf16().chain(std::iter::once(0)).collect();
                    let title_utf16: Vec<u16> = "提示".encode_utf16().chain(std::iter::once(0)).collect();
                    MessageBoxW(hwnd, msg_utf16.as_ptr(), title_utf16.as_ptr(), MB_OK | MB_ICONINFORMATION);
                    return;
                }
            } else {
                let msg = "SAM处理器未初始化，请先执行\"SAM编码图像\"";
                let msg_utf16: Vec<u16> = msg.encode_utf16().chain(std::iter::once(0)).collect();
                let title_utf16: Vec<u16> = "提示".encode_utf16().chain(std::iter::once(0)).collect();
                MessageBoxW(hwnd, msg_utf16.as_ptr(), title_utf16.as_ptr(), MB_OK | MB_ICONINFORMATION);
                return;
            }
            state.selecting = true;
            state.select_start = None;
            state.select_end = None;
            println!("🖱️ SAM交互分割：请在图像上按下并拖拽以框选目标，释放鼠标完成框选");
        }
        
        MENU_SAM_EXTRACT_MASK => {
            if !state.has_mask() {
                let msg = "没有可用的mask，请先进行SAM分割！\n\n使用\"SAM交互分割\"功能在图像上拖拽框选目标，生成mask后再使用此功能。";
                let msg_utf16: Vec<u16> = msg.encode_utf16().chain(std::iter::once(0)).collect();
                let title_utf16: Vec<u16> = "提示".encode_utf16().chain(std::iter::once(0)).collect();
                MessageBoxW(hwnd, msg_utf16.as_ptr(), title_utf16.as_ptr(), MB_OK | MB_ICONINFORMATION);
                return;
            }
            
            match extract_mask_and_display(state) {
                Ok(()) => {
                    println!("✅ 抠图并显示完成");
                    InvalidateRect(hwnd, null_mut(), 1);
                    
                    let msg = "抠图完成！\n\n当前显示的是从原图中裁剪出的mask区域。\n黑色区域表示非mask部分，彩色区域是mask内的原图像素。\n\n您可以使用\"保存文件\"功能保存这个抠图结果。";
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
                    
                    let msg = "选框区域裁剪完成！\n\n当前显示的是从原图中裁剪出的选框区域。\n您可以使用\"保存文件\"功能保存这个裁剪结果。";
                    let msg_utf16: Vec<u16> = msg.encode_utf16().chain(std::iter::once(0)).collect();
                    let title_utf16: Vec<u16> = "裁剪成功".encode_utf16().chain(std::iter::once(0)).collect();
                    MessageBoxW(hwnd, msg_utf16.as_ptr(), title_utf16.as_ptr(), MB_OK | MB_ICONINFORMATION);
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
		WM_ERASEBKGND => {
			// 我们自己进行背景清除与双缓冲绘制，阻止系统擦除背景以减少闪烁
			1
		}
        WM_LBUTTONDOWN => {
            let state = IMAGE_STATE.as_mut().unwrap();
            if state.has_image() {
                let mx = (lparam & 0xFFFF) as u16 as i16 as i32;
                let my = ((lparam >> 16) & 0xFFFF) as u16 as i16 as i32;
                if let Some((ix, iy)) = window_to_image_coords(hwnd, state.current_width, state.current_height, mx, my) {
                    if state.selecting {
                        // SAM交互分割模式
                        state.select_start = Some((ix, iy));
                        state.select_end = Some((ix, iy));
                    } else if state.selection_mode {
                        // 选框工具模式
                        state.selection_start = Some((ix, iy));
                        state.selection_end = Some((ix, iy));
                    }
                    InvalidateRect(hwnd, null_mut(), 1);
                }
            }
            0
        }
        WM_MOUSEMOVE => {
            if (wparam & MK_LBUTTON as usize) != 0 {
                let state = IMAGE_STATE.as_mut().unwrap();
                if state.has_image() {
                    let mx = (lparam & 0xFFFF) as u16 as i16 as i32;
                    let my = ((lparam >> 16) & 0xFFFF) as u16 as i16 as i32;
                    if let Some((ix, iy)) = window_to_image_coords(hwnd, state.current_width, state.current_height, mx, my) {
                        if state.selecting {
                            // SAM交互分割模式
                            state.select_end = Some((ix, iy));
                        } else if state.selection_mode {
                            // 选框工具模式
                            state.selection_end = Some((ix, iy));
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
                if let Some((ix, iy)) = window_to_image_coords(hwnd, state.current_width, state.current_height, mx, my) {
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
                    let (x, y, scaled_width, scaled_height) = calculate_scaled_size(
                        state.current_width,
                        state.current_height,
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

            // 将内存DC内容拷贝到前台，一次提交减少闪烁
            BitBlt(hdc, 0, 0, window_width, window_height, mem_dc, 0, 0, SRCCOPY);

            // 释放内存对象
            SelectObject(mem_dc, old_bmp);
            DeleteObject(mem_bmp as HGDIOBJ);
            DeleteDC(mem_dc);
            
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