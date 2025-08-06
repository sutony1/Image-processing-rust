use opencv::prelude::*;
use opencv::{core, imgproc};
use anyhow::Result;
use crate::ocv_bridge;

/// 简化版的色卡检测，返回四个角点
pub fn find_color_card_simplified(bgr: &core::Mat) -> Result<Option<[core::Point2f; 4]>> {
    println!("开始简化版色卡检测...");
    
    // 保存原始图像用于调试
    ocv_bridge::save_debug_image(bgr, "01_original.png")?;
    
    // 转换为灰度图
    let gray = to_gray(bgr)?;
    ocv_bridge::save_debug_image(&gray, "02_grayscale.png")?;
    
    // 高斯模糊减少噪声
    let blurred = gaussian(&gray, 5, 0.0)?;
    
    // 尝试多组Canny参数，寻找最佳边缘检测结果
    let canny_params = [
        (30, 100),  // 低阈值
        (50, 150),  // 中等阈值  
        (100, 200), // 高阈值
        (80, 160),  // 另一组参数
    ];
    
    let mut best_contours = core::Vector::<core::Vector<core::Point>>::new();
    let mut best_edges = core::Mat::default();
    let mut max_contour_area = 0.0;
    
    for &(low, high) in &canny_params {
        // 进行边缘检测
        let edges = canny(&blurred, low as f64, high as f64)?;
        
        // 形态学操作连接断开的边缘
        let closed = morph_close(&edges, 3)?;
        let dilated = dilate(&closed, 3, 1)?;
        
        // 寻找轮廓
        let contours = find_and_draw_contours(&dilated)?;
        
        if !contours.is_empty() {
            // 找到最大轮廓
            let mut max_area = 0.0;
            for i in 0..contours.len() {
                let area = imgproc::contour_area(&contours.get(i)?, false)?;
                if area > max_area {
                    max_area = area;
                }
            }
            
            // 如果这组参数产生了更大的轮廓，则保存
            if max_area > max_contour_area {
                max_contour_area = max_area;
                best_contours = contours;
                best_edges = dilated;
                println!("找到更大轮廓，面积: {:.0}, Canny参数: ({}, {})", max_area, low, high);
            }
        }
    }
    
    if best_edges.empty() {
        // 如果没有找到合适的边缘，使用默认参数
        let edges = canny(&blurred, 50.0, 150.0)?;
        best_edges = edges;
        best_contours = find_and_draw_contours(&best_edges)?;
    }
    
    ocv_bridge::save_debug_image(&best_edges, "03_best_canny_edges.png")?;
    
    // 过滤小轮廓 - 使用更严格的面积阈值
    let total_area = (bgr.rows() * bgr.cols()) as f64;
    let min_area = total_area * 0.05; // 图像总面积的5%
    
    let mut filtered_contours = core::Vector::<core::Vector<core::Point>>::new();
    for i in 0..best_contours.len() {
        let contour = best_contours.get(i)?;
        let area = imgproc::contour_area(&contour, false)?;
        if area > min_area {
            filtered_contours.push(contour);
        }
    }
    
    println!("图像总面积: {:.0}, 最小轮廓面积阈值: {:.0}", total_area, min_area);
    println!("过滤后的轮廓数量: {}", filtered_contours.len());
    
    if filtered_contours.is_empty() {
        println!("方法1失败：未找到有效轮廓，尝试方法2...");
        return try_method_2(bgr, &gray);
    }
    
    // 选择最佳轮廓 - 找到面积最大的轮廓
    let mut best_contour = core::Vector::<core::Point>::new();
    let mut best_score = 0.0;
    
    for i in 0..filtered_contours.len() {
        let contour = filtered_contours.get(i)?;
        let area = imgproc::contour_area(&contour, false)?;
        
        if area > best_score {
            best_score = area;
            best_contour = contour;
        }
    }
    
    if best_contour.is_empty() {
        println!("未找到合适的轮廓，尝试方法2...");
        return try_method_2(bgr, &gray);
    }
    
    println!("选择的最佳轮廓面积: {:.0}", best_score);
    
    // 轮廓近似获得四个角点
    if let Some(points) = get_four_corners(&best_contour)? {
        return Ok(Some(points));
    }
    
    println!("方法1未能获得四个角点，尝试方法2...");
    try_method_2(bgr, &gray)
}

/// 方法2：使用简单二值化
fn try_method_2(image: &core::Mat, gray: &core::Mat) -> Result<Option<[core::Point2f; 4]>> {
    println!("尝试方法2：使用简单二值化...");
    
    // 使用简单的全局二值化
    let mut binary = core::Mat::default();
    imgproc::threshold(gray, &mut binary, 0.0, 255.0, 
        imgproc::THRESH_BINARY + imgproc::THRESH_OTSU)?;
    ocv_bridge::save_debug_image(&binary, "04_method2_binary.png")?;
    
    // 反转二值图像
    let mut binary_inv = core::Mat::default();
    core::bitwise_not(&binary, &mut binary_inv, &core::no_array())?;
    ocv_bridge::save_debug_image(&binary_inv, "05_method2_binary_inv.png")?;
    
    // 尝试两种二值化结果
    let binaries = [("normal", binary), ("inverted", binary_inv)];
    
    for (name, binary_img) in binaries {
        println!("尝试{}二值化结果...", name);
        
        // 寻找轮廓
        let contours = find_and_draw_contours(&binary_img)?;
        
        // 过滤小轮廓
        let min_area = (image.rows() * image.cols()) as f64 * 0.01;
        
        let mut max_contour = core::Vector::<core::Point>::new();
        let mut max_area = 0.0;
        
        for i in 0..contours.len() {
            let contour = contours.get(i)?;
            let area = imgproc::contour_area(&contour, false)?;
            if area > min_area && area > max_area {
                max_area = area;
                max_contour = contour;
            }
        }
        
        if !max_contour.is_empty() {
            // 尝试获得四个角点
            if let Some(points) = get_four_corners(&max_contour)? {
                println!("方法2成功：使用{}二值化找到四个角点", name);
                return Ok(Some(points));
            }
        }
    }
    
    println!("方法2失败，尝试方法3...");
    try_method_3(image, gray)
}

/// 方法3：使用自适应二值化
fn try_method_3(image: &core::Mat, gray: &core::Mat) -> Result<Option<[core::Point2f; 4]>> {
    println!("尝试方法3：使用自适应二值化...");
    
    // 自适应二值化
    let binary = adaptive_threshold(gray, 255.0, 11, 2.0)?;
    ocv_bridge::save_debug_image(&binary, "06_method3_adaptive.png")?;
    
    // 反转二值图像
    let mut binary_inv = core::Mat::default();
    core::bitwise_not(&binary, &mut binary_inv, &core::no_array())?;
    
    // 尝试两种二值化结果
    let binaries = [("normal", binary), ("inverted", binary_inv)];
    
    for (name, binary_img) in binaries {
        println!("尝试{}自适应二值化结果...", name);
        
        // 寻找轮廓
        let contours = find_and_draw_contours(&binary_img)?;
        
        // 过滤小轮廓
        let min_area = (image.rows() * image.cols()) as f64 * 0.01;
        
        let mut max_contour = core::Vector::<core::Point>::new();
        let mut max_area = 0.0;
        
        for i in 0..contours.len() {
            let contour = contours.get(i)?;
            let area = imgproc::contour_area(&contour, false)?;
            if area > min_area && area > max_area {
                max_area = area;
                max_contour = contour;
            }
        }
        
        if !max_contour.is_empty() {
            // 尝试获得四个角点
            if let Some(points) = get_four_corners(&max_contour)? {
                println!("方法3成功：使用{}自适应二值化找到四个角点", name);
                return Ok(Some(points));
            }
        }
    }
    
    println!("所有简化方法都失败了");
    Ok(None)
}

/// 从轮廓获取四个角点
fn get_four_corners(contour: &core::Vector<core::Point>) -> Result<Option<[core::Point2f; 4]>> {
    // 轮廓近似 - 尝试不同的精度参数
    let peri = imgproc::arc_length(contour, true)?;
    
    let epsilon_factors = [0.02, 0.03, 0.04, 0.05, 0.06];
    
    for &epsilon_factor in &epsilon_factors {
        let mut approx = core::Vector::<core::Point>::new();
        imgproc::approx_poly_dp(contour, &mut approx, epsilon_factor * peri, true)?;
        
        println!("使用精度因子 {}, 检测到 {} 个角点", epsilon_factor, approx.len());
        
        if approx.len() == 4 {
            let mut points = [core::Point2f::new(0.0, 0.0); 4];
            for i in 0..4 {
                let pt = approx.get(i)?;
                points[i] = core::Point2f::new(pt.x as f32, pt.y as f32);
            }
            
            // 排序四个角点
            let ordered_points = order_points(points)?;
            println!("检测到四个角点 (精度因子={})", epsilon_factor);
            return Ok(Some(ordered_points));
        } else if approx.len() > 4 {
            println!("角点过多 ({} 个)，尝试选择最外围的4个点...", approx.len());
            continue;
        }
    }
    
    println!("无法获得正确的四个角点");
    Ok(None)
}

/// 对四个角点进行排序：左上、右上、右下、左下
fn order_points(pts: [core::Point2f; 4]) -> Result<[core::Point2f; 4]> {
    let mut rect = [core::Point2f::new(0.0, 0.0); 4];
    
    // 计算坐标和差值
    let mut sums = Vec::new();
    let mut diffs = Vec::new();
    
    for point in &pts {
        sums.push(point.x + point.y);
        diffs.push(point.x - point.y);
    }
    
    // 找到各个角点的索引
    let mut min_sum_idx = 0;
    let mut max_sum_idx = 0;
    let mut min_diff_idx = 0;
    let mut max_diff_idx = 0;
    
    for i in 1..4 {
        if sums[i] < sums[min_sum_idx] { min_sum_idx = i; }
        if sums[i] > sums[max_sum_idx] { max_sum_idx = i; }
        if diffs[i] < diffs[min_diff_idx] { min_diff_idx = i; }
        if diffs[i] > diffs[max_diff_idx] { max_diff_idx = i; }
    }
    
    // 按顺序排列：左上、右上、右下、左下
    rect[0] = pts[min_sum_idx];  // 左上 (x+y最小)
    rect[1] = pts[min_diff_idx]; // 右上 (x-y最小)  
    rect[2] = pts[max_sum_idx];  // 右下 (x+y最大)
    rect[3] = pts[max_diff_idx]; // 左下 (x-y最大)
    
    Ok(rect)
}

/// 透视变换：从四个角点变换图像
pub fn perspective_transform_from_points(bgr: &core::Mat, pts: [core::Point2f; 4]) -> Result<core::Mat> {
    let ordered_points = order_points(pts)?;
    
    // 计算输出图像的宽度和高度
    let tl = ordered_points[0];
    let tr = ordered_points[1]; 
    let br = ordered_points[2];
    let bl = ordered_points[3];
    
    let width_a = ((br.x - bl.x).powi(2) + (br.y - bl.y).powi(2)).sqrt();
    let width_b = ((tr.x - tl.x).powi(2) + (tr.y - tl.y).powi(2)).sqrt();
    let max_width = width_a.max(width_b) as i32;
    
    let height_a = ((tr.x - br.x).powi(2) + (tr.y - br.y).powi(2)).sqrt();
    let height_b = ((tl.x - bl.x).powi(2) + (tl.y - bl.y).powi(2)).sqrt();
    let max_height = height_a.max(height_b) as i32;
    
    // 设置输出图像的角点
    let mut src_points = core::Vector::<core::Point2f>::new();
    let mut dst_points = core::Vector::<core::Point2f>::new();
    
    for &pt in &ordered_points {
        src_points.push(pt);
    }
    
    dst_points.push(core::Point2f::new(0.0, 0.0));
    dst_points.push(core::Point2f::new((max_width - 1) as f32, 0.0));
    dst_points.push(core::Point2f::new((max_width - 1) as f32, (max_height - 1) as f32));
    dst_points.push(core::Point2f::new(0.0, (max_height - 1) as f32));
    
    // 计算透视变换矩阵
    let transform_matrix = imgproc::get_perspective_transform(&src_points, &dst_points, core::DECOMP_LU)?;
    
    // 进行透视变换
    let mut warped = core::Mat::default();
    imgproc::warp_perspective(
        bgr, 
        &mut warped, 
        &transform_matrix, 
        core::Size::new(max_width, max_height),
        imgproc::INTER_LINEAR,
        core::BORDER_CONSTANT,
        core::Scalar::default()
    )?;
    
    ocv_bridge::save_debug_image(&warped, "07_perspective_result.png")?;
    
    Ok(warped)
}

/// 简化透视变换：自动检测并变换
pub fn perspective_transform_simplified(bgr: &core::Mat) -> Result<Option<core::Mat>> {
    // 找到色卡的四个角点
    if let Some(points) = find_color_card_simplified(bgr)? {
        // 执行透视变换
        let result = perspective_transform_from_points(bgr, points)?;
        Ok(Some(result))
    } else {
        Ok(None)
    }
}

// ========== 通用图像处理函数 ==========

/// 转换为灰度图
pub fn to_gray(bgr: &core::Mat) -> Result<core::Mat> {
    let mut gray = core::Mat::default();
    imgproc::cvt_color(bgr, &mut gray, imgproc::COLOR_BGR2GRAY, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
    Ok(gray)
}

/// 高斯模糊
pub fn gaussian(gray: &core::Mat, ksize: i32, sigma: f64) -> Result<core::Mat> {
    let mut blurred = core::Mat::default();
    let kernel_size = core::Size::new(ksize, ksize);
    imgproc::gaussian_blur(gray, &mut blurred, kernel_size, sigma, 0.0, core::BORDER_DEFAULT, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
    Ok(blurred)
}

/// Canny边缘检测
pub fn canny(gray: &core::Mat, low_threshold: f64, high_threshold: f64) -> Result<core::Mat> {
    let mut edges = core::Mat::default();
    imgproc::canny(gray, &mut edges, low_threshold, high_threshold, 3, false)?;
    Ok(edges)
}

/// 自适应阈值
pub fn adaptive_threshold(gray: &core::Mat, max_val: f64, block_size: i32, c: f64) -> Result<core::Mat> {
    let mut binary = core::Mat::default();
    imgproc::adaptive_threshold(
        gray, 
        &mut binary, 
        max_val, 
        imgproc::ADAPTIVE_THRESH_GAUSSIAN_C,
        imgproc::THRESH_BINARY,
        block_size,
        c
    )?;
    Ok(binary)
}

/// 形态学闭运算
pub fn morph_close(binary: &core::Mat, ksize: i32) -> Result<core::Mat> {
    let kernel = imgproc::get_structuring_element(
        imgproc::MORPH_RECT,
        core::Size::new(ksize, ksize),
        core::Point::new(-1, -1)
    )?;
    
    let mut closed = core::Mat::default();
    imgproc::morphology_ex(
        binary,
        &mut closed, 
        imgproc::MORPH_CLOSE,
        &kernel,
        core::Point::new(-1, -1),
        1,
        core::BORDER_CONSTANT,
        imgproc::morphology_default_border_value()?
    )?;
    
    Ok(closed)
}

/// 膨胀操作
pub fn dilate(binary: &core::Mat, ksize: i32, iterations: i32) -> Result<core::Mat> {
    let kernel = imgproc::get_structuring_element(
        imgproc::MORPH_RECT,
        core::Size::new(ksize, ksize),
        core::Point::new(-1, -1)
    )?;
    
    let mut dilated = core::Mat::default();
    imgproc::dilate(
        binary,
        &mut dilated,
        &kernel,
        core::Point::new(-1, -1),
        iterations,
        core::BORDER_CONSTANT,
        imgproc::morphology_default_border_value()?
    )?;
    
    Ok(dilated)
}

/// 寻找并绘制轮廓
pub fn find_and_draw_contours(binary: &core::Mat) -> Result<core::Vector<core::Vector<core::Point>>> {
    let mut contours = core::Vector::new();
    
    imgproc::find_contours(
        binary,
        &mut contours,
        imgproc::RETR_EXTERNAL,
        imgproc::CHAIN_APPROX_SIMPLE,
        core::Point::default()
    )?;
    
    Ok(contours)
}