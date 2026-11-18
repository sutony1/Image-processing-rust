use opencv::prelude::*;
use opencv::{core, imgproc, imgcodecs};
use anyhow::{Context, Result};
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
    
    // 绘制所有轮廓
    let mut contour_img = bgr.clone();
    for i in 0..filtered_contours.len() {
        // 仅绘制所有轮廓
        imgproc::draw_contours(&mut contour_img, &filtered_contours, i as i32, 
                              core::Scalar::new(0.0, 255.0, 0.0, 0.0), 2, 
                              imgproc::LINE_8, &core::no_array(), i32::MAX, core::Point::new(0, 0))?;
    }
    ocv_bridge::save_debug_image(&contour_img, "04_detected_contours.png")?;
    
    if filtered_contours.is_empty() {
        println!("方法1失败：未找到有效轮廓，尝试方法2...");
        return try_method_2(bgr, &gray);
    }
    
    // 打印所有轮廓的信息
    println!("所有轮廓信息:");
    for i in 0..filtered_contours.len() {
        let contour = filtered_contours.get(i)?;
        let area = imgproc::contour_area(&contour, false)?;
        let perimeter = imgproc::arc_length(&contour, true)?;
        // 计算轮廓的紧凑度（面积/周长比）
        let compactness = if perimeter > 0.0 {
            4.0 * std::f64::consts::PI * area / (perimeter * perimeter)
        } else {
            0.0
        };
        println!("轮廓 {}: 面积={:.0}, 周长={:.0}, 紧凑度={:.3}", i, area, perimeter, compactness);
    }
    
    // 选择最佳轮廓 - 不仅考虑面积，还考虑形状特征
    let mut best_contour = core::Vector::<core::Point>::new();
    let mut best_score = 0.0;
    
    for i in 0..filtered_contours.len() {
        let contour = filtered_contours.get(i)?;
        let area = imgproc::contour_area(&contour, false)?;
        let perimeter = imgproc::arc_length(&contour, true)?;
        
        // 计算轮廓的凸包
        let mut hull = core::Vector::<core::Point>::new();
        imgproc::convex_hull(&contour, &mut hull, false, true)?;
        let hull_area = imgproc::contour_area(&hull, false)?;
        
        // 计算凸性比例（轮廓面积/凸包面积）
        let convexity_ratio = if hull_area > 0.0 { area / hull_area } else { 0.0 };
        
        // 计算紧凑度
        let compactness = if perimeter > 0.0 {
            4.0 * std::f64::consts::PI * area / (perimeter * perimeter)
        } else {
            0.0
        };
        
        // 综合评分：面积权重最高，但也考虑形状特征
        // 色卡通常是矩形，应该有较好的凸性比例和适中的紧凑度
        let area_score = area / total_area;  // 面积占比
        let shape_score = convexity_ratio * (compactness * 2.0).min(1.0);  // 形状评分
        
        // 综合评分
        let total_score = area_score * 0.7 + shape_score * 0.3;
        
        println!("轮廓面积={:.0}, 凸性比例={:.3}, 紧凑度={:.3}, 综合评分={:.3}", 
                 area, convexity_ratio, compactness, total_score);
        
        if total_score > best_score {
            best_score = total_score;
            best_contour = contour;
        }
    }
    
    if best_contour.is_empty() {
        println!("未找到合适的轮廓，尝试方法2...");
        return try_method_2(bgr, &gray);
    }
    
    println!("选择的最佳轮廓综合评分: {:.3}", best_score);
    
    // 绘制选择的最佳轮廓
    let mut best_contour_img = bgr.clone();
    let mut best_contours = core::Vector::<core::Vector<core::Point>>::new();
    best_contours.push(best_contour.clone());
    imgproc::draw_contours(&mut best_contour_img, &best_contours, -1, 
                          core::Scalar::new(0.0, 255.0, 0.0, 0.0), 3, 
                          imgproc::LINE_8, &core::no_array(), i32::MAX, core::Point::new(0, 0))?;
    ocv_bridge::save_debug_image(&best_contour_img, "05_best_contour.png")?;
    
    // 轮廓近似获得四个角点
    if let Some(points) = get_four_corners(&best_contour)? {
        // 绘制检测到的角点
        let mut points_img = bgr.clone();
        draw_points(&mut points_img, &points)?;
        ocv_bridge::save_debug_image(&points_img, "06_detected_corners.png")?;
        
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

/// 在图像上绘制角点
fn draw_points(image: &mut core::Mat, points: &[core::Point2f; 4]) -> Result<()> {
    let colors = [
        core::Scalar::new(0.0, 255.0, 0.0, 0.0),   // 绿色 - 左上
        core::Scalar::new(255.0, 0.0, 0.0, 0.0),   // 蓝色 - 右上  
        core::Scalar::new(0.0, 0.0, 255.0, 0.0),   // 红色 - 右下
        core::Scalar::new(255.0, 255.0, 0.0, 0.0), // 青色 - 左下
    ];
    
    for (i, &point) in points.iter().enumerate() {
        let center = core::Point::new(point.x as i32, point.y as i32);
        
        // 绘制圆点
        imgproc::circle(image, center, 10, colors[i], -1, imgproc::LINE_8, 0)?;
        
        // 绘制数字标签
        let text = format!("{}", i);
        let font_face = imgproc::FONT_HERSHEY_SIMPLEX;
        let font_scale = 1.0;
        let thickness = 2;
        
        let text_pos = core::Point::new(point.x as i32 + 15, point.y as i32 + 15);
        imgproc::put_text(image, &text, text_pos, font_face, font_scale, colors[i], thickness, imgproc::LINE_8, false)?;
    }
    
    Ok(())
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
            
            // 保存检测到的角点图像用于调试
            // 注意：这里我们暂时跳过绘制，因为需要传入原始图像才能绘制角点
            // 实际使用中，调用方会负责绘制和保存最终的角点图像
            
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
    // 在图像坐标系中：
    // - 左上: x+y 最小
    // - 右下: x+y 最大
    // - 右上: x-y 最大
    // - 左下: x-y 最小
    rect[0] = pts[min_sum_idx];  // 左上 (x+y最小)
    rect[1] = pts[max_diff_idx]; // 右上 (x-y最大)
    rect[2] = pts[max_sum_idx];  // 右下 (x+y最大)
    rect[3] = pts[min_diff_idx]; // 左下 (x-y最小)
    
    let contour = core::Vector::from_slice(&rect);
    if imgproc::is_contour_convex(&contour)? {
        return Ok(rect);
    }

    // On strong tilts the sum/difference extrema can select the same point twice.
    let center_x = pts.iter().map(|p| p.x).sum::<f32>() / 4.0;
    let center_y = pts.iter().map(|p| p.y).sum::<f32>() / 4.0;
    let angle = |p: &core::Point2f| (p.y - center_y).atan2(p.x - center_x);
    rect = pts;
    rect.sort_by(|a, b| angle(a).total_cmp(&angle(b)));
    let start = rect.iter().position(|p| *p == pts[min_sum_idx])
        .context("无法确定左上角")?;
    rect.rotate_left(start);
    Ok(rect)
}

/// 透视变换：从四个角点变换图像
pub fn perspective_transform_from_points(bgr: &core::Mat, pts: [core::Point2f; 4]) -> Result<core::Mat> {
    let ordered_points = order_points(pts)?;
    
    let warped = warp_perspective_quad(bgr, ordered_points)?;
    let mut annotated = bgr.try_clone()?;
    if let Err(error) = draw_points(&mut annotated, &ordered_points)
        .and_then(|_| ocv_bridge::save_debug_image(&annotated, "07_ordered_corners.png"))
    {
        println!("无法保存角点调试图: {}", error);
    }
    if let Err(error) = ocv_bridge::save_debug_image(&warped, "08_perspective_result.png") {
        println!("无法保存透视调试图: {}", error);
    }
    Ok(warped)
}

/// Warp corners supplied clockwise from the top left.
pub fn warp_perspective_quad(bgr: &core::Mat, ordered_points: [core::Point2f; 4]) -> Result<core::Mat> {
    if ordered_points.iter().any(|p| {
        !p.x.is_finite() || !p.y.is_finite() || p.x < 0.0 || p.y < 0.0
            || p.x >= bgr.cols() as f32 || p.y >= bgr.rows() as f32
    }) {
        anyhow::bail!("四个角点必须位于图像内。");
    }
    let src_points = core::Vector::from_slice(&ordered_points);
    if !imgproc::is_contour_convex(&src_points)? || imgproc::contour_area(&src_points, true)? < 4.0 {
        anyhow::bail!("四点须围成顺时针四边形，角点不能重复或交叉。");
    }

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
    if max_width < 2 || max_height < 2 {
        anyhow::bail!("所选四边形太窄。");
    }
    let mut dst_points = core::Vector::<core::Point2f>::new();
    
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
        
    Ok(warped)
}

/// 简化透视变换：自动检测并变换
pub fn perspective_transform_simplified(bgr: &core::Mat) -> Result<Option<core::Mat>> {
    // 找到色卡的四个角点
    if let Some(points) = find_color_card_simplified(bgr)? {
        // 执行透视变换
        let result = perspective_transform_from_points(bgr, points)?;

        // 如果原图为竖向（高>=宽），而结果为横向（宽>高），则将结果旋转为竖向
        let original_is_portrait = bgr.rows() >= bgr.cols();
        let result_is_landscape = result.cols() > result.rows();

        if original_is_portrait && result_is_landscape {
            // 90° 顺时针旋转 = 转置 + 水平翻转
            let mut transposed = core::Mat::default();
            core::transpose(&result, &mut transposed)?;
            let mut rotated = core::Mat::default();
            core::flip(&transposed, &mut rotated, 1)?; // 1: 水平翻转
            ocv_bridge::save_debug_image(&rotated, "09_oriented_result.png")?;
            return Ok(Some(rotated));
        }

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

/// 色块信息结构体
#[derive(Debug, Clone)]
pub struct ColorBlockInfo {
    pub block_number: usize,
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub h: u16,  // 0-360度
    pub s: u8,   // 0-100%
    pub v: u8,   // 0-100%
    pub l: u8,   // Lab L值
    pub a: u8,   // Lab a值
    pub b_val: u8, // Lab b值
    pub hex: String,
}

/// 将BGR颜色转换为HSV颜色空间（简化版本）
fn bgr_to_hsv(b: u8, g: u8, r: u8) -> Result<(u16, u8, u8)> {
    // 简化的颜色空间转换，手动计算
    let r_f = r as f32 / 255.0;
    let g_f = g as f32 / 255.0;
    let b_f = b as f32 / 255.0;
    
    let max_val = r_f.max(g_f).max(b_f);
    let min_val = r_f.min(g_f).min(b_f);
    let delta = max_val - min_val;
    
    // 计算H (色相)
    let h = if delta == 0.0 {
        0.0
    } else if max_val == r_f {
        60.0 * (((g_f - b_f) / delta) % 6.0)
    } else if max_val == g_f {
        60.0 * ((b_f - r_f) / delta + 2.0)
    } else {
        60.0 * ((r_f - g_f) / delta + 4.0)
    };
    
    let h = if h < 0.0 { h + 360.0 } else { h };
    
    // 计算S (饱和度)
    let s = if max_val == 0.0 { 0.0 } else { delta / max_val };
    
    // 计算V (明度)
    let v = max_val;
    
    Ok((h as u16, (s * 100.0) as u8, (v * 100.0) as u8))
}

/// 将BGR颜色转换为Lab颜色空间（简化版本）
fn bgr_to_lab(_b: u8, _g: u8, _r: u8) -> Result<(u8, u8, u8)> {
    // 简化版本，返回RGB作为Lab的近似值
    // 在实际应用中可能需要更精确的转换
    Ok((128, 128, 128)) // 返回中性值
}

/// 色块提取结果
pub struct ColorExtractionResult {
    pub color_blocks: Vec<ColorBlockInfo>,
    pub grid_image: core::Mat,
    pub color_palette: core::Mat,
}

/// 使用网格法从透视变换后的色卡图像中提取24个色块及其RGB值
pub fn extract_color_blocks_grid(image: &core::Mat) -> Result<ColorExtractionResult> {
    println!("开始使用网格法提取色块...");
    
    // 保存原始图像用于调试
    ocv_bridge::save_debug_image(image, "grid_01_original.png")?;
    
    // 获取图像尺寸
    let height = image.rows();
    let width = image.cols();
    
    // 转换为HSV色彩空间
    let mut hsv = core::Mat::default();
    imgproc::cvt_color(image, &mut hsv, imgproc::COLOR_BGR2HSV, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
    
    // 分离HSV通道
    let mut hsv_channels = core::Vector::<core::Mat>::new();
    core::split(&hsv, &mut hsv_channels)?;
    let v_channel = hsv_channels.get(2)?;
    let s_channel = hsv_channels.get(1)?;
    
    // 使用更严格的阈值来排除黑色区域（低亮度）
    let mut v_mask = core::Mat::default();
    imgproc::threshold(&v_channel, &mut v_mask, 40.0, 255.0, imgproc::THRESH_BINARY)?;
    
    let mut s_mask = core::Mat::default();
    imgproc::threshold(&s_channel, &mut s_mask, 30.0, 255.0, imgproc::THRESH_BINARY)?;
    
    // 组合亮度和饱和度掩码
    let mut combined_mask = core::Mat::default();
    core::bitwise_and(&v_mask, &s_mask, &mut combined_mask, &core::no_array())?;
    
    // 通过数学形态学操作改进掩码
    let kernel = imgproc::get_structuring_element(
        imgproc::MORPH_RECT,
        core::Size::new(7, 7),
        core::Point::new(-1, -1)
    )?;
    
    let mut morphed_mask = core::Mat::default();
    imgproc::morphology_ex(
        &combined_mask,
        &mut morphed_mask,
        imgproc::MORPH_CLOSE,
        &kernel,
        core::Point::new(-1, -1),
        1,
        core::BORDER_CONSTANT,
        imgproc::morphology_default_border_value()?
    )?;
    
    let mut final_mask = core::Mat::default();
    imgproc::morphology_ex(
        &morphed_mask,
        &mut final_mask,
        imgproc::MORPH_OPEN,
        &kernel,
        core::Point::new(-1, -1),
        1,
        core::BORDER_CONSTANT,
        imgproc::morphology_default_border_value()?
    )?;
    
    // 保存掩码用于调试
    ocv_bridge::save_debug_image(&final_mask, "grid_02_mask.png")?;
    
    // 可视化掩码（将黑色区域标记为红色）
    let mut mask_viz = image.clone();
    for y in 0..height {
        for x in 0..width {
            let mask_val = *final_mask.at_2d::<u8>(y, x)?;
            if mask_val == 0 {
                // 将黑色区域标记为红色
                let pixel = mask_viz.at_2d_mut::<core::Vec3b>(y, x)?;
                *pixel = core::Vec3b::from([0, 0, 255]); // BGR格式：红色
            }
        }
    }
    ocv_bridge::save_debug_image(&mask_viz, "grid_03_mask_visualization.png")?;
    
    // 获取非黑色区域的边界框
    let mut contours = core::Vector::<core::Vector<core::Point>>::new();
    imgproc::find_contours(
        &final_mask,
        &mut contours,
        imgproc::RETR_EXTERNAL,
        imgproc::CHAIN_APPROX_SIMPLE,
        core::Point::default()
    )?;
    
    if contours.is_empty() {
        return Err(anyhow::anyhow!("未找到有效的非黑色区域"));
    }
    
    // 找到最大非黑色区域（应该是色卡区域）
    let mut max_area = 0.0;
    let mut main_contour = core::Vector::<core::Point>::new();
    
    for i in 0..contours.len() {
        let contour = contours.get(i)?;
        let area = imgproc::contour_area(&contour, false)?;
        if area > max_area {
            max_area = area;
            main_contour = contour;
        }
    }
    
    let rect = imgproc::bounding_rect(&main_contour)?;
    let (x, y, w, h) = (rect.x, rect.y, rect.width, rect.height);
    
    println!("找到色卡区域: x={}, y={}, w={}, h={}", x, y, w, h);
    
    // 创建网格分割（6行4列，常见的色卡布局）
    let rows = 6;
    let cols = 4;
    
    // 计算每个色块的大小，基于非黑色区域的边界框
    let block_height = h / rows;
    let block_width = w / cols;
    
    println!("色块尺寸: {}x{}", block_width, block_height);
    
    // 确定每个色块的中心点和边界框
    let mut centers = Vec::new();
    let mut bboxes = Vec::new();
    
    for row in 0..rows {
        for col in 0..cols {
            // 计算色块的边界框
            let block_x = x + col * block_width;
            let block_y = y + row * block_height;
            
            // 计算色块的中心点坐标
            let center_x = block_x + block_width / 2;
            let center_y = block_y + block_height / 2;
            
            centers.push((center_x, center_y));
            bboxes.push((block_x, block_y, block_width, block_height));
        }
    }
    
    // 显示网格划分和色块中心点
    let mut grid_img = image.clone();
    
    // 绘制外边框
    imgproc::rectangle(
        &mut grid_img,
        core::Rect::new(x, y, w, h),
        core::Scalar::new(0.0, 255.0, 255.0, 0.0), // 黄色
        2,
        imgproc::LINE_8,
        0
    )?;
    
    // 绘制水平线
    for r in 1..rows {
        let y_line = y + r * block_height;
        imgproc::line(
            &mut grid_img,
            core::Point::new(x, y_line),
            core::Point::new(x + w, y_line),
            core::Scalar::new(255.0, 255.0, 255.0, 0.0), // 白色
            2,
            imgproc::LINE_8,
            0
        )?;
    }
    
    // 绘制垂直线
    for c in 1..cols {
        let x_line = x + c * block_width;
        imgproc::line(
            &mut grid_img,
            core::Point::new(x_line, y),
            core::Point::new(x_line, y + h),
            core::Scalar::new(255.0, 255.0, 255.0, 0.0), // 白色
            2,
            imgproc::LINE_8,
            0
        )?;
    }
    
    // 绘制中心点和标签
    for (i, &(center_x, center_y)) in centers.iter().enumerate() {
        imgproc::circle(
            &mut grid_img,
            core::Point::new(center_x, center_y),
            5,
            core::Scalar::new(0.0, 0.0, 255.0, 0.0), // 红色
            -1,
            imgproc::LINE_8,
            0
        )?;
        
        let text = format!("{}", i + 1);
        imgproc::put_text(
            &mut grid_img,
            &text,
            core::Point::new(center_x - 10, center_y + 10),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.8,
            core::Scalar::new(0.0, 0.0, 255.0, 0.0), // 红色
            2,
            imgproc::LINE_8,
            false
        )?;
    }
    
    // 先不保存，等待用户交互后再保存
    // ocv_bridge::save_debug_image(&grid_img, "grid_04_grid_lines.png")?;
    
    // 从每个色块中提取颜色样本
    let sample_size = std::cmp::min(block_width, block_height) / 3; // 采样区域大小
    let mut color_info = Vec::new();
    
    for (i, &(center_x, center_y)) in centers.iter().enumerate() {
        
        // 定义采样区域（色块中心的小区域）
        let sample_x1 = std::cmp::max(0, center_x - sample_size / 2);
        let sample_y1 = std::cmp::max(0, center_y - sample_size / 2);
        let sample_x2 = std::cmp::min(width, center_x + sample_size / 2);
        let sample_y2 = std::cmp::min(height, center_y + sample_size / 2);
        
        // 提取色块中心区域
        let sample_rect = core::Rect::new(
            sample_x1,
            sample_y1,
            sample_x2 - sample_x1,
            sample_y2 - sample_y1
        );
        let sample = core::Mat::roi(image, sample_rect)?;
        
        // 计算平均颜色
        let mean_scalar = core::mean(&sample, &core::no_array())?;
        let b = mean_scalar[0] as u8;
        let g = mean_scalar[1] as u8;
        let r = mean_scalar[2] as u8;
        
        // 转换为其他色彩空间
        let (h, s, v) = bgr_to_hsv(b, g, r).unwrap_or((0, 0, 0));
        let (l, a, b_val) = bgr_to_lab(b, g, r).unwrap_or((0, 0, 0));
        
        color_info.push(ColorBlockInfo {
            block_number: i + 1,
            r,
            g,
            b,
            h,
            s,
            v,
            l,
            a,
            b_val,
            hex: format!("#{:02X}{:02X}{:02X}", r, g, b),
        });
        
        println!("色块 {}: RGB({}, {}, {}) HSV({}, {}%, {}%) #{:02X}{:02X}{:02X}",
                 i + 1, r, g, b, h, s, v, r, g, b);
    }
    
    // 创建一个可视化展示所有提取的色块
    let block_size = 100;
    let mut color_display = core::Mat::zeros(
        rows * block_size,
        cols * block_size,
        core::CV_8UC3
    )?.to_mat()?;
    
    for (i, info) in color_info.iter().enumerate() {
        let row = i / cols as usize;
        let col = i % cols as usize;
        
        // BGR颜色填充
        let rect = core::Rect::new(
            (col * block_size as usize) as i32,
            (row * block_size as usize) as i32,
            block_size,
            block_size
        );
        
        let color = core::Scalar::new(info.b as f64, info.g as f64, info.r as f64, 0.0);
        imgproc::rectangle(
            &mut color_display,
            rect,
            color,
            -1,
            imgproc::LINE_8,
            0
        )?;
        
        // 根据颜色的亮度选择文字颜色
        let brightness = 0.299 * info.r as f64 + 0.587 * info.g as f64 + 0.114 * info.b as f64;
        let text_color = if brightness > 127.0 {
            core::Scalar::new(0.0, 0.0, 0.0, 0.0) // 黑色
        } else {
            core::Scalar::new(255.0, 255.0, 255.0, 0.0) // 白色
        };
        
        // 添加色块编号
        imgproc::put_text(
            &mut color_display,
            &format!("{}", i + 1),
            core::Point::new(
                (col * block_size as usize) as i32 + 10,
                (row * block_size as usize) as i32 + 30
            ),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.9,
            text_color,
            2,
            imgproc::LINE_8,
            false
        )?;
        
        // 添加RGB十六进制值
        imgproc::put_text(
            &mut color_display,
            &info.hex,
            core::Point::new(
                (col * block_size as usize) as i32 + 10,
                (row * block_size as usize) as i32 + 55
            ),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.4,
            text_color,
            1,
            imgproc::LINE_8,
            false
        )?;
        
        // 添加RGB值
        let rgb_text = format!("R:{} G:{} B:{}", info.r, info.g, info.b);
        imgproc::put_text(
            &mut color_display,
            &rgb_text,
            core::Point::new(
                (col * block_size as usize) as i32 + 10,
                (row * block_size as usize) as i32 + 75
            ),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.4,
            text_color,
            1,
            imgproc::LINE_8,
            false
        )?;
    }
    
    // 先不保存，等待用户交互后再保存
    // ocv_bridge::save_debug_image(&color_display, "grid_05_color_palette.png")?;
    
    println!("✅ 成功提取了 {} 个色块", color_info.len());
    
    Ok(ColorExtractionResult {
        color_blocks: color_info,
        grid_image: grid_img,
        color_palette: color_display,
    })
}

// ========== 色卡匹配算法 (基于模板匹配) ==========

/// 色卡匹配结果
pub struct ColorCardMatchResult {
    pub homography: Option<core::Mat>,
    pub template_corners: [core::Point2f; 4],
    pub projected_corners: Option<[core::Point2f; 4]>,
    pub warped_image: Option<core::Mat>,
    pub scene_with_detection: Option<core::Mat>,  // 标注了检测结果的原场景图像
    pub match_confidence: f64,
}

/// 使用模板匹配进行色卡检测和透视变换
pub fn detect_color_card_with_template_matching(
    template: &core::Mat,
    scene: &core::Mat
) -> Result<ColorCardMatchResult> {
    println!("开始基于模板匹配的色卡检测...");
    
    // 保存输入图像用于调试
    ocv_bridge::save_debug_image(template, "template_01_template.png")?;
    ocv_bridge::save_debug_image(scene, "template_02_scene.png")?;
    
    // 转换为灰度图
    let template_gray = to_gray(template)?;
    let scene_gray = to_gray(scene)?;
    
    // 进行模板匹配
    let mut result_mat = core::Mat::default();
    imgproc::match_template(&scene_gray, &template_gray, &mut result_mat, imgproc::TM_CCOEFF_NORMED, &core::no_array())?;
    
    // 找到最佳匹配位置
    let mut min_val = 0.0;
    let mut max_val = 0.0;
    let mut min_loc = core::Point::default();
    let mut max_loc = core::Point::default();
    
    core::min_max_loc(&result_mat, Some(&mut min_val), Some(&mut max_val), Some(&mut min_loc), Some(&mut max_loc), &core::no_array())?;
    
    println!("模板匹配最佳匹配度: {:.3}", max_val);
    
    if max_val < 0.3 {
        return Err(anyhow::anyhow!("模板匹配度太低: {:.3}", max_val));
    }
    
    // 计算匹配区域
    let template_width = template.cols();
    let template_height = template.rows();
    
    let match_rect = core::Rect::new(
        max_loc.x,
        max_loc.y,
        template_width,
        template_height
    );
    
    // 绘制匹配区域 - 创建带检测框的场景图像
    let mut scene_with_detection = scene.clone();
    
    // 绘制绿色矩形框
    imgproc::rectangle(
        &mut scene_with_detection,
        match_rect,
        core::Scalar::new(0.0, 255.0, 0.0, 0.0), // 绿色
        4,
        imgproc::LINE_8,
        0
    )?;
    
    // 在四个角点绘制圆点和标号
    let corners = [
        core::Point::new(max_loc.x, max_loc.y),                                    // 左上
        core::Point::new(max_loc.x + template_width, max_loc.y),                   // 右上
        core::Point::new(max_loc.x + template_width, max_loc.y + template_height), // 右下
        core::Point::new(max_loc.x, max_loc.y + template_height),                  // 左下
    ];
    
    let colors = [
        core::Scalar::new(255.0, 0.0, 0.0, 0.0),   // 蓝色
        core::Scalar::new(0.0, 255.0, 0.0, 0.0),   // 绿色
        core::Scalar::new(0.0, 0.0, 255.0, 0.0),   // 红色
        core::Scalar::new(255.0, 255.0, 0.0, 0.0), // 青色
    ];
    
    for (i, &corner) in corners.iter().enumerate() {
        // 绘制圆点
        imgproc::circle(
            &mut scene_with_detection,
            corner,
            8,
            colors[i],
            -1,
            imgproc::LINE_8,
            0
        )?;
        
        // 绘制标号
        let text = format!("{}", i + 1);
        imgproc::put_text(
            &mut scene_with_detection,
            &text,
            core::Point::new(corner.x + 12, corner.y - 12),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.8,
            core::Scalar::new(255.0, 255.0, 255.0, 0.0), // 白色
            2,
            imgproc::LINE_8,
            false
        )?;
    }
    
    // 添加匹配置信度文本
    let confidence_text = format!("Match: {:.1}%", max_val * 100.0);
    imgproc::put_text(
        &mut scene_with_detection,
        &confidence_text,
        core::Point::new(max_loc.x, max_loc.y - 20),
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.7,
        core::Scalar::new(0.0, 255.0, 0.0, 0.0), // 绿色
        2,
        imgproc::LINE_8,
        false
    )?;
    
    ocv_bridge::save_debug_image(&scene_with_detection, "template_03_match_result.png")?;
    
    // 定义四个角点
    let template_corners = get_template_corners(template)?;
    let projected_corners = [
        core::Point2f::new(max_loc.x as f32, max_loc.y as f32),
        core::Point2f::new((max_loc.x + template_width) as f32, max_loc.y as f32),
        core::Point2f::new((max_loc.x + template_width) as f32, (max_loc.y + template_height) as f32),
        core::Point2f::new(max_loc.x as f32, (max_loc.y + template_height) as f32),
    ];
    
    // 创建透视变换矩阵
    let mut src_pts = core::Vector::<core::Point2f>::new();
    let mut dst_pts = core::Vector::<core::Point2f>::new();
    
    for &corner in &projected_corners {
        src_pts.push(corner);
    }
    
    for &corner in &template_corners {
        dst_pts.push(corner);
    }
    
    let transform_matrix = imgproc::get_perspective_transform(&src_pts, &dst_pts, core::DECOMP_LU)?;
    
    // 进行透视变换
    let mut warped_img = core::Mat::default();
    imgproc::warp_perspective(
        scene,
        &mut warped_img,
        &transform_matrix,
        core::Size::new(template.cols(), template.rows()),
        imgproc::INTER_LINEAR,
        core::BORDER_CONSTANT,
        core::Scalar::default()
    )?;
    
    ocv_bridge::save_debug_image(&warped_img, "template_04_warped.png")?;
    
    Ok(ColorCardMatchResult {
        homography: Some(transform_matrix),
        template_corners,
        projected_corners: Some(projected_corners),
        warped_image: Some(warped_img),
        scene_with_detection: Some(scene_with_detection),
        match_confidence: max_val,
    })
}

/// 获取模板图像的四个角点
fn get_template_corners(template: &core::Mat) -> Result<[core::Point2f; 4]> {
    let h = template.rows() as f32;
    let w = template.cols() as f32;
    
    Ok([
        core::Point2f::new(0.0, 0.0),           // 左上
        core::Point2f::new(w - 1.0, 0.0),       // 右上
        core::Point2f::new(w - 1.0, h - 1.0),   // 右下
        core::Point2f::new(0.0, h - 1.0),       // 左下
    ])
}

/// 简化接口：返回完整的匹配结果
pub fn match_and_transform_color_card(
    template_path: &str,
    scene: &core::Mat
) -> Result<ColorCardMatchResult> {
    println!("加载模板图像: {}", template_path);
    
    // 先尝试直接读取
    let template = imgcodecs::imread(template_path, imgcodecs::IMREAD_COLOR)?;
    
    let template = if template.empty() {
        // 如果OpenCV直接读取失败，尝试使用备用方法
        println!("OpenCV直接读取模板图像失败，尝试备用方法...");
        
        // 使用标准库读取文件到内存
        let image_data = std::fs::read(template_path)
            .map_err(|e| anyhow::anyhow!("无法读取模板图像文件到内存: {}", e))?;
        
        // 使用OpenCV从内存数据解码图像
        let data_vec = core::Vector::from_slice(&image_data);
        let decoded = imgcodecs::imdecode(&data_vec, imgcodecs::IMREAD_COLOR)
            .map_err(|e| anyhow::anyhow!("无法解码模板图像数据: {}", e))?;
        
        if decoded.empty() {
            return Err(anyhow::anyhow!("解码后的模板图像为空，可能是不支持的格式"));
        }
        
        println!("模板图像加载成功（备用方法）");
        decoded
    } else {
        println!("模板图像加载成功（直接方法）");
        template
    };
    
    let result = detect_color_card_with_template_matching(&template, scene)?;
    
    println!("匹配结果:");
    println!("  匹配置信度: {:.3}", result.match_confidence);
    
    Ok(result)
}

/// 简化接口：使用现有的Mat对象进行色卡匹配
pub fn match_color_card_from_mats(
    template: &core::Mat,
    scene: &core::Mat
) -> Result<ColorCardMatchResult> {
    detect_color_card_with_template_matching(template, scene)
}