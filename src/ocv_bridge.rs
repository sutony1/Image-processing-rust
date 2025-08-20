use opencv::prelude::*;
use anyhow::{Result, Context};

/// 图像缓冲区结构，用于与Win32 GDI交互
#[derive(Debug, Clone)]
pub struct ImageBuf {
    pub pixels: Vec<u8>,
    pub w: i32,
    pub h: i32,
}

impl ImageBuf {
    pub fn new(w: i32, h: i32) -> Self {
        Self {
            pixels: vec![0; (w * h * 4) as usize], // BGRA格式，每像素4字节
            w,
            h,
        }
    }
}

/// 将OpenCV Mat转换为BGRA像素格式，用于Win32显示
pub fn mat_to_bgra(mat: &opencv::core::Mat) -> Result<ImageBuf> {
    let rows = mat.rows();
    let cols = mat.cols();
    let channels = mat.channels();
    
    if rows <= 0 || cols <= 0 {
        return Err(anyhow::anyhow!("Invalid mat dimensions: {}x{}", cols, rows));
    }
    
    let mut result = ImageBuf::new(cols, rows);
    
    match channels {
        1 => {
            // 灰度图像，转换为BGRA
            for y in 0..rows {
                for x in 0..cols {
                    let gray_val = *mat.at_2d::<u8>(y, x)
                        .context("Failed to read gray pixel")?;
                    
                    let idx = ((y * cols + x) * 4) as usize;
                    result.pixels[idx] = gray_val;     // B
                    result.pixels[idx + 1] = gray_val; // G  
                    result.pixels[idx + 2] = gray_val; // R
                    result.pixels[idx + 3] = 255;      // A
                }
            }
        }
        3 => {
            // BGR图像，转换为BGRA
            for y in 0..rows {
                for x in 0..cols {
                    let pixel = mat.at_2d::<opencv::core::Vec3b>(y, x)
                        .context("Failed to read BGR pixel")?;
                    
                    let idx = ((y * cols + x) * 4) as usize;
                    result.pixels[idx] = pixel[0];     // B
                    result.pixels[idx + 1] = pixel[1]; // G
                    result.pixels[idx + 2] = pixel[2]; // R
                    result.pixels[idx + 3] = 255;      // A
                }
            }
        }
        4 => {
            // BGRA图像，直接复制
            for y in 0..rows {
                for x in 0..cols {
                    let pixel = mat.at_2d::<opencv::core::Vec4b>(y, x)
                        .context("Failed to read BGRA pixel")?;
                    
                    let idx = ((y * cols + x) * 4) as usize;
                    result.pixels[idx] = pixel[0];     // B
                    result.pixels[idx + 1] = pixel[1]; // G
                    result.pixels[idx + 2] = pixel[2]; // R
                    result.pixels[idx + 3] = pixel[3]; // A
                }
            }
        }
        _ => {
            return Err(anyhow::anyhow!("Unsupported channel count: {}", channels));
        }
    }
    
    Ok(result)
}

/// 将RGB8数据转换为OpenCV BGR Mat（备用函数）
pub fn rgb8_to_mat_bgr(rgb: &[u8], w: i32, h: i32) -> Result<opencv::core::Mat> {
    if rgb.len() != (w * h * 3) as usize {
        return Err(anyhow::anyhow!(
            "RGB data size mismatch: expected {}, got {}", 
            w * h * 3, rgb.len()
        ));
    }
    
    let mut bgr_data = Vec::with_capacity(rgb.len());
    
    // 转换RGB到BGR
    for chunk in rgb.chunks_exact(3) {
        bgr_data.push(chunk[2]); // B
        bgr_data.push(chunk[1]); // G
        bgr_data.push(chunk[0]); // R
    }
    
    let mat = opencv::core::Mat::new_rows_cols_with_data(
        h, w, &bgr_data
    ).context("Failed to create BGR Mat")?;
    
    Ok(mat.try_clone()?)
}

/// 保存调试图像到指定路径
pub fn save_debug_image(mat: &opencv::core::Mat, filename: &str) -> Result<()> {
    // 确保调试输出目录存在
    std::fs::create_dir_all("./debug_out")
        .context("Failed to create debug_out directory")?;
    
    let path = format!("./debug_out/{}", filename);
    opencv::imgcodecs::imwrite(&path, mat, &opencv::core::Vector::new())
        .context(format!("Failed to save debug image: {}", path))?;
    
    println!("调试图像已保存: {}", path);
    Ok(())
}

/// 保存图像到文件，支持多种格式和质量设置
/// 
/// # 参数
/// * `image` - 要保存的图像（BGR格式）
/// * `file_path` - 保存路径
/// * `quality` - JPEG质量（1-100，仅对JPEG格式有效）
/// 
/// # 支持格式
/// * .jpg/.jpeg - JPEG格式，支持质量设置
/// * .png - PNG格式，无损压缩
/// * .bmp - BMP格式，无压缩
/// * .tiff - TIFF格式
pub fn save_image(image: &opencv::core::Mat, file_path: &str, quality: i32) -> Result<()> {
    if image.empty() {
        return Err(anyhow::anyhow!("图像为空，无法保存"));
    }
    
    // 获取文件扩展名
    let path = std::path::Path::new(file_path);
    let extension = path.extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("jpg")
        .to_lowercase();
    
    // 根据扩展名设置保存参数
    let mut params = opencv::core::Vector::<i32>::new();
    
    match extension.as_str() {
        "jpg" | "jpeg" => {
            // JPEG格式：只设置质量参数，避免其他可能不支持的参数
            params.push(opencv::imgcodecs::IMWRITE_JPEG_QUALITY);
            params.push(quality.clamp(1, 100)); // 确保质量在1-100范围内
        },
        "png" => {
            // PNG格式：设置压缩级别（0-9，0=无压缩，9=最大压缩）
            let compression_level = if quality >= 90 { 0 } else if quality >= 70 { 3 } else { 6 };
            params.push(opencv::imgcodecs::IMWRITE_PNG_COMPRESSION);
            params.push(compression_level);
        },
        "tiff" => {
            // TIFF格式：设置压缩
            params.push(opencv::imgcodecs::IMWRITE_TIFF_COMPRESSION);
            params.push(1); // 使用LZW压缩
        },
        _ => {
            // 其他格式：使用默认参数
        }
    }
    
    println!("保存图像: {}x{} -> {}", image.cols(), image.rows(), file_path);
    println!("  格式: {}, 质量: {}, 参数数量: {}", extension, quality, params.len());
    
    // 检查目录是否存在
    if let Some(parent_dir) = path.parent() {
        if !parent_dir.exists() {
            println!("  创建目录: {:?}", parent_dir);
            std::fs::create_dir_all(parent_dir)
                .context("无法创建目录")?;
        }
    }
    
    // 尝试使用简化的文件名（移除中文字符可能导致的问题）
    let safe_file_path = if file_path.contains("自己写的色彩管理软件") {
        // 如果路径包含中文，尝试保存到当前目录
        let file_name = path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("resized_image.jpg");
        format!("./{}", file_name)
    } else {
        file_path.to_string()
    };
    
    println!("  实际保存路径: {}", safe_file_path);
    
    // 保存图像
    let success = opencv::imgcodecs::imwrite(&safe_file_path, image, &params)
        .context(format!("OpenCV imwrite 调用失败，路径: {}", safe_file_path))?;
    
    if success {
        println!("✅ 图像保存成功: {}", safe_file_path);
        
        // 检查文件是否真的被创建了
        if std::path::Path::new(&safe_file_path).exists() {
            let metadata = std::fs::metadata(&safe_file_path)
                .context("无法读取保存文件的元数据")?;
            println!("  文件大小: {} 字节", metadata.len());
        } else {
            return Err(anyhow::anyhow!("文件保存声称成功但文件不存在: {}", safe_file_path));
        }
        
        Ok(())
    } else {
        Err(anyhow::anyhow!("OpenCV imwrite 返回false，保存失败: {}", safe_file_path))
    }
}

/// 保存调整大小后的图像
/// 
/// # 参数
/// * `original_image` - 原始图像
/// * `new_width` - 新宽度
/// * `new_height` - 新高度
/// * `file_path` - 保存路径
/// * `quality` - 保存质量（1-100）
/// * `interpolation` - 插值算法
/// 
/// # 插值算法选项
/// * `INTER_NEAREST` - 最近邻插值（最快，质量最低）
/// * `INTER_LINEAR` - 双线性插值（平衡速度和质量，推荐）
/// * `INTER_CUBIC` - 双三次插值（较高质量，较慢）
/// * `INTER_LANCZOS4` - Lanczos插值（最高质量，最慢）
/// * `INTER_AREA` - 区域插值（适合缩小图像）
pub fn save_resized_image(
    original_image: &opencv::core::Mat,
    new_width: i32,
    new_height: i32,
    file_path: &str,
    quality: i32,
    interpolation: i32,
) -> Result<()> {
    if original_image.empty() {
        return Err(anyhow::anyhow!("原始图像为空，无法调整大小"));
    }
    
    if new_width <= 0 || new_height <= 0 {
        return Err(anyhow::anyhow!("无效的尺寸: {}x{}", new_width, new_height));
    }
    
    println!("开始调整图像大小...");
    println!("  原始尺寸: {}x{}", original_image.cols(), original_image.rows());
    println!("  目标尺寸: {}x{}", new_width, new_height);
    println!("  插值算法: {}", get_interpolation_name(interpolation));
    
    // 调整图像大小
    let mut resized = opencv::core::Mat::default();
    let size = opencv::core::Size::new(new_width, new_height);
    
    opencv::imgproc::resize(
        original_image,
        &mut resized,
        size,
        0.0,  // fx
        0.0,  // fy
        interpolation,
    )?;
    
    println!("✅ 图像大小调整完成: {}x{} -> {}x{}", 
        original_image.cols(), original_image.rows(), 
        new_width, new_height);
    
    // 保存调整后的图像
    save_image(&resized, file_path, quality)
}

/// 获取插值算法名称
fn get_interpolation_name(interpolation: i32) -> &'static str {
    match interpolation {
        opencv::imgproc::INTER_NEAREST => "最近邻插值",
        opencv::imgproc::INTER_LINEAR => "双线性插值",
        opencv::imgproc::INTER_CUBIC => "双三次插值",
        opencv::imgproc::INTER_AREA => "区域插值",
        opencv::imgproc::INTER_LANCZOS4 => "Lanczos插值",
        _ => "未知插值算法",
    }
}

/// 保存图像并返回文件大小信息
pub fn save_image_with_info(
    image: &opencv::core::Mat,
    file_path: &str,
    quality: i32,
) -> Result<(u64, String)> {
    // 保存图像
    save_image(image, file_path, quality)?;
    
    // 获取文件大小
    let metadata = std::fs::metadata(file_path)
        .map_err(|e| anyhow::anyhow!("无法获取文件元数据: {}", e))?;
    
    let file_size = metadata.len();
    let size_str = format_file_size(file_size);
    
    println!("📁 文件信息: {} ({})", file_path, size_str);
    
    Ok((file_size, size_str))
}

/// 格式化文件大小
fn format_file_size(bytes: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;
    
    if bytes < KB as u64 {
        format!("{} B", bytes)
    } else if bytes < MB as u64 {
        format!("{:.1} KB", bytes as f64 / KB)
    } else if bytes < GB as u64 {
        format!("{:.1} MB", bytes as f64 / MB)
    } else {
        format!("{:.1} GB", bytes as f64 / GB)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::core::*;
    
    #[test]
    fn test_mat_to_bgra_gray() -> Result<()> {
        // 创建一个3x3的灰度测试图像
        let mut gray_mat = Mat::zeros(3, 3, CV_8UC1)?.to_mat()?;
        *gray_mat.at_2d_mut::<u8>(1, 1)? = 128; // 中心像素设为128
        
        let result = mat_to_bgra(&gray_mat)?;
        
        assert_eq!(result.w, 3);
        assert_eq!(result.h, 3);
        assert_eq!(result.pixels.len(), 36); // 3*3*4
        
        // 检查中心像素
        let center_idx = (1 * 3 + 1) * 4;
        assert_eq!(result.pixels[center_idx], 128);     // B
        assert_eq!(result.pixels[center_idx + 1], 128); // G
        assert_eq!(result.pixels[center_idx + 2], 128); // R
        assert_eq!(result.pixels[center_idx + 3], 255); // A
        
        Ok(())
    }
    
    #[test]
    fn test_rgb8_to_mat_bgr() -> Result<()> {
        let rgb_data = vec![
            255, 0, 0,   // 红色
            0, 255, 0,   // 绿色  
            0, 0, 255,   // 蓝色
        ];
        
        let mat = rgb8_to_mat_bgr(&rgb_data, 3, 1)?;
        
        assert_eq!(mat.cols(), 3);
        assert_eq!(mat.rows(), 1);
        assert_eq!(mat.channels(), 3);
        
        // 检查颜色转换
        let pixel0 = mat.at_2d::<Vec3b>(0, 0)?;
        assert_eq!(pixel0[0], 0);   // B
        assert_eq!(pixel0[1], 0);   // G
        assert_eq!(pixel0[2], 255); // R (原RGB的红色)
        
        Ok(())
    }
}
