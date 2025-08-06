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
