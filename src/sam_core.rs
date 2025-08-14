//! SAM核心实现 - 直接集成版本
//! 基于sam_onnx的实现，但直接包含在img_view_cv中

use anyhow::{bail, Result};
use image::{DynamicImage, GenericImageView, RgbImage, Pixel};
use ndarray::{arr0, Array, Array2, Array4, s, CowArray};
use ort::{
    Session, SessionBuilder, GraphOptimizationLevel, Environment,
    Value,
};
use std::sync::Arc;

pub struct SamPaths<'a> {
    pub encoder_onnx: &'a str,
    pub decoder_onnx: &'a str,
}

pub struct SamMask {
    /// 阈值化后的 [H,W] u8 (0/255)，按原图大小
    pub mask_u8: Array2<u8>,
    /// IOU 置信（如果有的话）
    pub iou: Option<f32>,
}

pub struct SamResult {
    /// 多个mask结果
    pub masks: Vec<Array2<f32>>, // 保持原始概率值
    /// 对应的IOU分数
    pub scores: Vec<f32>,
}

/// SAM 的 ONNX 推理封装（ViT-H, 1024 输入）
pub struct SamOnnx {
    encoder: Session,
    decoder: Session,
}

impl SamOnnx {
    pub fn new(paths: SamPaths) -> Result<Self> {
        println!("初始化SAM模型...");
        
        // 检查模型文件
        if !std::path::Path::new(paths.encoder_onnx).exists() {
            bail!("Encoder模型文件不存在: {}", paths.encoder_onnx);
        }
        if !std::path::Path::new(paths.decoder_onnx).exists() {
            bail!("Decoder模型文件不存在: {}", paths.decoder_onnx);
        }
        
        // 初始化ort环境
        let env = Arc::new(Environment::builder()
            .with_name("sam_onnx")
            .build()?);
        
        // 创建encoder session
        let encoder = SessionBuilder::new(&env)?  
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .with_model_from_file(paths.encoder_onnx)?;
            
        // 创建decoder session
        let decoder = SessionBuilder::new(&env)?  
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .with_model_from_file(paths.decoder_onnx)?;
        
        Ok(Self { encoder, decoder })
    }

    /// 入口：给定图片与一个提示点（x,y，原图坐标；正点/负点），输出分割 mask
    pub fn run_point_prompt(
        &self,
        img: &DynamicImage,
        point_xy_in_original: (f32, f32),
        is_positive: bool,
        prob_threshold: f32,
    ) -> Result<SamMask> {
        let (orig_w, orig_h) = img.dimensions();
        
        // 预处理
        let PreprocOut {
            nchw_1024,
            scale,
            pad_hw,
            resized_hw: _,
        } = preprocess_image(img)?;
        
        // 运行encoder推理
        println!("  运行encoder推理...");
        
        // 转换为HWC格式 (SAM encoder期望HWC输入)
        let mut hwc_data = Vec::with_capacity(1024 * 1024 * 3);
        for y in 0..1024 {
            for x in 0..1024 {
                hwc_data.push(nchw_1024[(0, 0, y, x)]); // R
                hwc_data.push(nchw_1024[(0, 1, y, x)]); // G  
                hwc_data.push(nchw_1024[(0, 2, y, x)]); // B
            }
        }
        
        // 创建输入tensor
        let hwc_array = CowArray::from(Array::from_shape_vec(
            vec![1024, 1024, 3],
            hwc_data
        )?.into_dyn());
        
        let img_input = Value::from_array(self.encoder.allocator(), &hwc_array)?;
        let enc_outputs = self.encoder.run(vec![img_input])?;
        
        // 获取image embeddings (第一个输出)
        let emb_value = &enc_outputs[0];
        let emb_tensor = emb_value.try_extract::<f32>()?;
        let emb_view = emb_tensor.view();
        let emb_shape = emb_view.shape();
        
        if emb_shape.len() != 4 || emb_shape[0] != 1 || emb_shape[1] != 256 || emb_shape[2] != 64 || emb_shape[3] != 64 {
            bail!("Unexpected encoder output shape: {:?} (expect [1,256,64,64])", emb_shape);
        }
        
        let emb_data: Vec<f32> = emb_view.as_slice().unwrap().to_vec();
        let _emb = Array4::from_shape_vec(
            (1, 256, 64, 64),
            emb_data.clone()
        )?;
        
        // 构造decoder输入
        let (px, py) = point_xy_in_original;
        let mapped_x = px * scale + pad_hw.1 as f32;
        let mapped_y = py * scale + pad_hw.0 as f32;
        
        // 运行decoder推理
        println!("  运行decoder推理...");
        
        // 构造decoder输入
        let point_coords = Array::from_shape_vec((1, 1, 2), vec![mapped_x, mapped_y])?;
        let point_labels = Array::from_shape_vec((1, 1), vec![if is_positive { 1f32 } else { 0f32 }])?;
        let mask_input = Array4::<f32>::zeros((1, 1, 256, 256));
        let has_mask_input = arr0(0f32).into_shape((1,))?;
        let orig_im_size = Array::from_vec(vec![orig_h as f32, orig_w as f32]).into_shape((2,))?;
        
        // 准备decoder输入 - 保持所有数组的生命周期
        let emb_array = Array::from_shape_vec(
            vec![1, 256, 64, 64],
            emb_data.clone()
        )?;
        let emb_cow = CowArray::from(emb_array.into_dyn());
        
        let coords_cow = CowArray::from(point_coords.into_dyn());
        let labels_cow = CowArray::from(point_labels.into_dyn());
        let mask_cow = CowArray::from(mask_input.into_dyn());
        let has_mask_cow = CowArray::from(has_mask_input.into_dyn());
        let orig_size_cow = CowArray::from(orig_im_size.into_dyn());
        
        // 创建Value对象
        let emb_input = Value::from_array(
            self.decoder.allocator(),
            &emb_cow
        )?;
        
        let coords_input = Value::from_array(
            self.decoder.allocator(),
            &coords_cow
        )?;
        
        let labels_input = Value::from_array(
            self.decoder.allocator(),
            &labels_cow
        )?;
        
        let mask_in = Value::from_array(
            self.decoder.allocator(),
            &mask_cow
        )?;
        
        let has_mask = Value::from_array(
            self.decoder.allocator(),
            &has_mask_cow
        )?;
        
        let orig_size = Value::from_array(
            self.decoder.allocator(),
            &orig_size_cow
        )?;
        
        // 运行decoder (按照输入顺序)
        let dec_outs = self.decoder.run(vec![
            emb_input,
            coords_input,
            labels_input,
            mask_in,
            has_mask,
            orig_size
        ])?;
        
        // 获取mask输出 (第一个输出)
        let masks_value = &dec_outs[0];
        let masks_tensor = masks_value.try_extract::<f32>()?;
        let masks_view = masks_tensor.view();
        let mask_shape = masks_view.shape();
        
        if mask_shape.len() != 4 {
            bail!("Unexpected decoder mask shape: {:?}", mask_shape);
        }
        
        // 取第一个mask [256, 256]
        let low_res = Array4::from_shape_vec(
            (mask_shape[0], mask_shape[1], mask_shape[2], mask_shape[3]),
            masks_view.as_slice().unwrap().to_vec()
        )?;
        let low_res = low_res.slice(s![0, 0, .., ..]).to_owned();
        
        // 上采样到原图大小
        let mask = bilinear_resize_2d(&low_res, orig_h as usize, orig_w as usize);
        let mask = mask.mapv(|p| if p >= prob_threshold { 255u8 } else { 0u8 });
        
        // 获取IOU分数 (第二个输出，如果有的话)
        let iou = if dec_outs.len() > 1 {
            dec_outs[1].try_extract::<f32>()
                .ok()
                .and_then(|iou_tensor| {
                    iou_tensor.view().as_slice().and_then(|s| s.first().copied())
                })
        } else {
            None
        };
        
        Ok(SamMask { 
            mask_u8: mask,
            iou
        })
    }

    /// BGR 字节缓冲输入的推理接口
    pub fn run_point_prompt_bgr_bytes(
        &self,
        bgr_bytes: &[u8],
        width: usize,
        height: usize,
        stride: usize,
        point_xy_in_original: (f32, f32),
        is_positive: bool,
        prob_threshold: f32,
    ) -> Result<SamMask> {
        // 验证输入参数
        if stride < width * 3 {
            bail!("Stride ({}) must be at least width * 3 ({})", stride, width * 3);
        }
        if bgr_bytes.len() < height * stride {
            bail!("BGR bytes length ({}) insufficient for {}x{} image with stride {}", 
                  bgr_bytes.len(), width, height, stride);
        }

        // 将 BGR 字节数据转换为 RGB image
        let mut rgb_data = Vec::with_capacity(width * height * 3);
        for y in 0..height {
            let row_start = y * stride;
            for x in 0..width {
                let pixel_start = row_start + x * 3;
                // BGR -> RGB
                rgb_data.push(bgr_bytes[pixel_start + 2]); // R
                rgb_data.push(bgr_bytes[pixel_start + 1]); // G
                rgb_data.push(bgr_bytes[pixel_start]);     // B
            }
        }

        // 创建 RgbImage
        let rgb_img = RgbImage::from_raw(width as u32, height as u32, rgb_data)
            .ok_or_else(|| anyhow::anyhow!("Failed to create RGB image from BGR data"))?;
        
        // 转换为 DynamicImage
        let dynamic_img = DynamicImage::ImageRgb8(rgb_img);

        // 调用原有的推理函数
        self.run_point_prompt(&dynamic_img, point_xy_in_original, is_positive, prob_threshold)
    }

    /// 单独运行encoder，返回image embeddings
    pub fn encode_image(&self, img: &DynamicImage) -> Result<Array4<f32>> {
        // 预处理
        let PreprocOut {
            nchw_1024,
            scale: _,
            pad_hw: _,
            resized_hw: _,
        } = preprocess_image(img)?;
        
        // 运行encoder推理
        println!("  运行encoder推理...");
        
        // 转换为HWC格式 (SAM encoder期望HWC输入)
        let mut hwc_data = Vec::with_capacity(1024 * 1024 * 3);
        for y in 0..1024 {
            for x in 0..1024 {
                hwc_data.push(nchw_1024[(0, 0, y, x)]); // R
                hwc_data.push(nchw_1024[(0, 1, y, x)]); // G  
                hwc_data.push(nchw_1024[(0, 2, y, x)]); // B
            }
        }
        
        let hwc_array = CowArray::from(Array::from_shape_vec(
            vec![1024, 1024, 3],
            hwc_data
        )?.into_dyn());
        
        let img_input = Value::from_array(self.encoder.allocator(), &hwc_array)?;
        let enc_outputs = self.encoder.run(vec![img_input])?;
        
        // 获取image embeddings (第一个输出)
        let emb_value = &enc_outputs[0];
        let emb_tensor = emb_value.try_extract::<f32>()?;
        let emb_view = emb_tensor.view();
        let emb_shape = emb_view.shape();
        
        if emb_shape.len() != 4 || emb_shape[0] != 1 || emb_shape[1] != 256 || emb_shape[2] != 64 || emb_shape[3] != 64 {
            bail!("Unexpected encoder output shape: {:?} (expect [1,256,64,64])", emb_shape);
        }
        
        let emb_data: Vec<f32> = emb_view.as_slice().unwrap().to_vec();
        let emb = Array4::from_shape_vec(
            (1, 256, 64, 64),
            emb_data
        )?;
        
        Ok(emb)
    }

    /// 使用预计算的embeddings运行decoder (多mask输出)
    pub fn predict_with_embeddings_multi(
        &self,
        embeddings: &Array4<f32>,
        img_size: (u32, u32),
        point_xy: (f32, f32),
        is_positive: bool,
    ) -> Result<SamResult> {
        let (orig_w, orig_h) = img_size;
        
        // 正确计算SAM预处理的坐标映射
        let long = orig_w.max(orig_h) as f32;
        let scale = 1024f32 / long;
        let new_w = (orig_w as f32 * scale).round() as u32;
        let new_h = (orig_h as f32 * scale).round() as u32;
        
        // SAM将图像放在左上角，右侧和底部填充0
        let pad_hw = (0u32, 0u32); // SAM标准预处理：左上角对齐，无padding偏移
        
        // 构造decoder输入 - 重要：坐标映射要与预处理保持一致
        let (px, py) = point_xy;
        let mapped_x = px * scale + pad_hw.1 as f32;
        let mapped_y = py * scale + pad_hw.0 as f32;
        
        println!("📍 坐标映射: ({:.1}, {:.1}) -> ({:.1}, {:.1}), scale: {:.3}", 
                 px, py, mapped_x, mapped_y, scale);
        
        let point_coords = Array::from_shape_vec((1, 1, 2), vec![mapped_x, mapped_y])?;
        let point_labels = Array::from_shape_vec((1, 1), vec![if is_positive { 1f32 } else { 0f32 }])?;
        let mask_input = Array4::<f32>::zeros((1, 1, 256, 256));
        let has_mask_input = arr0(0f32).into_shape((1,))?;
        let orig_im_size = Array::from_vec(vec![orig_h as f32, orig_w as f32]).into_shape((2,))?;
        
        // 准备decoder输入 - 保持所有数组的生命周期
        let emb_array = Array::from_shape_vec(
            vec![1, 256, 64, 64],
            embeddings.as_slice().unwrap().to_vec()
        )?;
        let emb_cow = CowArray::from(emb_array.into_dyn());
        
        let coords_cow = CowArray::from(point_coords.into_dyn());
        let labels_cow = CowArray::from(point_labels.into_dyn());
        let mask_cow = CowArray::from(mask_input.into_dyn());
        let has_mask_cow = CowArray::from(has_mask_input.into_dyn());
        let orig_size_cow = CowArray::from(orig_im_size.into_dyn());
        
        // 创建Value对象
        let emb_input = Value::from_array(self.decoder.allocator(), &emb_cow)?;
        let coords_input = Value::from_array(self.decoder.allocator(), &coords_cow)?;
        let labels_input = Value::from_array(self.decoder.allocator(), &labels_cow)?;
        let mask_in = Value::from_array(self.decoder.allocator(), &mask_cow)?;
        let has_mask = Value::from_array(self.decoder.allocator(), &has_mask_cow)?;
        let orig_size = Value::from_array(self.decoder.allocator(), &orig_size_cow)?;
        
        // 运行decoder - 使用命名输入以确保顺序正确
        // 注意：ort 1.16可能不支持inputs!宏，我们需要按正确顺序提供
        println!("🚀 运行decoder推理...");
        let dec_outs = self.decoder.run(vec![
            emb_input,      // image_embeddings
            coords_input,   // point_coords
            labels_input,   // point_labels
            mask_in,        // mask_input
            has_mask,       // has_mask_input
            orig_size       // orig_im_size
        ])?;
        
        // 获取mask输出 - 注意：不同的ONNX导出可能有不同的输出顺序
        println!("📦 Decoder输出数量: {}", dec_outs.len());
        
        // 查找正确的mask输出（应该是256x256的）
        let mut masks_output = None;
        let mut iou_output = None;
        
        for (i, output) in dec_outs.iter().enumerate() {
            if let Ok(tensor) = output.try_extract::<f32>() {
                let tensor_view = tensor.view();
                let shape = tensor_view.shape();
                println!("  输出[{}]形状: {:?}", i, shape);
                
                // mask输出应该是 [1, N, 256, 256] 其中N=1或3
                if shape.len() == 4 && shape[2] == 256 && shape[3] == 256 {
                    masks_output = Some((i, tensor));
                }
                // IOU输出应该是 [1, N] 其中N=1或3
                else if shape.len() == 2 && shape[0] == 1 {
                    iou_output = Some((i, tensor));
                }
            }
        }
        
        let (mask_idx, masks_tensor) = masks_output
            .ok_or_else(|| anyhow::anyhow!("未找到256x256的mask输出"))?;
        
        let masks_view = masks_tensor.view();
        let mask_shape = masks_view.shape();
        
        println!("🔍 使用输出[{}]作为mask，形状: {:?}", mask_idx, mask_shape);
        
        // 处理mask输出
        // 注意：ONNX导出的SAM可能是[1,1,256,256]（单mask）或[1,3,256,256]（多mask）
        let low_res_4d = Array4::from_shape_vec(
            (mask_shape[0], mask_shape[1], mask_shape[2], mask_shape[3]),
            masks_view.as_slice().unwrap().to_vec()
        )?;
        
        let num_masks = mask_shape[1];
        let mut masks = Vec::new();
        
        for i in 0..num_masks {
            let low_res = low_res_4d.slice(s![0, i, .., ..]).to_owned();
            
            // 上采样到原图大小，保持原始概率值
            let mask = bilinear_resize_2d(&low_res, orig_h as usize, orig_w as usize);
            masks.push(mask);
        }
        
        // 获取IOU分数
        let mut scores = Vec::new();
        if let Some((iou_idx, iou_tensor)) = iou_output {
            println!("🎯 使用输出[{}]作为IOU分数", iou_idx);
            let iou_view = iou_tensor.view();
            if let Some(iou_slice) = iou_view.as_slice() {
                for i in 0..num_masks.min(iou_slice.len()) {
                    scores.push(iou_slice[i]);
                }
            }
        }
        
        // 如果没有scores，填充默认值
        while scores.len() < masks.len() {
            scores.push(0.0);
        }
        
        Ok(SamResult { 
            masks,
            scores
        })
    }
}

struct PreprocOut {
    /// [1,3,1024,1024] float32 RGB NCHW，归一化
    nchw_1024: Array4<f32>,
    /// 原图 -> 1024 的缩放比例（等比缩放）
    scale: f32,
    /// (pad_top, pad_left)
    pad_hw: (u32, u32),
    /// resize 后尺寸 (h,w)（未 pad 前）
    resized_hw: (u32, u32),
}

/// SAM 预处理：最长边缩放到 1024，保持比例，之后右侧/下侧进行零填充到 1024×1024；RGB、mean/std 归一化、NCHW
fn preprocess_image(img: &DynamicImage) -> Result<PreprocOut> {
    let (w, h) = img.dimensions();
    let long = w.max(h) as f32;
    let scale = 1024f32 / long;
    let new_w = (w as f32 * scale).round() as u32;
    let new_h = (h as f32 * scale).round() as u32;

    // resize 到 new_w x new_h
    let resized = img.resize_exact(new_w, new_h, image::imageops::FilterType::Lanczos3);

    // 创建 1024x1024 黑色背景
    let mut padded = image::RgbImage::new(1024, 1024);
    
    // 复制 resized 到左上角
    for y in 0..new_h {
        for x in 0..new_w {
            let pixel = resized.get_pixel(x, y);
            // 直接从像素获取RGB值
            let rgba = pixel.to_rgba();
            let rgb = image::Rgb([rgba[0], rgba[1], rgba[2]]);
            padded.put_pixel(x, y, rgb);
        }
    }

    // 转换为 NCHW float32，并进行归一化
    let mean = [123.675, 116.28, 103.53];
    let std = [58.395, 57.12, 57.375];
    
    let mut nchw = Array4::<f32>::zeros((1, 3, 1024, 1024));
    for y in 0..1024 {
        for x in 0..1024 {
            let pixel = padded.get_pixel(x, y);
            nchw[(0, 0, y as usize, x as usize)] = (pixel[0] as f32 - mean[0]) / std[0];
            nchw[(0, 1, y as usize, x as usize)] = (pixel[1] as f32 - mean[1]) / std[1];
            nchw[(0, 2, y as usize, x as usize)] = (pixel[2] as f32 - mean[2]) / std[2];
        }
    }

    Ok(PreprocOut {
        nchw_1024: nchw,
        scale,
        pad_hw: (0, 0),     // 我们把 pad 放在右/下，所以 top/left 为 0
        resized_hw: (new_h, new_w),
    })
}

/// 双线性插值把 [H_in,W_in] -> [H_out,W_out]
fn bilinear_resize_2d(src: &Array2<f32>, h_out: usize, w_out: usize) -> Array2<f32> {
    let (h_in, w_in) = src.dim();
    if h_in == h_out && w_in == w_out {
        return src.clone();
    }
    
    let mut dst = Array2::<f32>::zeros((h_out, w_out));
    let scale_y = (h_in as f32) / (h_out as f32);
    let scale_x = (w_in as f32) / (w_out as f32);
    
    for y_out in 0..h_out {
        for x_out in 0..w_out {
            let y_in = y_out as f32 * scale_y;
            let x_in = x_out as f32 * scale_x;
            
            let y0 = y_in.floor() as usize;
            let x0 = x_in.floor() as usize;
            let y1 = (y0 + 1).min(h_in - 1);
            let x1 = (x0 + 1).min(w_in - 1);
            
            let dy = y_in - y0 as f32;
            let dx = x_in - x0 as f32;
            
            let v00 = src[(y0, x0)];
            let v01 = src[(y0, x1)];
            let v10 = src[(y1, x0)];
            let v11 = src[(y1, x1)];
            
            let v0 = v00 * (1.0 - dx) + v01 * dx;
            let v1 = v10 * (1.0 - dx) + v11 * dx;
            let v = v0 * (1.0 - dy) + v1 * dy;
            
            dst[(y_out, x_out)] = v;
        }
    }
    
    dst
}