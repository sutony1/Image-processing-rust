//! SAM核心实现 - 完整版本
//! 从sam_onnx_rust复制的完整实现，确保坐标映射精度

use anyhow::Result;
use ndarray::{Array, IxDyn};
use ort::{Environment, Session, SessionBuilder, Value};
use opencv::{core, imgproc};
use opencv::prelude::{MatTraitConst, MatTrait, Mat};
use std::sync::Arc;

// 从sam_onnx_rust复制的SAMProcessor实现
pub struct SAMProcessor {
    encoder: Session,
    decoder: Session,
    image_embeddings: Option<Array<f32, IxDyn>>,
    // 原图尺寸
    orig_width: f32,
    orig_height: f32,
    // 等比缩放后的有效尺寸（在 1024×1024 预处理坐标系内）
    new_w: i32,
    new_h: i32,
    // 1024 / max(orig_w, orig_h)
    scale: f32,
    // 兼容不同导出脚本：true = 传 (new_h,new_w)，false = 传 (orig_h,orig_w)
    use_new_size_for_decoder: bool,
}

impl SAMProcessor {
    pub fn new() -> Result<Self> {
        let env = Arc::new(Environment::builder().with_name("SAM").build()?);

        // 解析模型路径：优先环境变量 -> 可执行目录 -> CWD -> 工程根
        fn resolve_model_path(file: &str) -> std::path::PathBuf {
            if let Ok(dir) = std::env::var("SAM_MODEL_DIR") {
                let p = std::path::Path::new(&dir).join(file);
                if p.exists() { return p; }
            }
            if let Ok(exe) = std::env::current_exe() {
                if let Some(dir) = exe.parent() {
                    let p = dir.join(file);
                    if p.exists() { return p; }
                }
            }
            if let Ok(cwd) = std::env::current_dir() {
                let p = cwd.join(file);
                if p.exists() { return p; }
            }
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(file)
        }

        let encoder_path = resolve_model_path("vit_t_encoder.onnx");
        let decoder_path = resolve_model_path("vit_t_decoder.onnx");

        println!("加载SAM模型:\n  Encoder: {}\n  Decoder: {}",
            encoder_path.to_string_lossy(),
            decoder_path.to_string_lossy());

        let encoder = SessionBuilder::new(&env)?
            .with_model_from_file(&encoder_path)?;
        let decoder = SessionBuilder::new(&env)?
            .with_model_from_file(&decoder_path)?;

        Ok(Self {
            encoder,
            decoder,
            image_embeddings: None,
            orig_width: 0.0,
            orig_height: 0.0,
            new_w: 0,
            new_h: 0,
            scale: 1.0,
            use_new_size_for_decoder: true, // 多数 ONNX 导出用 new_h/new_w
        })
    }

    pub fn encode_image(&mut self, mat: &core::Mat) -> Result<()> {
        self.orig_width  = mat.cols() as f32;
        self.orig_height = mat.rows() as f32;

        let (embeddings, new_w, new_h, scale) = self.get_image_embeddings(mat)?;
        self.image_embeddings = Some(embeddings);
        self.new_w = new_w;
        self.new_h = new_h;
        self.scale = scale;
        Ok(())
    }

    pub fn decode_prompt(&self, x1: f32, y1: f32, x2: f32, y2: f32)
        -> Result<(Vec<u8>, usize, usize)>
    {
        let embeddings = self.image_embeddings.as_ref()
            .ok_or_else(|| anyhow::anyhow!("图像未编码，请先调用 encode_image"))?;

        println!("🔍 SAM分割调试信息:");
        println!("  输入坐标: ({:.2}, {:.2}) -> ({:.2}, {:.2})", x1, y1, x2, y2);
        println!("  原图尺寸: {}x{}", self.orig_width, self.orig_height);
        println!("  缩放后尺寸: {}x{}", self.new_w, self.new_h);
        println!("  缩放比例: {:.6}", self.scale);

        // —— 坐标映射到 1024 预处理坐标系（仅按 scale，不加 0.5）——
        let clampf = |v: f32, maxv: i32| v.max(0.0).min((maxv - 1).max(0) as f32);
        let sx1 = clampf(x1 * self.scale, self.new_w);
        let sy1 = clampf(y1 * self.scale, self.new_h);
        let sx2 = clampf(x2 * self.scale, self.new_w);
        let sy2 = clampf(y2 * self.scale, self.new_h);

        println!("  预处理坐标: ({:.2}, {:.2}) -> ({:.2}, {:.2})", sx1, sy1, sx2, sy2);
        
        // 计算输入框和预处理框的宽高比
        let input_width = (x2 - x1).abs();
        let input_height = (y2 - y1).abs();
        let preprocess_width = (sx2 - sx1).abs();
        let preprocess_height = (sy2 - sy1).abs();
        println!("  输入框尺寸: {:.2}x{:.2}, 宽高比: {:.3}", input_width, input_height, input_width / input_height);
        println!("  预处理框尺寸: {:.2}x{:.2}, 宽高比: {:.3}", preprocess_width, preprocess_height, preprocess_width / preprocess_height);

        let point_coords = Array::from_shape_vec((1, 2, 2), vec![
            sx1, sy1,  // box 左上
            sx2, sy2,  // box 右下
        ])?.into_dyn();

        // Box 两端在 SAM 里用 label [2,3]
        let point_labels = Array::from_shape_vec((1, 2), vec![2.0_f32, 3.0_f32])?.into_dyn();

        // 无先验 mask
        let mask_input = Array::from_shape_vec((1, 1, 256, 256), vec![0.0_f32; 256 * 256])?.into_dyn();
        let has_mask_input = Array::from_vec(vec![0.0_f32]).into_dyn();

        // orig_im_size：优先传 new_h/new_w；如遇到导出差异，可改为 orig_h/orig_w
        let (oh, ow) = (self.orig_height, self.orig_width);
        let (nh, nw) = (self.new_h as f32, self.new_w as f32);
        let ois = if self.use_new_size_for_decoder {
            Array::from_vec(vec![nh, nw]).into_dyn()
        } else {
            Array::from_vec(vec![oh, ow]).into_dyn()
        };

        // —— 运行 decoder ——
        // 注意：不同导出版本输出形状可能是 (1,1,256,256) 或 (1,N,256,256)
        let embeddings_layout     = embeddings.as_standard_layout();
        let point_coords_layout   = point_coords.as_standard_layout();
        let point_labels_layout   = point_labels.as_standard_layout();
        let mask_input_layout     = mask_input.as_standard_layout();
        let has_mask_input_layout = has_mask_input.as_standard_layout();
        let ois_layout            = ois.as_standard_layout();

        println!("  输入tensor形状:");
        println!("    embeddings: {:?}", embeddings.shape());
        println!("    point_coords: {:?}", point_coords.shape());
        println!("    point_labels: {:?}", point_labels.shape());
        println!("    mask_input: {:?}", mask_input.shape());
        println!("    has_mask_input: {:?}", has_mask_input.shape());
        println!("    orig_im_size: {:?}", ois.shape());

        let decoder_inputs = vec![
            Value::from_array(self.decoder.allocator(), &embeddings_layout)?,
            Value::from_array(self.decoder.allocator(), &point_coords_layout)?,
            Value::from_array(self.decoder.allocator(), &point_labels_layout)?,
            Value::from_array(self.decoder.allocator(), &mask_input_layout)?,
            Value::from_array(self.decoder.allocator(), &has_mask_input_layout)?,
            Value::from_array(self.decoder.allocator(), &ois_layout)?,
        ];

        let outputs = self.decoder.run(decoder_inputs)?;
        println!("  输出tensor数量: {}", outputs.len());
        
        // 输出0通常是 logits（不是 sigmoid 后概率）
        let masks_tensor = outputs.get(0).unwrap().try_extract::<f32>()?;
        let masks = masks_tensor.view(); // 动态维度
        
        println!("  masks_tensor形状: {:?}", masks.shape());

        // 若有 iou_scores，选得分最高的那张
        let mut best_idx: usize = 0;
        if let Some(iou_any) = outputs.get(1) {
            if let Ok(iou) = iou_any.try_extract::<f32>() {
                if let Some(scores) = iou.view().as_slice() {
                    if !scores.is_empty() {
                        let mut maxi = 0usize;
                        let mut maxv = scores[0];
                        for (i, &v) in scores.iter().enumerate() {
                            if v > maxv { maxv = v; maxi = i; }
                        }
                        best_idx = maxi;
                    }
                }
            }
        }

        // 解析形状，兼容 (1,1,256,256) / (1,N,256,256) / (1,256,256)
        let shape = masks.shape();
        let (mut h, mut w, mut base) = (256i32, 256i32, 0usize);
        println!("  解析mask形状: {:?}", shape);
        match shape {
            // (1,1,256,256)
            [1, 1, hh, ww] => { 
                h = *hh as i32; w = *ww as i32; base = 0; 
                println!("    匹配模式: (1,1,{},{})", h, w);
            }
            // (1,N,256,256)
            [1, n, hh, ww] => { 
                h = *hh as i32; w = *ww as i32;
                base = best_idx.min(*n as usize - 1) * ((*hh as usize) * (*ww as usize)); 
                println!("    匹配模式: (1,{},{},{}), best_idx={}, base={}", n, h, w, best_idx, base);
            }
            // (1,256,256)
            [1, hh, ww]    => { 
                h = *hh as i32; w = *ww as i32; base = 0; 
                println!("    匹配模式: (1,{},{})", h, w);
            }
            _ => {
                println!("    警告: 未识别的形状模式: {:?}", shape);
            }
        }
        println!("    最终mask尺寸: {}x{}, base偏移: {}", h, w, base);

        let data = masks.as_slice().unwrap_or(&[]);
        let stride = (h as usize) * (w as usize);
        let start = base.min(data.len().saturating_sub(stride));
        
        println!("  mask数据信息:");
        println!("    数据长度: {}, stride: {}, start: {}", data.len(), stride, start);
        println!("    有效数据范围: [{}..{}]", start, start + stride);

        // 创建256x256的mask数组
        let mut mask_256_data = vec![0.0f32; (h * w) as usize];
        let mut positive_count = 0;
        for yy in 0..h {
            for xx in 0..w {
                let logit = data[start + (yy as usize) * (w as usize) + (xx as usize)];
                let is_positive = logit > 0.0;
                if is_positive { positive_count += 1; }
                mask_256_data[(yy * w + xx) as usize] = if is_positive { 1.0 } else { 0.0 };
            }
        }
        println!("    256x256 mask统计: 总像素={}, 正像素={}, 正像素比例={:.2}%", 
                h * w, positive_count, (positive_count as f32 / (h * w) as f32) * 100.0);

        // 创建OpenCV Mat
        let mut mask_256 = Mat::new_rows_cols_with_default(h, w, core::CV_32F, core::Scalar::all(0.0))?;
        unsafe {
            let total = (h * w) as usize;
            std::ptr::copy_nonoverlapping(mask_256_data.as_ptr(), mask_256.data_mut() as *mut f32, total);
        }

        println!("    创建mask Mat: {}x{}", h, w);

        // 直接使用模型输出的mask，不需要上采样
        let mut mask_to_resize = mask_256;
        println!("    直接使用模型输出的mask尺寸: {}x{}", h, w);

        // 如果模型输出的尺寸与期望的new_w×new_h不匹配，需要调整
        if h != self.new_h || w != self.new_w {
            println!("    警告: 模型输出尺寸({}x{})与期望尺寸({}x{})不匹配", h, w, self.new_w, self.new_h);
            println!("    将直接使用模型输出尺寸进行后续处理");
        }

        // 直接使用mask，不需要裁剪ROI
        let mask_unpad = mask_to_resize;
        println!("    跳过ROI裁剪，直接使用完整mask");

        // 还原到原图大小
        let mut mask_orig = Mat::default();
        imgproc::resize(
            &mask_unpad, &mut mask_orig,
            core::Size::new(self.orig_width as i32, self.orig_height as i32),
            0.0, 0.0, imgproc::INTER_LINEAR
        )?;
        println!("    还原到原图尺寸: {}x{}", self.orig_width, self.orig_height);

        // 导出 0/1 Vec<u8>
        let ow = self.orig_width as i32;
        let oh = self.orig_height as i32;
        let mut out = vec![0u8; (ow * oh) as usize];
        
        // 从mask_orig读取数据
        let mask_data = unsafe { std::slice::from_raw_parts(mask_orig.data() as *const f32, (ow * oh) as usize) };
        let mut final_positive_count = 0;
        for i in 0..(ow * oh) as usize {
            let is_positive = mask_data[i] > 0.5;
            if is_positive { final_positive_count += 1; }
            out[i] = if is_positive { 1 } else { 0 };
        }
        
        println!("  最终输出统计:");
        println!("    输出尺寸: {}x{}", ow, oh);
        println!("    总像素: {}", ow * oh);
        println!("    正像素: {}", final_positive_count);
        println!("    正像素比例: {:.2}%", (final_positive_count as f32 / (ow * oh) as f32) * 100.0);
        println!("✅ SAM分割完成");
        
        Ok((out, ow as usize, oh as usize))
    }

    fn get_image_embeddings(&self, mat: &core::Mat) -> Result<(Array<f32, IxDyn>, i32, i32, f32)> {
        // —— 保持比例缩放 + pad 到 1024×1024 —— 
        let orig_w = mat.cols();
        let orig_h = mat.rows();
        let scale = 1024f32 / (orig_w.max(orig_h) as f32);
        // floor 保持与坐标一致
        let new_w = ((orig_w as f32) * scale).floor() as i32;
        let new_h = ((orig_h as f32) * scale).floor() as i32;

        println!("🔧 图像预处理调试:");
        println!("  原图尺寸: {}x{}", orig_w, orig_h);
        println!("  缩放比例: {:.6}", scale);
        println!("  缩放后尺寸: {}x{}", new_w, new_h);
        println!("  宽高比变化: 原图={:.3}, 缩放后={:.3}", 
                orig_w as f32 / orig_h as f32, new_w as f32 / new_h as f32);

        // 直接使用BGR数据，避免复杂的颜色转换
        let mut resized = Mat::default();
        imgproc::resize(mat, &mut resized, core::Size::new(new_w, new_h), 0.0, 0.0, imgproc::INTER_LINEAR)?;

        // float32
        let mut resized_f32 = Mat::default();
        resized.convert_to(&mut resized_f32, core::CV_32F, 1.0, 0.0)?;

        // —— 构造 NCHW = 1×3×1024×1024：先全 0（pad=0），只在有效区域写入"归一化后的像素" —— 
        let mut input = Array::zeros((1, 3, 1024usize, 1024usize)).into_dyn();
        // mean/std（BGR 顺序，因为我们保持BGR格式）
        const MEAN: [f32; 3] = [103.53, 116.28, 123.675]; // BGR顺序
        const STD : [f32; 3] = [57.375, 57.12 , 58.395 ]; // BGR顺序

        for y in 0..new_h {
            for x in 0..new_w {
                let p = *resized_f32.at_2d::<core::Vec3f>(y, x)?; // BGR in 0..255
                // 注意：OpenCV Vec3f 顺序就是 (B,G,R) 因为我们保持BGR格式
                input[[0, 0, y as usize, x as usize]] = (p[0] - MEAN[0]) / STD[0];
                input[[0, 1, y as usize, x as usize]] = (p[1] - MEAN[1]) / STD[1];
                input[[0, 2, y as usize, x as usize]] = (p[2] - MEAN[2]) / STD[2];
            }
        }
        // 其余 padding 区域保持 0（符合官方）

        let input_layout = input.as_standard_layout();
        let outputs = self.encoder.run(vec![Value::from_array(self.encoder.allocator(), &input_layout)?])?;
        let output_tensor = outputs.get(0).unwrap().try_extract::<f32>()?;
        // 保留你原来的转置方式，若形状不对可打印 shape 调整
        let embeddings = output_tensor.view().t().reversed_axes().into_owned();
        Ok((embeddings, new_w, new_h, scale))
    }

    pub fn is_encoded(&self) -> bool {
        self.image_embeddings.is_some()
    }

    // 可选：切换 decoder 使用的 orig_im_size 语义（遇到模型差异时用）
    pub fn set_use_new_size_for_decoder(&mut self, yes: bool) {
        self.use_new_size_for_decoder = yes;
    }
}
