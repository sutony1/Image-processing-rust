use std::sync::Arc;
use std::path::Path;
use ndarray::{Array, IxDyn};
use ort::{Environment, Session, SessionBuilder, Value};
use opencv::core;
use opencv::prelude::MatTraitConst;
use image::{ImageBuffer, RgbImage, imageops::FilterType};
use anyhow::Result;

pub struct SAMProcessor {
    encoder: Session,
    decoder: Session,
    image_embeddings: Option<Array<f32, IxDyn>>,
    orig_width: f32,
    orig_height: f32,
    resized_width: f32,
    resized_height: f32,
}

impl SAMProcessor {
    pub fn new() -> Result<Self> {
        let env = Arc::new(Environment::builder().with_name("SAM").build()?);

        // 解析模型路径：优先使用环境变量，其次exe目录，其次当前工作目录，最后工程根目录
        fn resolve_model_path(file: &str) -> std::path::PathBuf {
            if let Ok(dir) = std::env::var("SAM_MODEL_DIR") {
                let p = Path::new(&dir).join(file);
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
            // 工程根目录（编译时常量）
            Path::new(env!("CARGO_MANIFEST_DIR")).join(file)
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
            resized_width: 0.0,
            resized_height: 0.0,
        })
    }

    pub fn encode_image(&mut self, mat: &core::Mat) -> Result<()> {
        // 获取图像尺寸
        self.orig_width = mat.cols() as f32;
        self.orig_height = mat.rows() as f32;
        
        // 编码图像
        let (embeddings, resized_w, resized_h) = self.get_image_embeddings(mat)?;
        self.image_embeddings = Some(embeddings);
        self.resized_width = resized_w;
        self.resized_height = resized_h;
        
        Ok(())
    }

    pub fn decode_prompt(&self, x1: f32, y1: f32, x2: f32, y2: f32) -> Result<(Vec<u8>, usize, usize)> {
        let embeddings = self.image_embeddings.as_ref()
            .ok_or_else(|| anyhow::anyhow!("图像未编码，请先调用encode_image"))?;

        // 准备解码器输入
        let point_coords = Array::from_shape_vec((1, 2, 2), vec![
            x1 * (self.resized_width / self.orig_width),
            y1 * (self.resized_height / self.orig_height),
            x2 * (self.resized_width / self.orig_width),
            y2 * (self.resized_height / self.orig_height),
        ])?.into_dyn();

        let point_labels = Array::from_shape_vec((1, 2), vec![2.0_f32, 3.0_f32])?.into_dyn();
        let mask_input = Array::from_shape_vec((1, 1, 256, 256), 
            vec![0.0_f32; 256 * 256])?.into_dyn();
        let has_mask_input = Array::from_vec(vec![0.0_f32]).into_dyn();
        let orig_im_size = Array::from_vec(vec![self.orig_height, self.orig_width]).into_dyn();

        // 修复临时值生命周期问题 - 创建绑定变量
        let embeddings_layout = embeddings.as_standard_layout();
        let point_coords_layout = point_coords.as_standard_layout();
        let point_labels_layout = point_labels.as_standard_layout();
        let mask_input_layout = mask_input.as_standard_layout();
        let has_mask_input_layout = has_mask_input.as_standard_layout();
        let orig_im_size_layout = orig_im_size.as_standard_layout();

        // 运行解码器
        let decoder_inputs = vec![
            Value::from_array(self.decoder.allocator(), &embeddings_layout)?,
            Value::from_array(self.decoder.allocator(), &point_coords_layout)?,
            Value::from_array(self.decoder.allocator(), &point_labels_layout)?,
            Value::from_array(self.decoder.allocator(), &mask_input_layout)?,
            Value::from_array(self.decoder.allocator(), &has_mask_input_layout)?,
            Value::from_array(self.decoder.allocator(), &orig_im_size_layout)?,
        ];

        let outputs = self.decoder.run(decoder_inputs)?;
        
        // 修复临时值生命周期问题 - 创建绑定变量
        let output_tensor = outputs.get(0).unwrap().try_extract::<f32>()?;
        let output_view = output_tensor.view();
        let output = output_view.t().reversed_axes().into_dyn();
        let shape = output.shape().to_vec();
        let (mask_h, mask_w) = if shape.len() >= 2 {
            (shape[shape.len() - 2] as usize, shape[shape.len() - 1] as usize)
        } else {
            (256usize, 256usize)
        };

        // 处理输出掩码
        let mask = output.map(|item| if *item > 0.0 { 1_u8 } else { 0_u8 }).into_raw_vec();
        Ok((mask, mask_w, mask_h))
    }

    fn mat_to_rgb(&self, mat: &core::Mat) -> Result<Vec<u8>> {
        // 将OpenCV BGR Mat转换为RGB字节数组
        let mut rgb_data = Vec::new();
        let rows = mat.rows();
        let cols = mat.cols();
        
        for y in 0..rows {
            for x in 0..cols {
                let pixel = mat.at_2d::<opencv::core::Vec3b>(y, x)?;
                // BGR -> RGB
                rgb_data.push(pixel[2]); // R
                rgb_data.push(pixel[1]); // G  
                rgb_data.push(pixel[0]); // B
            }
        }
        
        Ok(rgb_data)
    }

    fn get_image_embeddings(&self, mat: &core::Mat) -> Result<(Array<f32, IxDyn>, f32, f32)> {
        // 将Mat转换为RGB图像，并调整大小到1024x1024
        let cols = mat.cols() as u32;
        let rows = mat.rows() as u32;
        let rgb_data = self.mat_to_rgb(mat)?;
        let rgb_image: RgbImage = ImageBuffer::from_raw(cols, rows, rgb_data)
            .ok_or_else(|| anyhow::anyhow!("RGB缓冲区尺寸与图像尺寸不匹配"))?;
        let img_resized: RgbImage = image::imageops::resize(&rgb_image, 1024, 1024, FilterType::CatmullRom);

        // 准备输入张量
        let mut input = Array::zeros((1, 3, 1024, 1024)).into_dyn();
        let mean = vec![123.675, 116.28, 103.53];
        let std = vec![58.395, 57.12, 57.375];
        
        let width = img_resized.width();
        let height = img_resized.height();
        for y in 0..height {
            for x in 0..width {
                let p = img_resized.get_pixel(x, y);
                let r = p.0[0];
                let g = p.0[1];
                let b = p.0[2];
                let xi = x as usize;
                let yi = y as usize;
                input[[0, 0, yi, xi]] = (r as f32 - mean[0]) / std[0];
                input[[0, 1, yi, xi]] = (g as f32 - mean[1]) / std[1];
                input[[0, 2, yi, xi]] = (b as f32 - mean[2]) / std[2];
            }
        }

        // 运行编码器
        let input_layout = input.as_standard_layout();
        let input_values = vec![Value::from_array(self.encoder.allocator(), &input_layout)?];
        let outputs = self.encoder.run(input_values)?;
        
        // 修复临时值生命周期问题 - 创建绑定变量
        let output_tensor = outputs.get(0).unwrap().try_extract::<f32>()?;
        let embeddings = output_tensor.view().t().reversed_axes().into_owned();

        Ok((embeddings, 1024.0, 1024.0))
    }

    pub fn is_encoded(&self) -> bool {
        self.image_embeddings.is_some()
    }
}