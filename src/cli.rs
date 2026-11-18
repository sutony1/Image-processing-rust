//! One-shot CLI: `img_view_cv --cli <cmd> [args...]`

use crate::ipc::{self, IpcRequest};

pub fn run(args: &[String]) {
    if args.is_empty() || args[0] == "help" || args[0] == "-h" {
        print_help();
        return;
    }
    if let Err(e) = ipc::ensure_gui_running() {
        eprintln!("{}", e);
        std::process::exit(1);
    }
    let mut req = IpcRequest {
        id: 1,
        cmd: args[0].clone(),
        path: None,
        x1: None,
        y1: None,
        x2: None,
        y2: None,
        dir: None,
        angle: None,
        points: None,
    };
    match args[0].as_str() {
        "open" | "save" => {
            req.path = args.get(1).cloned();
            if req.path.is_none() {
                eprintln!("需要路径: --cli {} <path>", args[0]);
                std::process::exit(2);
            }
        }
        "screenshot" | "shot" => {
            req.cmd = "screenshot".into();
            req.path = args.get(1).cloned();
        }
        "sam-crop" | "sam_crop" => {
            req.cmd = "sam_crop".into();
            if args.len() < 5 {
                eprintln!("用法: --cli sam-crop <x1> <y1> <x2> <y2>");
                std::process::exit(2);
            }
            req.x1 = args[1].parse().ok();
            req.y1 = args[2].parse().ok();
            req.x2 = args[3].parse().ok();
            req.y2 = args[4].parse().ok();
        }
        "rotate" => {
            req.dir = args.get(1).cloned();
            req.angle = args.get(1).and_then(|s| s.parse().ok());
        }
        "extract-colors" => req.cmd = "extract_colors".into(),
        "auto-perspective" => req.cmd = "auto_perspective".into(),
        "manual-perspective" => {
            req.cmd = "manual_perspective".into();
            eprintln!("手动四点请用 JSON 控制口：cmd=manual_perspective, points=[[x,y]x4]");
            std::process::exit(2);
        }
        _ => {}
    }
    match ipc::send_request(&req) {
        Ok(resp) => {
            let pretty = serde_json::to_string_pretty(&resp).unwrap_or_else(|_| "{}".into());
            println!("{}", pretty);
            if !resp.ok {
                std::process::exit(1);
            }
        }
        Err(e) => {
            eprintln!("{}", e);
            std::process::exit(1);
        }
    }
}

fn print_help() {
    println!(
        "\
img_view_cv CLI — 控制已打开的图像查看器（没有则自动启动）

  img_view_cv --cli status
  img_view_cv --cli open <path>
  img_view_cv --cli save <path>
  img_view_cv --cli close
  img_view_cv --cli restore
  img_view_cv --cli gray
  img_view_cv --cli canny
  img_view_cv --cli threshold
  img_view_cv --cli auto-perspective
  img_view_cv --cli extract-colors
  img_view_cv --cli sam-crop <x1> <y1> <x2> <y2>
  img_view_cv --cli rotate 90cw|90ccw|180|<degrees>
  img_view_cv --cli next
  img_view_cv --cli prev
  img_view_cv --cli screenshot [输出目录]
      保存两张 PNG：客户区画面 + 整个活动窗口（标题栏和菜单）
"
    );
}
