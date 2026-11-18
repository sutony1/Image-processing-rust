//! Local JSON-line TCP control port so --cli can drive the GUI.

use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc;
use std::time::Duration;
use winapi::shared::minwindef::{LPARAM, WPARAM};
use winapi::um::winuser::{PostMessageW, WM_APP};

pub const DEFAULT_PORT: u16 = 17332;
pub const WM_IPC_COMMAND: u32 = WM_APP + 42;

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct IpcRequest {
    #[serde(default)]
    pub id: u64,
    pub cmd: String,
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub x1: Option<f32>,
    #[serde(default)]
    pub y1: Option<f32>,
    #[serde(default)]
    pub x2: Option<f32>,
    #[serde(default)]
    pub y2: Option<f32>,
    #[serde(default)]
    pub dir: Option<String>,
    #[serde(default)]
    pub angle: Option<f64>,
    #[serde(default)]
    pub points: Option<Vec<[f32; 2]>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct IpcResponse {
    pub id: u64,
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

pub struct IpcJob {
    pub line: String,
    pub tx: mpsc::Sender<String>,
}

pub fn port() -> u16 {
    std::env::var("IMG_VIEW_CV_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_PORT)
}

pub fn start_server(hwnd: isize) {
    std::thread::spawn(move || {
        let addr = format!("127.0.0.1:{}", port());
        let listener = match TcpListener::bind(&addr) {
            Ok(l) => l,
            Err(_) => return,
        };
        let _ = listener.set_nonblocking(false);
        for stream in listener.incoming() {
            match stream {
                Ok(s) => {
                    let _ = s.set_nodelay(true);
                    std::thread::spawn(move || handle_client(hwnd, s));
                }
                Err(_) => break,
            }
        }
    });
}

fn handle_client(hwnd: isize, stream: TcpStream) {
    let mut writer = match stream.try_clone() {
        Ok(w) => w,
        Err(_) => return,
    };
    let mut reader = BufReader::new(stream);
    loop {
        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {}
            Err(_) => break,
        }
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }
        let (tx, rx) = mpsc::channel();
        let job = Box::new(IpcJob { line, tx });
        let posted = unsafe {
            PostMessageW(
                hwnd as _,
                WM_IPC_COMMAND,
                0 as WPARAM,
                Box::into_raw(job) as LPARAM,
            )
        };
        let reply = if posted == 0 {
            "{\"id\":0,\"ok\":false,\"error\":\"PostMessage failed\"}".to_string()
        } else {
            match rx.recv_timeout(Duration::from_secs(180)) {
                Ok(s) => s,
                Err(_) => "{\"id\":0,\"ok\":false,\"error\":\"timeout waiting for UI thread\"}".to_string(),
            }
        };
        if writeln!(writer, "{}", reply).is_err() {
            break;
        }
        let _ = writer.flush();
    }
}

pub fn send_request(req: &IpcRequest) -> Result<IpcResponse, String> {
    let addr = format!("127.0.0.1:{}", port());
    let mut stream = TcpStream::connect_timeout(
        &addr.parse().map_err(|e| format!("{}", e))?,
        Duration::from_secs(2),
    )
    .map_err(|e| format!("无法连接图像查看器 ({}): {}。请先启动程序。", addr, e))?;
    let _ = stream.set_read_timeout(Some(Duration::from_secs(180)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(10)));
    let body = serde_json::to_string(req).map_err(|e| e.to_string())?;
    writeln!(stream, "{}", body).map_err(|e| e.to_string())?;
    stream.flush().map_err(|e| e.to_string())?;
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader.read_line(&mut line).map_err(|e| e.to_string())?;
    serde_json::from_str(line.trim()).map_err(|e| format!("坏的 IPC 响应: {} ({})", e, line))
}

pub fn ensure_gui_running() -> Result<(), String> {
    if send_request(&IpcRequest {
        id: 0,
        cmd: "ping".into(),
        path: None,
        x1: None,
        y1: None,
        x2: None,
        y2: None,
        dir: None,
        angle: None,
        points: None,
    })
    .is_ok()
    {
        return Ok(());
    }
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    use std::os::windows::process::CommandExt;
    const DETACHED_PROCESS: u32 = 0x00000008;
    const CREATE_NEW_PROCESS_GROUP: u32 = 0x00000200;
    std::process::Command::new(&exe)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .creation_flags(DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP)
        .spawn()
        .map_err(|e| format!("无法启动图像查看器: {}", e))?;
    for _ in 0..40 {
        std::thread::sleep(Duration::from_millis(250));
        if send_request(&IpcRequest {
            id: 0,
            cmd: "ping".into(),
            path: None,
            x1: None,
            y1: None,
            x2: None,
            y2: None,
            dir: None,
            angle: None,
            points: None,
        })
        .is_ok()
        {
            return Ok(());
        }
    }
    Err("已启动图像查看器，但控制端口未就绪（127.0.0.1:17332）".into())
}
