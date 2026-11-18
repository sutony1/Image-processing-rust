//! Capture the viewer client area and the full window (title bar + menu).

use anyhow::{Context, Result};
use image::{ImageBuffer, Rgba};
use std::path::{Path, PathBuf};
use std::ptr::null_mut;
use std::time::{SystemTime, UNIX_EPOCH};
use winapi::shared::minwindef::*;
use winapi::shared::windef::*;
use winapi::um::wingdi::*;
use winapi::um::winuser::*;

const PW_RENDERFULLCONTENT: UINT = 2;

pub unsafe fn capture_viewer(hwnd: HWND, out_dir: Option<&str>) -> Result<(PathBuf, PathBuf)> {
    if IsIconic(hwnd) != 0 {
        ShowWindow(hwnd, SW_RESTORE);
    }
    SetForegroundWindow(hwnd);
    BringWindowToTop(hwnd);
    InvalidateRect(hwnd, null_mut(), 0);
    UpdateWindow(hwnd);

    let dir = resolve_out_dir(out_dir)?;
    std::fs::create_dir_all(&dir).with_context(|| format!("无法创建目录 {}", dir.display()))?;
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let client_path = dir.join(format!("screenshot_client_{}.png", stamp));
    let window_path = dir.join(format!("screenshot_window_{}.png", stamp));

    let (cw, ch, client_bgra) = capture_client(hwnd)?;
    save_bgra_png(&client_path, cw, ch, &client_bgra)?;

    let (ww, wh, window_bgra) = capture_full_window(hwnd)?;
    save_bgra_png(&window_path, ww, wh, &window_bgra)?;

    Ok((client_path, window_path))
}

fn resolve_out_dir(path: Option<&str>) -> Result<PathBuf> {
    if let Some(p) = path {
        let pb = PathBuf::from(p);
        if pb.extension().is_some() {
            if let Some(parent) = pb.parent() {
                if !parent.as_os_str().is_empty() {
                    return Ok(parent.to_path_buf());
                }
            }
            return Ok(PathBuf::from("."));
        }
        return Ok(pb);
    }
    Ok(PathBuf::from("debug_out"))
}

unsafe fn capture_client(hwnd: HWND) -> Result<(i32, i32, Vec<u8>)> {
    let mut rc: RECT = std::mem::zeroed();
    if GetClientRect(hwnd, &mut rc) == 0 {
        anyhow::bail!("GetClientRect 失败");
    }
    let w = rc.right - rc.left;
    let h = rc.bottom - rc.top;
    if w <= 0 || h <= 0 {
        anyhow::bail!("客户区尺寸无效: {}x{}", w, h);
    }
    let hdc = GetDC(hwnd);
    if hdc.is_null() {
        anyhow::bail!("GetDC 失败");
    }
    let result = blit_hdc(hdc, 0, 0, w, h);
    ReleaseDC(hwnd, hdc);
    result
}

unsafe fn capture_full_window(hwnd: HWND) -> Result<(i32, i32, Vec<u8>)> {
    let mut wr: RECT = std::mem::zeroed();
    if GetWindowRect(hwnd, &mut wr) == 0 {
        anyhow::bail!("GetWindowRect 失败");
    }
    let w = wr.right - wr.left;
    let h = wr.bottom - wr.top;
    if w <= 0 || h <= 0 {
        anyhow::bail!("窗口尺寸无效: {}x{}", w, h);
    }

    let hdc_win = GetWindowDC(hwnd);
    if hdc_win.is_null() {
        anyhow::bail!("GetWindowDC 失败");
    }
    let mem_dc = CreateCompatibleDC(hdc_win);
    let bmp = CreateCompatibleBitmap(hdc_win, w, h);
    let old = SelectObject(mem_dc, bmp as *mut _);

    let printed = PrintWindow(hwnd, mem_dc, PW_RENDERFULLCONTENT);
    if printed == 0 {
        let screen = GetDC(null_mut());
        BitBlt(mem_dc, 0, 0, w, h, screen, wr.left, wr.top, SRCCOPY);
        ReleaseDC(null_mut(), screen);
    }

    let pixels = dibits(mem_dc, bmp, w, h)?;
    SelectObject(mem_dc, old);
    DeleteObject(bmp as *mut _);
    DeleteDC(mem_dc);
    ReleaseDC(hwnd, hdc_win);
    Ok((w, h, pixels))
}

unsafe fn blit_hdc(hdc: HDC, src_x: i32, src_y: i32, w: i32, h: i32) -> Result<(i32, i32, Vec<u8>)> {
    let mem_dc = CreateCompatibleDC(hdc);
    let bmp = CreateCompatibleBitmap(hdc, w, h);
    let old = SelectObject(mem_dc, bmp as *mut _);
    BitBlt(mem_dc, 0, 0, w, h, hdc, src_x, src_y, SRCCOPY);
    let pixels = dibits(mem_dc, bmp, w, h)?;
    SelectObject(mem_dc, old);
    DeleteObject(bmp as *mut _);
    DeleteDC(mem_dc);
    Ok((w, h, pixels))
}

unsafe fn dibits(hdc: HDC, bmp: HBITMAP, w: i32, h: i32) -> Result<Vec<u8>> {
    let mut info: BITMAPINFO = std::mem::zeroed();
    info.bmiHeader.biSize = std::mem::size_of::<BITMAPINFOHEADER>() as DWORD;
    info.bmiHeader.biWidth = w;
    info.bmiHeader.biHeight = -h;
    info.bmiHeader.biPlanes = 1;
    info.bmiHeader.biBitCount = 32;
    info.bmiHeader.biCompression = BI_RGB;
    let mut pixels = vec![0u8; (w as usize) * (h as usize) * 4];
    let got = GetDIBits(
        hdc,
        bmp,
        0,
        h as UINT,
        pixels.as_mut_ptr() as *mut _,
        &mut info,
        DIB_RGB_COLORS,
    );
    if got == 0 {
        anyhow::bail!("GetDIBits 失败");
    }
    Ok(pixels)
}

fn save_bgra_png(path: &Path, w: i32, h: i32, bgra: &[u8]) -> Result<()> {
    let mut rgba = bgra.to_vec();
    for px in rgba.chunks_exact_mut(4) {
        px.swap(0, 2);
        px[3] = 255;
    }
    let img = ImageBuffer::<Rgba<u8>, _>::from_raw(w as u32, h as u32, rgba)
        .ok_or_else(|| anyhow::anyhow!("无法组装 PNG 像素"))?;
    img.save(path).with_context(|| format!("保存失败 {}", path.display()))?;
    Ok(())
}
