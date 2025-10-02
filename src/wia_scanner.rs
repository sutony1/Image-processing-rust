use std::path::PathBuf;
use anyhow::{Result, Context};
use windows::{
    core::*,
    Win32::System::Com::*,
    Win32::Foundation::*,
};

// WIA常量定义
const WIA_DEVICE_DIALOG_SINGLE_IMAGE: u32 = 0x00000002;
const WIA_INTENT_IMAGE_TYPE_COLOR: i32 = 0x00000001;
const WIA_INTENT_MINIMIZE_SIZE: i32 = 0x00010000;
const WIA_IPA_DATATYPE: &str = "WIA_IPA_DATATYPE";
const WIA_IPA_DEPTH: &str = "WIA_IPA_DEPTH";
const WIA_IPA_PIXELS_PER_LINE: &str = "WIA_IPA_PIXELS_PER_LINE";
const WIA_IPA_NUMBER_OF_LINES: &str = "WIA_IPA_NUMBER_OF_LINES";
const WIA_IPA_HORIZONTAL_RESOLUTION: &str = "WIA_IPA_HORIZONTAL_RESOLUTION";
const WIA_IPA_VERTICAL_RESOLUTION: &str = "WIA_IPA_VERTICAL_RESOLUTION";
const WIA_IPA_FORMAT: &str = "WIA_IPA_FORMAT";

// WIA COM接口的GUID定义
// WIA 1.0 IWiaDevMgr的CLSID
const CLSID_WIADEVMGR: GUID = GUID::from_u128(0xa1f4e726_8cf1_11d1_bf92_0060081ed811);
// WIA 2.0 IWiaDevMgr2的CLSID (Vista+)
const CLSID_WIADEVMGR2: GUID = GUID::from_u128(0xb6c292bc_7c88_41ee_8b54_8ec92617e599);
// IWiaDevMgr的IID
const IID_IWIADEVMGR: GUID = GUID::from_u128(0x5eb2502a_8cf1_11d1_bf92_0060081ed811);
// WIA CommonDialog的CLSID (推荐使用，更简单稳定)
const CLSID_WIA_COMMON_DIALOG: GUID = GUID::from_u128(0x850d1d11_6bbe_4e99_811f_84556e55cf19);

// WIA接口的COM声明（简化版）
#[repr(C)]
pub struct IWiaDevMgr {
    vtable: *const IWiaDevMgrVtbl,
}

#[repr(C)]
struct IWiaDevMgrVtbl {
    // IUnknown方法
    query_interface: unsafe extern "system" fn(*const IWiaDevMgr, *const GUID, *mut *mut std::ffi::c_void) -> HRESULT,
    add_ref: unsafe extern "system" fn(*const IWiaDevMgr) -> u32,
    release: unsafe extern "system" fn(*const IWiaDevMgr) -> u32,
    // IWiaDevMgr方法
    enum_device_info: unsafe extern "system" fn(*const IWiaDevMgr, i32, *mut *mut std::ffi::c_void) -> HRESULT,
    create_device: unsafe extern "system" fn(*const IWiaDevMgr, BSTR, *mut *mut std::ffi::c_void) -> HRESULT,
    select_device_dlg: unsafe extern "system" fn(*const IWiaDevMgr, HWND, i32, i32, *mut BSTR, *mut *mut std::ffi::c_void) -> HRESULT,
    select_device_dlg_id: unsafe extern "system" fn(*const IWiaDevMgr, HWND, i32, i32, *mut BSTR, *mut *mut std::ffi::c_void) -> HRESULT,
    get_image_dlg: unsafe extern "system" fn(*const IWiaDevMgr, HWND, i32, i32, i32, *mut std::ffi::c_void, BSTR) -> HRESULT,
    register_event_callback_interface: unsafe extern "system" fn(*const IWiaDevMgr, i32, *const GUID, *const GUID, *mut std::ffi::c_void, *mut *mut std::ffi::c_void) -> HRESULT,
    register_event_callback_program: unsafe extern "system" fn(*const IWiaDevMgr, i32, *const GUID, *const GUID, BSTR, BSTR, BSTR, *mut *mut std::ffi::c_void) -> HRESULT,
    register_event_callback_clsid: unsafe extern "system" fn(*const IWiaDevMgr, i32, *const GUID, *const GUID, *const GUID, BSTR, BSTR, *mut *mut std::ffi::c_void) -> HRESULT,
}

/// 初始化COM环境
pub fn initialize_com() -> Result<()> {
    unsafe {
        CoInitializeEx(None, COINIT_APARTMENTTHREADED)
            .ok()
            .context("COM初始化失败")?;
    }
    Ok(())
}

/// 清理COM环境
pub fn uninitialize_com() {
    unsafe {
        CoUninitialize();
    }
}

/// 使用WIA API获取扫描图像
/// 
/// 参数：
/// - hwnd: 父窗口句柄
/// - output_path: 输出文件路径（可选，如果为None则保存到临时目录）
/// 
/// 返回：扫描到的图像文件路径
pub fn acquire_image_from_scanner(hwnd: HWND, output_path: Option<&str>) -> Result<String> {
    println!("🖨️ 开始初始化WIA扫描功能...");
    
    // 初始化COM
    initialize_com().context("初始化COM失败")?;
    
    // 确定输出路径
    let output_file = if let Some(path) = output_path {
        PathBuf::from(path)
    } else {
        // 获取用户的图片文件夹路径
        let pictures_folder = get_pictures_folder()
            .context("获取图片文件夹路径失败")?;
        
        // 生成带时间戳的文件名
        let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
        pictures_folder.join(format!("WIA_Scan_{}.png", timestamp))
    };
    
    println!("📁 输出路径: {}", output_file.display());
    
    // 尝试调用WIA对话框
    let result = unsafe {
        acquire_image_with_dialog(hwnd, &output_file)
    };
    
    // 清理COM
    uninitialize_com();
    
    result
}

/// 获取Windows图片文件夹路径
fn get_pictures_folder() -> Result<PathBuf> {
    use std::env;
    
    // 获取用户配置文件路径
    let user_profile = env::var("USERPROFILE")
        .context("无法获取USERPROFILE环境变量")?;
    
    let pictures_path = PathBuf::from(user_profile).join("Pictures");
    
    // 确保图片文件夹存在
    if !pictures_path.exists() {
        std::fs::create_dir_all(&pictures_path)
            .context("创建图片文件夹失败")?;
    }
    
    Ok(pictures_path)
}

/// 使用WIA对话框获取图像（使用简化的PowerShell方法）
unsafe fn acquire_image_with_dialog(_hwnd: HWND, output_path: &PathBuf) -> Result<String> {
    println!("🖼️ 使用 PowerShell 脚本调用 WIA 扫描...");
    
    // 生成临时 PowerShell 脚本，设置为 300 DPI（避免中文以防编码问题）
    let output_path_str = output_path.to_string_lossy().to_string();
    let ps_script = format!(
        r#"
$deviceManager = New-Object -ComObject WIA.CommonDialog
$device = $deviceManager.ShowSelectDevice()
if ($device) {{
    $item = $device.Items[1]
    
    # Set scan resolution to 300 DPI
    try {{
        $item.Properties["6147"].Value = 300
        $item.Properties["6148"].Value = 300
        Write-Host "[OK] Resolution set to 300 DPI"
    }} catch {{
        Write-Host "[WARN] Cannot set resolution"
    }}
    
    # Set color mode to RGB (value 3 = color, not 2)
    try {{
        $item.Properties["6146"].Value = 3
        Write-Host "[OK] Color mode set to RGB"
    }} catch {{
        Write-Host "[WARN] Cannot set color mode"
    }}
    
    $image = $deviceManager.ShowTransfer($item)
    if ($image) {{
        $image.SaveFile("{}")
        Write-Host "[SUCCESS] Scan completed"
        exit 0
    }}
}}
Write-Host "[FAIL] Scan failed or cancelled"
exit 1
"#,
        output_path_str.replace("\\", "\\\\")
    );
    
    // 写入临时脚本文件
    let temp_script = std::env::temp_dir().join("wia_scan.ps1");
    std::fs::write(&temp_script, ps_script)
        .context("无法创建临时 PowerShell 脚本")?;
    
    // 执行 PowerShell 脚本
    let output = std::process::Command::new("powershell.exe")
        .args(&[
            "-ExecutionPolicy", "Bypass",
            "-NoProfile",
            "-File", temp_script.to_str().unwrap()
        ])
        .output()
        .context("无法执行 PowerShell 脚本")?;
    
    // 删除临时脚本
    let _ = std::fs::remove_file(&temp_script);
    
    // 总是输出 PowerShell 的结果以便调试
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    println!("PowerShell 输出:\n{}", stdout);
    if !stderr.is_empty() {
        println!("PowerShell 错误:\n{}", stderr);
    }
    
    // 检查执行结果
    if !output.status.success() {
        return Err(anyhow::anyhow!("WIA 扫描失败或被用户取消\n\n详细信息:\n{}\n{}", stdout, stderr));
    }
    
    println!("✅ 扫描完成，图像已保存到: {}", output_path.display());
    
    // 验证文件是否存在
    if !output_path.exists() {
        return Err(anyhow::anyhow!("扫描完成但文件未生成: {}", output_path.display()));
    }
    
    Ok(output_path.to_string_lossy().to_string())
}

/// 使用WIA COM对话框获取图像（备用方法，保留以供参考）
#[allow(dead_code)]
unsafe fn acquire_image_with_dialog_com(hwnd: HWND, output_path: &PathBuf) -> Result<String> {
    // 先尝试 WIA 2.0 (Windows Vista+)
    let mut dev_mgr_result: windows::core::Result<IUnknown> = CoCreateInstance(
        &CLSID_WIADEVMGR2,
        None,
        CLSCTX_LOCAL_SERVER
    );
    
    // 如果 WIA 2.0 失败，尝试 WIA 1.0
    if dev_mgr_result.is_err() {
        println!("⚠️ WIA 2.0 不可用，尝试 WIA 1.0...");
        dev_mgr_result = CoCreateInstance(
            &CLSID_WIADEVMGR,
            None,
            CLSCTX_LOCAL_SERVER
        );
    }
    
    let dev_mgr_unknown = match dev_mgr_result {
        Ok(obj) => {
            println!("✅ WIA DevMgr对象创建成功");
            obj
        },
        Err(e) => {
            return Err(anyhow::anyhow!(
                "创建WIA DevMgr对象失败: {:?}\n\n可能的原因:\n\
                1. 系统未安装WIA服务或已禁用\n\
                2. 扫描仪驱动不支持WIA接口（仅支持TWAIN）\n\
                3. 需要以管理员权限运行\n\n\
                建议: Epson扫描仪通常使用TWAIN接口，请考虑:\n\
                - 在Photoshop等软件中使用 文件->导入->TWAIN 导入\n\
                - 或使用Epson Scan软件直接扫描后导入图片", 
                e
            ))
        }
    };
    
    // Convert IUnknown to our custom IWiaDevMgr interface pointer
    let dev_mgr = dev_mgr_unknown.as_raw() as *const IWiaDevMgr;
    
    // 将输出路径转换为BSTR
    let output_path_str = output_path.to_string_lossy().to_string();
    let output_bstr = BSTR::from(output_path_str.as_str());
    
    // 直接调用GetImageDlg，让它自己处理设备选择
    println!("🖼️ 正在显示WIA扫描对话框...");
    
    let hr = ((*(*dev_mgr).vtable).get_image_dlg)(
        dev_mgr,
        hwnd,
        1, // 设备类型: 1 = 扫描仪
        WIA_INTENT_IMAGE_TYPE_COLOR | WIA_INTENT_MINIMIZE_SIZE, // 意图
        WIA_DEVICE_DIALOG_SINGLE_IMAGE as i32, // 标志: 单张图像
        std::ptr::null_mut(), // NULL让对话框自己选择设备
        output_bstr, // 输出文件路径
    );
    
    // 释放DevMgr对象
    ((*(*dev_mgr).vtable).release)(dev_mgr);
    
    if hr.is_err() {
        let error_code = hr.0;
        let error_msg = match error_code {
            -2145320860 => "用户取消了扫描操作",  // 0x80210064
            -2145320959 => "设备离线或未连接",      // 0x80210001
            -2147417842 => {                          // 0x8001010E
                "设备通信错误 (RPC_E_DISCONNECTED)\n\n\
                可能的原因:\n\
                1. 扫描仪未开机或未连接\n\
                2. 扫描仪驱动不完全支持 WIA 接口\n\
                3. 扫描仪仅支持 TWAIN 接口\n\n\
                建议解决方案:\n\
                • 确保扫描仪已开机并通过 USB 连接\n\
                • 在 Windows 设备管理器中检查扫描仪状态\n\
                • 使用 Epson Scan 软件直接扫描后导入图片\n\
                • 或在其他软件（如 Photoshop）中使用 TWAIN 接口扫描"
            },
            -2147024809 => "参数无效",                // 0x80070057
            _ => "未知的 WIA 错误"
        };
        return Err(anyhow::anyhow!("WIA GetImageDlg失败: HRESULT = HRESULT({:#010X})\n\n{}", error_code as u32, error_msg));
    }
    
    println!("✅ 扫描完成，图像已保存到: {}", output_path.display());
    
    // 验证文件是否存在
    if !output_path.exists() {
        return Err(anyhow::anyhow!("扫描完成但文件未生成: {}", output_path.display()));
    }
    
    Ok(output_path.to_string_lossy().to_string())
}

/// 测试WIA功能是否可用
pub fn test_wia_availability() -> bool {
    if initialize_com().is_err() {
        return false;
    }
    
    let available = unsafe {
        let dev_mgr_result: windows::core::Result<IUnknown> = CoCreateInstance(
            &CLSID_WIADEVMGR,
            None,
            CLSCTX_LOCAL_SERVER
        );
        
        if let Ok(dev_mgr_unknown) = dev_mgr_result {
            // Convert IUnknown to our custom IWiaDevMgr interface pointer
            let dev_mgr = dev_mgr_unknown.as_raw() as *const IWiaDevMgr;
            if !dev_mgr.is_null() {
                ((*(*dev_mgr).vtable).release)(dev_mgr);
                true
            } else {
                false
            }
        } else {
            false
        }
    };
    
    uninitialize_com();
    available
}

