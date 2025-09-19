@echo off
echo 正在部署 img_view_cv 便携版...
echo.

REM 创建部署目录
set DEPLOY_DIR=img_view_cv_portable
if exist "%DEPLOY_DIR%" (
    echo 删除现有部署目录...
    rmdir /s /q "%DEPLOY_DIR%"
)

echo 创建部署目录: %DEPLOY_DIR%
mkdir "%DEPLOY_DIR%"

echo 复制可执行文件...
copy "target\debug\img_view_cv.exe" "%DEPLOY_DIR%\"

echo 复制ONNX模型文件...
copy "target\debug\vit_t_encoder.onnx" "%DEPLOY_DIR%\"
copy "target\debug\vit_t_decoder.onnx" "%DEPLOY_DIR%\"

echo 复制配置文件...
if exist "config.yaml" copy "config.yaml" "%DEPLOY_DIR%\"

echo 复制README文件...
if exist "README.md" copy "README.md" "%DEPLOY_DIR%\"

echo.
echo 部署完成！便携版程序位于: %DEPLOY_DIR%\
echo 您可以将整个 %DEPLOY_DIR% 文件夹复制到任何Windows电脑上运行。
echo.
pause
