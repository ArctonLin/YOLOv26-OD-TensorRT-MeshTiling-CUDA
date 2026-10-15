@echo off
REM Windows equivalent of run.sh (Visual Studio .sln build instead of CMake/make)
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

REM ---------------------------------------------------------------------------
REM Third-party locations (override by setting these env vars before calling)
REM ---------------------------------------------------------------------------
if not defined OPENCV_ROOT   set "OPENCV_ROOT=D:\Compile\OpenCV-4.11.0\build\install"
if not defined TENSORRT_ROOT set "TENSORRT_ROOT=D:\Compile\TensorRT-10.12.0.36.Windows.win10.cuda-12.9\TensorRT-10.12.0.36"
if not defined CUDA_PATH     set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
if not defined BUILD_CONFIG  set "BUILD_CONFIG=Release"

REM Runtime DLL search path (TensorRT / OpenCV / CUDA) + trtexec
set "PATH=%TENSORRT_ROOT%\bin;%TENSORRT_ROOT%\lib;%OPENCV_ROOT%\x64\vc17\bin;%CUDA_PATH%\bin;%PATH%"

REM Python: prefer environment variable, active conda env, local venv, or known conda env
if not defined PYTHON (
    if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" set "PYTHON=%CONDA_PREFIX%\python.exe"
    if not defined PYTHON if exist "venv\Scripts\python.exe" set "PYTHON=venv\Scripts\python.exe"
    if not defined PYTHON if exist "C:\Users\Card\anaconda3\envs\yolo11\python.exe" set "PYTHON=C:\Users\Card\anaconda3\envs\yolo11\python.exe"
    if not defined PYTHON set "PYTHON=python"
)

REM ---------------------------------------------------------------------------
REM Export model using config.json if engine does not exist
REM ---------------------------------------------------------------------------
echo Checking TRT model status...
if exist "yolo26n.engine" goto :engine_ready

echo Exporting model to ONNX using settings from config.json...
"%PYTHON%" scripts\export_model.py --config config.json
if errorlevel 1 goto :fail

echo Building TensorRT engine using trtexec to match C++ API version perfectly...
trtexec --onnx=yolo26n.onnx --saveEngine=yolo26n.engine --fp16
if errorlevel 1 goto :fail
goto :build

:engine_ready
echo TRT engine already exists, skipping export.

REM ---------------------------------------------------------------------------
REM Build C++ project (MSBuild + .sln)
REM ---------------------------------------------------------------------------
:build
echo Building C++ Project...
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "MSBUILD="
if not exist "%VSWHERE%" goto :msbuild_from_path
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -requires Microsoft.Component.MSBuild -find MSBuild\**\Bin\MSBuild.exe`) do set "MSBUILD=%%i"
:msbuild_from_path
if not defined MSBUILD where msbuild >nul 2>nul && set "MSBUILD=msbuild"
if not defined MSBUILD (
    echo Error: MSBuild not found. Please install Visual Studio 2022 with the C++ workload.
    goto :fail
)

"%MSBUILD%" yolo-tensorrt-meshtiling-cudadecode.sln /m /nologo /v:minimal /p:Configuration=%BUILD_CONFIG% /p:Platform=x64
if errorlevel 1 goto :fail

REM ---------------------------------------------------------------------------
REM Run Video Processing
REM ---------------------------------------------------------------------------
set "INPUT_VIDEO=4k-highway-traffic.mp4"
set "OUTPUT_VIDEO=4k-highway-traffic-detect.mp4"

if not exist "%INPUT_VIDEO%" (
    echo Error: Input video %INPUT_VIDEO% not found!
    goto :fail
)

echo Running YOLO Detector on Video...
"bin\x64\%BUILD_CONFIG%\yolo_detector.exe" config.json "%INPUT_VIDEO%" "%OUTPUT_VIDEO%"
if errorlevel 1 goto :fail

REM ---------------------------------------------------------------------------
REM Compress Output Video with ffmpeg libx264 to 3Mbps
REM ---------------------------------------------------------------------------
echo Compressing %OUTPUT_VIDEO% to 3Mbps using ffmpeg libx264...
ffmpeg -y -i "%OUTPUT_VIDEO%" -c:v libx264 -b:v 3M -maxrate 3.5M -bufsize 6M -pix_fmt yuv420p -preset fast "%OUTPUT_VIDEO%.tmp.mp4"
if not errorlevel 1 (
    move /y "%OUTPUT_VIDEO%.tmp.mp4" "%OUTPUT_VIDEO%"
)

endlocal
exit /b 0

:fail
echo.
echo *** run.bat failed (errorlevel %errorlevel%) ***
endlocal
exit /b 1
