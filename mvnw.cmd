@echo off
REM Maven Wrapper startup script for Windows

setlocal

set PRG=%~f0

REM Resolve links - %0 may be a symlink
:resolve
if not "%PRG%"=="" (
  for /f "delims=" %%i in ('dir /b /s "%PRG%" 2^>nul') do set "PRG=%%i"
  if not "%PRG:~-4%"==".lnk" goto resolved
  REM It's a shortcut, resolve it
  powershell -command "$sh = New-Object -ComObject WScript.Shell; $sc = $sh.CreateShortcut('%PRG%'); echo $sc.TargetPath" > "%TEMP%\mvnw_target.txt" 2>nul
  set /p PRG=<"%TEMP%\mvnw_target.txt"
  del "%TEMP%\mvnw_target.txt" 2>nul
  goto resolve
)
:resolved

REM Get Maven home
if "%MAVEN_HOME%"=="" if "%M2_HOME%"=="" (
  for %%i in ("%PRG%\..\..") do set "MAVEN_HOME=%%~fi"
)

REM Java command
if "%JAVA_HOME%"=="" (
  set JAVACMD=java
) else (
  set JAVACMD="%JAVA_HOME%\bin\java.exe"
)

REM Wrapper jar
set WRAPPER_JAR="%~dp0.mvn\wrapper\maven-wrapper.jar"

REM Check if wrapper jar exists
if not exist %WRAPPER_JAR% (
  echo Downloading Maven wrapper...
  set WRAPPER_URL=https://repo.maven.apache.org/maven2/org/apache/maven/wrapper/maven-wrapper/3.2.0/maven-wrapper-3.2.0.jar
  powershell -command "Invoke-WebRequest -Uri %WRAPPER_URL% -OutFile %WRAPPER_JAR%"
  if errorlevel 1 (
    echo ERROR: Failed to download Maven wrapper
    exit /b 1
  )
)

REM Run Maven wrapper
"%JAVACMD%" ^
  -Dmaven.multiModuleProjectDirectory="%~dp0.." ^
  -jar %WRAPPER_JAR% ^
  %*