@echo off
REM Test Runner Batch Script for Windows
REM Usage: test.bat [quick|full|help]

echo 🚀 Weather Data Imputation - Test Runner
echo ========================================

if "%1"=="quick" (
    echo ⚡ Running Quick Tests...
    python quick_test.py
    goto :end
)

if "%1"=="full" (
    echo 📊 Running Full Test Suite...
    python run_all_tests.py
    goto :end
)

if "%1"=="help" (
    echo 📋 Available Commands:
    echo   test.bat quick    - Run quick tests (recommended for development)
    echo   test.bat full     - Run full test suite with detailed report
    echo   test.bat help     - Show this help message
    echo.
    echo 📝 Examples:
    echo   test.bat quick
    echo   test.bat full
    goto :end
)

REM Default: run quick tests
echo ⚡ Running Quick Tests (default)...
echo Use 'test.bat help' for more options
echo.
python quick_test.py

:end
echo.
echo ✅ Test execution completed!
pause 