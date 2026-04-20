@echo off
chcp 65001 >nul

echo Запуск в режиме сервера...
call run.bat --server-only --port 8080 %*