@echo off
chcp 1251 >nul

echo Запуск в режиме сервера...
call run.bat --server-only --port 8190 %*