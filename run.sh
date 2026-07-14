# 停止后端
pkill -f uvicorn
# 启动后端
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000


# 停止前端
pkill -f next-server
# 启动前端
npm --prefix apps/web run dev -- --hostname 0.0.0.0 --port 3000
