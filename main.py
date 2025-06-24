#!/usr/bin/env python3
"""
WhisperX API 启动入口

这是一个简单的启动文件，用于运行 WhisperX API 服务。
实际的应用代码位于 src/whisperx_api/ 目录中。
"""

if __name__ == "__main__":
    import uvicorn
    from src.whisperx_api.config import HOST, PORT
    
    print("🎤 启动 WhisperX API 服务...")
    print(f"📍 服务地址: http://{HOST}:{PORT}")
    print(f"📖 API 文档: http://{HOST}:{PORT}/docs")
    print(f"🔍 健康检查: http://{HOST}:{PORT}/health")
    
    uvicorn.run(
        "src.whisperx_api.app:app",
        host=HOST,
        port=PORT,
        reload=True,
        reload_dirs=["src"]
    ) 
