import asyncio
import threading
import concurrent.futures
from .voxcpm_nanovllm import VoxCPMDubbingEngine  # 引入我们封装好的异步引擎

class SyncDubbingBridge:
    """
    工业级：同步项目与异步 VoxCPM 引擎的通信桥接器
    严格的单例模式，内部维护一个永不阻塞主线程的后台 Event Loop。
    """
    _instance = None
    _lock = threading.Lock()  # 线程锁，防止并发初始化

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if not cls._instance:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, **engine_config):
        # 防重复初始化保护
        if self._initialized:
            return

        self.engine = None
        self.engine_config = engine_config
        self.loop = asyncio.new_event_loop()
        self._startup_event = threading.Event()  # 启动屏障

        # 1. 启动守护线程 (Daemon=True 意味着主程序退出时，它会自动死亡)
        self.thread = threading.Thread(
            target=self._run_event_loop,
            daemon=True,
            name="VoxCPMDaemonThread"
        )
        self.thread.start()

        # 2. 通知后台线程开始加载 GPU 模型
        print("🚀 [桥接器] 正在启动后台异步引擎线程...")
        asyncio.run_coroutine_threadsafe(self._async_initialize(), self.loop)

        # 3. 阻塞当前（主）线程，直到模型预热完成
        print("⏳ [桥接器] 正在加载模型权重至显存，请稍候...")
        self._startup_event.wait()
        print("✅ [桥接器] 异步引擎底层就绪，同步接口已开放！")

        self._initialized = True

    def _run_event_loop(self):
        """后台守护线程的死循环"""
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_forever()
        finally:
            self.loop.close()
            print("🛑 [桥接器] 后台事件循环已销毁。")

    async def _async_initialize(self):
        """在后台执行的异步初始化"""
        try:
            self.engine = VoxCPMDubbingEngine(
                model_path=self.engine_config["model_path"],
                min_reference_ms=self.engine_config["min_reference_ms"],
                cfg_value=self.engine_config["cfg_value"],
                max_seqs=self.engine_config["max_seqs"],
            )
            await self.engine.startup()
        except Exception as e:
            print(f"❌ [桥接器] 引擎初始化惨烈失败: {e}")
        finally:
            # 无论成功失败，都必须释放屏障，否则主线程会死锁
            self._startup_event.set()

    # ==========================================
    # 暴露给主项目的纯同步 API
    # ==========================================

    def extract_latents(self, audio_path: str, timeout: int = 60):
        """
        同步调用：提取兜底特征
        """
        if not self.engine:
            raise RuntimeError("底层引擎未正确初始化！")

        future = asyncio.run_coroutine_threadsafe(
            self.engine.extract_latents(audio_path), self.loop
        )
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"提取特征超时 ({timeout}s)，可能是底层死锁。")

    def process_batch(self, task_list: list, fallback_latents_dict: dict, timeout: int = 600):
        """
        同步调用：处理批量视频配音任务
        """
        if not self.engine:
            raise RuntimeError("底层引擎未正确初始化！")

        future = asyncio.run_coroutine_threadsafe(
            self.engine.process_batch(task_list, fallback_latents_dict),
            self.loop
        )
        try:
            # 视频批量配音可能耗时较长，timeout 默认设为 10 分钟
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"批处理任务超时 ({timeout}s)，可能是并发量过大。")

    def shutdown(self):
        """
        同步调用：优雅关机
        在主项目收到 SIGTERM 或正常退出前调用，确保显存被干净释放。
        """
        print("🛑 [桥接器] 收到关机指令，正在清理后台线程...")
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5)
        print("🛑 [桥接器] 引擎已安全关闭。")
