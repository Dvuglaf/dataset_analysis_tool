from __future__ import annotations

from enum import Enum, auto

"""
Простейший менеджер фоновых задач (JobManager) для GUI:

- JobInfo: описание задачи (тип, статус, прогресс, описание)
- JobManager: хранит все задачи и умеет запускать функции в отдельных потоках
- Worker: обёртка над функцией, которая живёт в QThread и шлёт сигналы прогресса

Использование:

    job = job_manager.run_job(
        job_type="export_scene",
        description="Export scene 1 of video_001",
        func=export_scene_fn,
        args=(...,),
        kwargs={...},
    )

    # Внутри export_scene_fn можно периодически вызывать progress_callback(p: float)
    def export_scene_fn(..., progress_callback=None):
        for i in range(total_steps):
            ...
            if progress_callback is not None:
                progress_callback(i / total_steps)
"""

from typing import Any, Callable, Dict, Optional

from PySide6.QtCore import QObject, Signal, Slot, QThread, Qt


class JobStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    DONE = auto()
    ERROR = auto()
    CANCELLED = auto()
    NONE = auto()


class JobInfo:
    """
    Базовое описание задачи. Используется как универсальный тип
    во всех сигналах и коллекциях.
    """

    def __init__(self, id: str, type: str, description: str):
        self.id = id
        self.type = type
        self.description = description
        self.status: JobStatus | None = None


class PendingJobInfo(JobInfo):
    def __init__(self, id: str, type: str, description: str):
        super().__init__(id, type, description)
        self.status = JobStatus.PENDING
    
    @classmethod
    def from_job(cls, job: JobInfo) -> "PendingJobInfo":
        return cls(
            id=job.id,
            type=job.type,
            description=job.description,
        )


class RunningJobInfo(JobInfo):
    def __init__(self, id: str, type: str, description: str, progress: float = 0.0):
        super().__init__(id, type, description)
        self.progress = progress
        self.status = JobStatus.RUNNING

    @classmethod
    def from_job(cls, job: JobInfo, progress: float = 0.0) -> "RunningJobInfo":
        return cls(
            id=job.id,
            type=job.type,
            description=job.description,
            progress=progress
        )


class DoneJobInfo(JobInfo):
    def __init__(self, id: str, type: str, description: str, result: dict):
        super().__init__(id, type, description)
        self.result = result
        self.status = JobStatus.DONE

    @classmethod
    def from_job(cls, job: JobInfo, result: dict) -> "DoneJobInfo":
        return cls(
            id=job.id,
            type=job.type,
            description=job.description,
            result=result
        )


class ErrorJobInfo(JobInfo):
    def __init__(self, id: str, type: str, description: str, message: str):
        super().__init__(id, type, description)
        self.message = message
        self.status = JobStatus.ERROR

    @classmethod
    def from_job(cls, job: JobInfo, message: str) -> "ErrorJobInfo":
        return cls(
            id=job.id,
            type=job.type,
            description=job.description,
            message=message
        )


class CancelledJobInfo(JobInfo):
    def __init__(self, id: str, type: str, description: str):
        super().__init__(id, type, description)
        self.status = JobStatus.CANCELLED
    
    @classmethod
    def from_job(cls, job: JobInfo) -> "CancelledJobInfo":
        return cls(
            id=job.id,
            type=job.type,
            description=job.description,
        )


class Worker(QObject):
    """
    Обёртка над произвольной функцией, выполняемой в отдельном QThread.
    """

    progress = Signal(float)  # 0..1
    finished = Signal(object)  # result
    failed = Signal(str)  # error message

    def __init__(self, func: Callable[..., Any], *args, **kwargs):
        super().__init__()
        self._func = func
        self._args = args
        self._kwargs = kwargs

    @Slot()
    def run(self):
        try:

            result = self._func(*self._args, **self._kwargs)
            self.finished.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class JobManager(QObject):
    """
    Глобальный менеджер фоновых задач.
    """

    job_added = Signal(JobInfo)
    job_updated = Signal(JobInfo)
    job_removed = Signal(str)
    jobs_changed = Signal(list)  # список JobInfo

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._jobs: Dict[str, JobInfo] = {}
        self._threads: Dict[str, QThread] = {}
        self._workers: Dict[str, Worker] = {}
        self._counter: int = 0

    # ------------------------
    # Публичный API
    # ------------------------
    def jobs(self) -> list[JobInfo]:
        return list(self._jobs.values())

    def run_job(
        self,
        job_type: str,
        description: str,
        func: Callable[..., Any],
        *args,
        on_done: Optional[Callable[[JobInfo, Any], None]] = None,
        **kwargs,
    ) -> JobInfo:
        """
        Создаёт задачу и запускает func в отдельном QThread.
        В func должен быть параметр progress_callback, если нужен прогресс.
        """
        job_id = self._next_id()
        base = JobInfo(id=job_id, type=job_type, description=description)
        job: JobInfo = RunningJobInfo.from_job(base)
        self._jobs[job_id] = job

        self.job_added.emit(job)
        self.jobs_changed.emit(self.jobs())

        worker = Worker(func, *args, **kwargs)
        self._workers[job_id] = worker
        thread = QThread(self)
        worker.moveToThread(thread)

        # связи сигналов
        thread.started.connect(worker.run)
        worker.progress.connect(lambda p, jid=job_id: self._on_progress(jid, p))

        # по завершении сначала обновляем статус, затем вызываем пользовательский callback (если есть)
        def _handle_finished(result: Any, jid: str = job_id, cb: Optional[Callable[[JobInfo, Any], None]] = on_done):
            done_job = self._on_finished(jid, result)
            if cb is not None and done_job is not None:
                cb(done_job, result)

        worker.finished.connect(_handle_finished)
        worker.failed.connect(lambda msg, jid=job_id: self._on_failed(jid, msg))

        # по окончании работы освобождаем ресурсы
        # quit() должен быть вызван в главном потоке, поэтому используем QueuedConnection
        def _on_worker_finished():
            thread.quit()
        
        def _on_worker_failed():
            thread.quit()
        
        # подключаем с Qt.QueuedConnection чтобы гарантировать выполнение в главном потоке
        worker.finished.connect(_on_worker_finished, Qt.QueuedConnection)
        worker.failed.connect(_on_worker_failed, Qt.QueuedConnection)
        
        # cleanup выполнится в главном потоке после завершения thread
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(lambda jid=job_id: self._cleanup_thread(jid))

        self._threads[job_id] = thread

        self.job_updated.emit(job)
        self.jobs_changed.emit(self.jobs())
        
        thread.start()

        return job

    def mark_cancelled(self, job_id: str):
        job = self._jobs.get(job_id)
        if not job:
            return
        if job.status in (JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELLED):
            return
        cancelled = CancelledJobInfo.from_job(job)
        self._jobs[job_id] = cancelled
        self.job_updated.emit(cancelled)
        self.jobs_changed.emit(self.jobs())

    # ------------------------
    # Внутренние слоты
    # ------------------------
    def _next_id(self) -> str:
        self._counter += 1
        return f"job_{self._counter:04d}"

    def _on_progress(self, job_id: str, value: float):
        job = self._jobs.get(job_id)
        if not job:
            return
        # обновляем progress только для RunningJobInfo
        if isinstance(job, RunningJobInfo):
            job.progress = max(0.0, min(1.0, float(value)))
            self.job_updated.emit(job)
            self.jobs_changed.emit(self.jobs())
        else:
            # если job не RunningJobInfo, создаём новый RunningJobInfo с обновлённым progress
            if isinstance(job, JobInfo):
                running = RunningJobInfo.from_job(job, max(0.0, min(1.0, float(value))))
                self._jobs[job_id] = running
                self.job_updated.emit(running)
                self.jobs_changed.emit(self.jobs())

    def _on_finished(self, job_id: str, result: Any) -> Optional[JobInfo]:
        job = self._jobs.get(job_id)
        if not job:
            return None

        # конвертируем result в dict если нужно
        if not isinstance(result, dict):
            result = {"result": result}

        done = DoneJobInfo.from_job(job, result)
        self._jobs[job_id] = done

        self.job_updated.emit(done)
        self.jobs_changed.emit(self.jobs())
        return done

    def _on_failed(self, job_id: str, message: str):
        job = self._jobs.get(job_id)
        if not job:
            return
        self._workers.pop(job_id, None)
        err = ErrorJobInfo.from_job(job, message)
        self._jobs[job_id] = err
        self.job_updated.emit(err)
        self.jobs_changed.emit(self.jobs())

    def _cleanup_thread(self, job_id: str):
        thread = self._threads.pop(job_id, None)
        if thread is not None:
            thread.deleteLater()
        self._workers.pop(job_id, None)


