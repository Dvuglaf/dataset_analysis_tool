import csv
import hashlib
import multiprocessing as mp
import threading
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Union

import numpy as np
import pandas as pd
import torch

from custom_types import VideoWrapper


def flat_list(array: list[any], dtype=np.float32) -> np.ndarray:
    if isinstance(array, np.ndarray):
        return array.astype(dtype)
    if isinstance(array, list) and all(isinstance(item, list) for item in array):
        return np.array([item for sublist in array for item in sublist], dtype=dtype)
    return np.array(array, dtype=dtype)


@dataclass
class VideoSegment:
    path: str | Path
    start_frame: int
    end_frame: int
    step: int = 1


@dataclass
class Metric:
    """Результат вычисления метрики"""
    name: str
    value: Union[float, int, np.ndarray]
    raw_data: np.ndarray | None


class BaseMetric(ABC):
    """Базовый класс для метрик"""
    batch_size: int = 1
    device_id: str = "cpu"
    model: Any = None

    def compute(self, ready_frames: list[np.ndarray | torch.Tensor]) -> Metric:
        self._ensure_model()

        all_results = []
        batch = []
        previous_batch = []

        for idx, frame in enumerate(ready_frames):
            if idx % 4 != 0:
                continue
            batch.append(frame)

            if len(batch) >= self.batch_size:
                result = self._process_batch(batch, previous_batch)
                if isinstance(result, list):
                    all_results.extend(result)
                else:
                    all_results.append(result)
                previous_batch = batch.copy()
                batch.clear()

        batch.clear()
        previous_batch.clear()

        if self.name == "temporal_clip_consistency":
            all_results = np.array(all_results)
            return Metric(name=self.name, value=all_results.reshape(-1, 512), raw_data=flat_list(all_results))
        else:
            return Metric(name=self.name, value=np.mean(flat_list(all_results)), raw_data=flat_list(all_results))

    @property
    @abstractmethod
    def name(self) -> str:
        """Название метрики"""
        pass

    def _ensure_model(self) -> None:
        """Lazy-инициализация модели. Переопределить в дочернем классе."""
        pass

    @abstractmethod
    def _process_batch(self, batch: list[np.ndarray], previous_batch: list[np.ndarray] | None = None) -> Any:
        """Обработать батч кадров. Переопределить в дочернем классе."""
        raise NotImplementedError("Метод _process_batch должен быть переопределён")

    def _aggregate_results(self, all_results: List[Any]) -> Union[float, np.ndarray]:
        """Агрегировать результаты всех батчей. По умолчанию - среднее значение."""
        if not all_results:
            return 0.0
        flat = []
        for r in all_results:
            if isinstance(r, list):
                flat.extend(r)
            else:
                flat.append(r)
        return np.mean(flat).item()


def save_metric_result(results_dir: Path, video_path: Path, results: Dict[str, Any], lock, use_hash: bool = True):
    """Сохранить результат одного видео (thread-safe)

    Args:
        use_hash: Добавлять hash пути к имени файла для уникальности (по умолчанию True)
    """
    video_name = video_path.stem

    if use_hash:
        # Создаем уникальный идентификатор на основе полного пути к файлу
        # Это позволит различать файлы с одинаковым именем из разных директорий
        path_hash = hashlib.md5(str(video_path).encode()).hexdigest()[:8]
        unique_prefix = f"{video_name}_{path_hash}"
    else:
        unique_prefix = video_name

    # Сохраняем массивы в отдельные файлы
    for metric_name, value in list(results.items()):
        if isinstance(value, np.ndarray):
            array_filename = f"{unique_prefix}_{metric_name}.npy"
            array_path = results_dir / "arrays" / array_filename
            array_path.parent.mkdir(exist_ok=True)
            np.save(array_path, value)
            # results[metric_name] = str(array_path)
            results[metric_name] = str(array_filename)

    # Thread-safe запись в CSV
    csv_path = results_dir / "video_metrics.csv"
    with lock:
        file_exists = csv_path.exists()
        with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            if results:
                fieldnames = list(results.keys())
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                if not file_exists:
                    writer.writeheader()

                writer.writerow(results)


def compute_video_metrics(video_path: Path, metrics: List[BaseMetric]) -> Dict[str, Any]:
    """Вычислить все метрики для одного видео"""
    results = {
        "video_name": video_path.name,
        "video_path": str(video_path)
    }

    for metric in metrics:
        try:
            result = metric.compute(video_path)
            for m in result.metrics:
                results[m.name] = m.value
        except Exception as e:
            print(f"Ошибка при вычислении {metric.name} для {video_path}: {e}")
            results[metric.name] = None

    return results


def process_videos_worker(
        worker_id: int,
        device_id: int,
        video_queue: mp.Queue,
        results_dir: Path,
        metrics_config: List[Dict],  # Конфигурация метрик
        log_path: Path,
        lock,
        progress_dict: Dict,
        total_videos: int,
        use_hash: bool = True
):
    """Воркер для обработки видео (multiprocessing)"""

    try:
        print(f"Воркер {worker_id} (Device {device_id}) запущен")

        # Инициализируем метрики для этого воркера
        metrics = []
        for metric_cfg in metrics_config:
            metric_class = metric_cfg['class']
            metric_args = metric_cfg.get('args', {}).copy()  # Параметры конструктора
            extra_attrs = metric_cfg.get('extra_attrs', {})  # Дополнительные атрибуты

            # Автоматически добавляем device_id если метрика его поддерживает
            import inspect
            if 'device_id' in inspect.signature(metric_class.__init__).parameters:
                metric_args['device_id'] = device_id

            try:
                # Создаем экземпляр метрики только с параметрами конструктора
                metric_instance = metric_class(**metric_args)

                # Устанавливаем дополнительные атрибуты после создания
                for attr_name, attr_value in extra_attrs.items():
                    setattr(metric_instance, attr_name, attr_value)

                metrics.append(metric_instance)

                if metric_args or extra_attrs:
                    print(f"Воркер {worker_id}: Инициализирована метрика {metric_class.__name__}")
                    if metric_args:
                        print(f"  Параметры конструктора: {metric_args}")
                    if extra_attrs:
                        print(f"  Дополнительные атрибуты: {extra_attrs}")
                else:
                    print(f"Воркер {worker_id}: Инициализирована метрика {metric_class.__name__} (без параметров)")

            except Exception as e:
                print(f"Воркер {worker_id}: Ошибка инициализации метрики {metric_class.__name__}: {e}")
                traceback.print_exc()
                continue

        if not metrics:
            print(f"Воркер {worker_id}: Нет валидных метрик, завершаем работу")
            return

        processed_count = 0

        while True:
            try:
                # Получаем видео из очереди с таймаутом
                video_path = video_queue.get(timeout=30)
                if video_path is None:  # Сигнал завершения
                    break

            except:  # Очередь пуста
                print(f"Воркер {worker_id} - очередь пуста, завершаем работу")
                break

            try:
                # Вычисляем метрики
                results = compute_video_metrics(video_path, metrics)

                # Сохраняем результат
                save_metric_result(results_dir, video_path, results, lock, use_hash)

                processed_count += 1

                # Обновляем прогресс
                with lock:
                    progress_dict[worker_id] = processed_count
                    total_processed = sum(progress_dict.values())
                    if total_processed % 10 == 0 or total_processed == total_videos:
                        print(f"Воркер {worker_id}: {processed_count} видео. "
                              f"Общий прогресс: {total_processed}/{total_videos}")

            except Exception as e:
                with lock:
                    with open(log_path, "a", encoding='utf-8') as log_file:
                        log_file.write(f"Воркер {worker_id} - Ошибка обработки {video_path}: {str(e)}\n")
                        log_file.write(f"Traceback: {traceback.format_exc()}\n")
                print(f"Воркер {worker_id} - Ошибка обработки {video_path}: {str(e)}")

        print(f"Воркер {worker_id} завершил работу. Обработано: {processed_count} видео")

    except Exception as e:
        with lock:
            with open(log_path, "a", encoding='utf-8') as log_file:
                log_file.write(f"Воркер {worker_id} - Критическая ошибка: {str(e)}\n")
                log_file.write(f"Traceback: {traceback.format_exc()}\n")
        print(f"Воркер {worker_id} - Критическая ошибка: {str(e)}")


def progress_monitor(progress_dict: Dict, total_videos: int, lock, stop_event):
    """Мониторинг прогресса выполнения"""
    last_total = 0
    start_time = time.time()

    while not stop_event.is_set():
        time.sleep(60)  # Обновляем каждую минуту

        with lock:
            total_processed = sum(progress_dict.values())

        if total_processed > last_total:
            elapsed_time = time.time() - start_time
            speed = total_processed / elapsed_time * 3600 if elapsed_time > 0 else 0  # видео в час
            eta = (total_videos - total_processed) / (total_processed / elapsed_time) if total_processed > 0 else 0

            print(f"\n=== ПРОГРЕСС ===")
            print(f"Обработано: {total_processed}/{total_videos} ({total_processed / total_videos * 100:.1f}%)")
            print(f"Скорость: {speed:.1f} видео/час")
            print(f"Примерное время до завершения: {eta / 3600:.1f} часов")
            print(f"По воркерам: {dict(progress_dict)}")
            print("===============\n")

            last_total = total_processed


class VideoProcessor:
    """Основной класс для обработки видео с поддержкой multiprocessing"""

    # Поддерживаемые расширения видеофайлов
    VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}

    def __init__(self,
                 output_dir: Path,
                 metrics: List[BaseMetric] = None,
                 num_workers: int = 1,
                 num_gpus: int = None,
                 gpus_id: list[int] = None,
                 use_hash: bool = True):
        """
        Args:
            output_dir: Директория для сохранения результатов
            metrics: Список метрик для вычисления
            num_workers: Количество процессов для обработки
            num_gpus: Количество GPU (None = автоопределение)
            use_hash: Добавлять hash пути к имени файла для уникальности (по умолчанию True)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Директория для сохранения массивов
        self.arrays_dir = self.output_dir / "arrays"
        self.arrays_dir.mkdir(exist_ok=True)

        self.csv_path = self.output_dir / "video_metrics.csv"
        self.log_path = self.output_dir / "processing_errors.log"

        self.metrics = metrics or self._get_default_metrics()
        self.num_workers = num_workers

        # GPU configuration
        try:
            import torch
            self.available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        except ImportError:
            self.available_gpus = 0

        self.num_gpus = num_gpus if num_gpus is not None else self.available_gpus
        self.num_gpus = min(self.num_gpus, self.available_gpus) if self.available_gpus > 0 else 0

        self.gpus_id = gpus_id
        if self.gpus_id:
            self.num_gpus = len(gpus_id)

        self.use_hash = use_hash

        print(f"Доступно GPU: {self.available_gpus}")
        print(f"Будет использовано GPU: {self.num_gpus}")

    def _get_default_metrics(self) -> List[BaseMetric]:
        """Получить список метрик по умолчанию"""
        return [
        ]

    def _metrics_to_config(self) -> List[Dict]:
        """Преобразовать метрики в конфигурацию для передачи в процессы"""
        config = []
        for metric in self.metrics:
            import inspect

            # Получаем параметры конструктора метрики
            constructor_params = inspect.signature(metric.__class__.__init__).parameters
            constructor_param_names = set(constructor_params.keys()) - {'self'}

            metric_cfg = {
                'class': metric.__class__,
                'args': {},
                'extra_attrs': {}  # Для атрибутов, которые не в конструкторе
            }

            # Извлекаем атрибуты метрики
            for attr_name, attr_value in metric.__dict__.items():
                if attr_name.startswith('_') or callable(attr_value):
                    continue

                # Проверяем, что значение можно сериализовать
                try:
                    import pickle
                    pickle.dumps(attr_value)

                    if attr_name in constructor_param_names:
                        # Атрибут является параметром конструктора
                        metric_cfg['args'][attr_name] = attr_value
                    else:
                        # Атрибут не является параметром конструктора
                        metric_cfg['extra_attrs'][attr_name] = attr_value

                except (TypeError, pickle.PicklingError):
                    print(
                        f"Предупреждение: Атрибут {attr_name} метрики {metric.__class__.__name__} не может быть сериализован")

            config.append(metric_cfg)
        return config

    def find_video_files(self, input_dir: Path) -> List[Path]:
        """Найти все видеофайлы в директории"""
        video_files = []
        input_dir = Path(input_dir)

        if not input_dir.exists():
            raise ValueError(f"Директория {input_dir} не существует")

        for file_path in input_dir.rglob("*"):
            if file_path.suffix.lower() in self.VIDEO_EXTENSIONS:
                video_files.append(file_path)

        return video_files

    def process_videos(self, input: Path | list[Path | str]) -> None:
        """Обработать все видео в директории с multiprocessing"""
        print("Поиск видеофайлов...")
        if type(input) is Path:
            video_files = self.find_video_files(input)
        elif type(input) is list:
            video_files = input
        else:
            raise ValueError("")

        if not video_files:
            print(f"Видеофайлы не найдены в директории {input}")
            return

        total_videos = len(video_files)
        print(f"Найдено {total_videos} видеофайлов")
        print(f"Используется метрик: {len(self.metrics)}")
        print(f"Количество воркеров: {self.num_workers}")

        # Очистка CSV файла
        if self.csv_path.exists():
            self.csv_path.unlink()

        # Multiprocessing setup
        ctx = mp.get_context("spawn")  # Важно для CUDA
        video_queue = ctx.Queue()

        # Заполняем очередь видеофайлами
        for video_path in video_files:
            video_queue.put(video_path)

        # Добавляем сигналы завершения
        for _ in range(self.num_workers):
            video_queue.put(None)

        # Конфигурация метрик
        metrics_config = self._metrics_to_config()

        # Структуры для мониторинга
        lock = ctx.Lock()
        manager = ctx.Manager()
        progress_dict = manager.dict({i: 0 for i in range(self.num_workers)})
        stop_event = ctx.Event()

        # Запуск мониторинга прогресса
        progress_thread = threading.Thread(
            target=progress_monitor,
            args=(progress_dict, total_videos, lock, stop_event)
        )
        progress_thread.daemon = True
        progress_thread.start()

        # Определяем device_id для каждого воркера
        def get_device_id(worker_id):
            if self.gpus_id:
                return f"cuda:{self.gpus_id[worker_id]}"
            if self.num_gpus > 0:
                return f"cuda:{worker_id % self.num_gpus}"
            return "cpu"  # CPU

        # Запускаем воркеры
        processes = []
        for i in range(self.num_workers):
            device_id = get_device_id(i)

            p = ctx.Process(
                target=process_videos_worker,
                args=(
                    i,  # worker_id
                    device_id,  # device_id
                    video_queue,  # video_queue
                    self.output_dir,  # results_dir
                    metrics_config,  # metrics_config
                    self.log_path,  # log_path
                    lock,  # lock
                    progress_dict,  # progress_dict
                    total_videos,  # total_videos
                    self.use_hash  # use_hash
                )
            )
            p.start()
            processes.append(p)

        try:
            # Ждем завершения всех процессов
            for p in processes:
                p.join()

        except KeyboardInterrupt:
            print("Остановка по Ctrl+C")
            for p in processes:
                p.terminate()

        finally:
            stop_event.set()
            progress_thread.join(timeout=5)

        print(f"\nОбработка завершена!")
        print(f"Результаты сохранены в: {self.csv_path}")
        if self.arrays_dir.exists() and any(self.arrays_dir.iterdir()):
            print(f"Массивы сохранены в: {self.arrays_dir}")

        # Показываем статистику
        self._show_results()

    def _show_results(self):
        """Показать статистику результатов"""
        if not self.csv_path.exists():
            return

        try:
            df = pd.read_csv(self.csv_path)
            print(f"\n=== РЕЗУЛЬТАТЫ ===")
            print(f"Обработано видео: {len(df)}")

            if 'duration_seconds' in df.columns:
                total_duration = df['duration_seconds'].sum()
                avg_duration = df['duration_seconds'].mean()
                print(f"Общая длительность: {total_duration:.2f} сек ({total_duration / 3600:.2f} часов)")
                print(f"Средняя длительность: {avg_duration:.2f} секунд")

            print("Первые 5 записей:")
            print(df.head())

        except Exception as e:
            print(f"Ошибка при показе результатов: {e}")