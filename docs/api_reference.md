# API Reference

*Generated on 2026-02-10 14:55:49*


## Project Structure

- **start.py** - Module



## api\api_interface.py

### Module: api_interface

API интерфейс для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет REST API для взаимодействия между различными компонентами проекта.

### Classes

#### NanoprobeAPI

```python
class NanoprobeAPI
```

Класс API интерфейса проекта
Обеспечивает REST API для взаимодействия между компонентами и внешними системами.

### Functions

#### main

```python
def main(

)
```

Главная функция для запуска API сервера

#### __init__

```python
def __init__(
self
)
```

Инициализирует API интерфейс

**Parameters:**

- `self`


#### setup_routes

```python
def setup_routes(
self
)
```

Настраивает маршруты API

**Parameters:**

- `self`


#### create_surface

```python
def create_surface(
self
)
```

Создает новую поверхность для симуляции

**Parameters:**

- `self`


#### scan_surface

```python
def scan_surface(
self
)
```

Выполняет сканирование поверхности СЗМ

**Parameters:**

- `self`


#### process_image

```python
def process_image(
self
)
```

Обрабатывает изображение

**Parameters:**

- `self`


#### decode_sstv

```python
def decode_sstv(
self
)
```

Декодирует SSTV сигнал

**Parameters:**

- `self`


#### start_simulation

```python
def start_simulation(
self
)
```

Запускает новую симуляцию

**Parameters:**

- `self`


#### _run_simulation

```python
def _run_simulation(
self, simulation_id: str, simulation_type: str, parameters: Dict[str, Any]
)
```

Выполняет симуляцию в отдельном потоке

Args:
simulation_id: ID симуляции
simulation_type: Тип симуляции
parameters: Параметры симуляции

**Parameters:**

- `self`
- `simulation_id` (*str*)
- `simulation_type` (*str*)
- `parameters` (*Dict[str, Any]*)


#### _generate_simulation_results

```python
def _generate_simulation_results(
self, simulation_type: str, parameters: Dict[str, Any]
)
 -> Dict[str, Any]
```

Генерирует результаты симуляции

Args:
simulation_type: Тип симуляции
parameters: Параметры симуляции

Returns:
Словарь с результатами

**Parameters:**

- `self`
- `simulation_type` (*str*)
- `parameters` (*Dict[str, Any]*)


**Returns:** *Dict[str, Any]*


#### get_simulation_status

```python
def get_simulation_status(
self, simulation_id: str
)
```

Возвращает статус симуляции

Args:
simulation_id: ID симуляции

**Parameters:**

- `self`
- `simulation_id` (*str*)


#### get_simulation_results

```python
def get_simulation_results(
self, simulation_id: str
)
```

Возвращает результаты симуляции

Args:
simulation_id: ID симуляции

**Parameters:**

- `self`
- `simulation_id` (*str*)


#### upload_data

```python
def upload_data(
self
)
```

Загружает данные в систему

**Parameters:**

- `self`


#### list_data

```python
def list_data(
self
)
```

Возвращает список доступных данных

**Parameters:**

- `self`


#### get_system_info

```python
def get_system_info(
self
)
```

Возвращает информацию о системе

**Parameters:**

- `self`


#### get_system_status

```python
def get_system_status(
self
)
```

Возвращает статус системы

**Parameters:**

- `self`


#### run

```python
def run(
self = 'localhost', host: str = 5000, port: int = False, debug: bool
)
```

Запускает API сервер

Args:
host: Хост для прослушивания
port: Порт для прослушивания
debug: Режим отладки

**Parameters:**

- `self`
- `host` (*str*)
- `port` (*int*)
- `debug` (*bool*)




## api\data_exchange.py

### Module: data_exchange

Модуль форматов обмена данными для проекта Лаборатория моделирования нанозонда
Этот модуль определяет стандартные форматы для обмена данными между компонентами проекта.

### Classes

#### DataFormatSpec

```python
class DataFormatSpec
```

Класс спецификации форматов данных
Определяет стандартные форматы для обмена данными между компонентами проекта.

#### SurfaceDataConverter

```python
class SurfaceDataConverter
```

Класс для конвертации данных поверхности
Обеспечивает преобразование данных поверхности между 
различными форматами.

#### ScanResultsConverter

```python
class ScanResultsConverter
```

Класс для конвертации результатов сканирования
Обеспечивает преобразование результатов сканирования между 
различными форматами.

#### ImageDataConverter

```python
class ImageDataConverter
```

Класс для конвертации данных изображений
Обеспечивает преобразование данных изображений между 
различными форматами.

#### SSTVSignalConverter

```python
class SSTVSignalConverter
```

Класс для конвертации данных SSTV сигналов
Обеспечивает преобразование данных SSTV сигналов между 
различными форматами.

#### SimulationConfigConverter

```python
class SimulationConfigConverter
```

Класс для конвертации конфигурации симуляции
Обеспечивает преобразование конфигурации симуляции между 
различными форматами.

#### AnalyticsReportConverter

```python
class AnalyticsReportConverter
```

Класс для конвертации аналитических отчетов
Обеспечивает преобразование аналитических отчетов между 
различными форматами.

#### DataExchangeManager

```python
class DataExchangeManager
```

Класс менеджера обмена данными
Обеспечивает централизованное управление конвертацией данных 
между различными форматами.

### Functions

#### main

```python
def main(

)
```

Главная функция для демонстрации возможностей модуля обмена данными

#### validate_format

`@staticmethod`

```python
@staticmethod
def validate_format(
data: Dict[str, Any], format_type: str
)
 -> bool
```

Проверяет соответствие данных заданному формату

Args:
data: Данные для проверки
format_type: Тип формата

Returns:
True если данные соответствуют формату, иначе False

**Parameters:**

- `data` (*Dict[str, Any]*)
- `format_type` (*str*)


**Returns:** *bool*


#### get_schema

`@staticmethod`

```python
@staticmethod
def get_schema(
format_type: str
)
 -> Dict[str, Any]
```

Возвращает схему для заданного формата

Args:
format_type: Тип формата

Returns:
Словарь с описанием схемы

**Parameters:**

- `format_type` (*str*)


**Returns:** *Dict[str, Any]*


#### numpy_to_standard

`@staticmethod`

```python
@staticmethod
def numpy_to_standard(
surface_array: np.ndarray
)
 -> Dict[str, Any]
```

Преобразует numpy массив поверхности в стандартный формат

Args:
surface_array: Numpy массив поверхности

Returns:
Словарь в стандартном формате

**Parameters:**

- `surface_array` (*np.ndarray*)


**Returns:** *Dict[str, Any]*


#### standard_to_numpy

`@staticmethod`

```python
@staticmethod
def standard_to_numpy(
surface_data: Dict[str, Any]
)
 -> np.ndarray
```

Преобразует стандартный формат в numpy массив

Args:
surface_data: Данные поверхности в стандартном формате

Returns:
Numpy массив поверхности

**Parameters:**

- `surface_data` (*Dict[str, Any]*)


**Returns:** *np.ndarray*


#### encode_base64

`@staticmethod`

```python
@staticmethod
def encode_base64(
surface_array: np.ndarray
)
 -> str
```

Кодирует массив поверхности в base64

Args:
surface_array: Numpy массив поверхности

Returns:
Строка в формате base64

**Parameters:**

- `surface_array` (*np.ndarray*)


**Returns:** *str*


#### decode_base64

`@staticmethod`

```python
@staticmethod
def decode_base64(
encoded_data: str = 'float64', shape: tuple, dtype: str
)
 -> np.ndarray
```

Декодирует base64 строку в массив поверхности

Args:
encoded_data: Закодированные данные в base64
shape: Форма массива (rows, cols)
dtype: Тип данных

Returns:
Numpy массив поверхности

**Parameters:**

- `encoded_data` (*str*)
- `shape` (*tuple*)
- `dtype` (*str*)


**Returns:** *np.ndarray*


#### numpy_to_standard

`@staticmethod`

```python
@staticmethod
def numpy_to_standard(
scan_array: np.ndarray, surface_id: str
)
 -> Dict[str, Any]
```

Преобразует numpy массив результатов сканирования в стандартный формат

Args:
scan_array: Numpy массив результатов сканирования
surface_id: ID поверхности

Returns:
Словарь в стандартном формате

**Parameters:**

- `scan_array` (*np.ndarray*)
- `surface_id` (*str*)


**Returns:** *Dict[str, Any]*


#### standard_to_numpy

`@staticmethod`

```python
@staticmethod
def standard_to_numpy(
scan_data: Dict[str, Any]
)
 -> np.ndarray
```

Преобразует стандартный формат результатов сканирования в numpy массив

Args:
scan_data: Данные сканирования в стандартном формате

Returns:
Numpy массив результатов сканирования

**Parameters:**

- `scan_data` (*Dict[str, Any]*)


**Returns:** *np.ndarray*


#### numpy_to_standard

`@staticmethod`

```python
@staticmethod
def numpy_to_standard(
image_array: np.ndarray
)
 -> Dict[str, Any]
```

Преобразует numpy массив изображения в стандартный формат

Args:
image_array: Numpy массив изображения

Returns:
Словарь в стандартном формате

**Parameters:**

- `image_array` (*np.ndarray*)


**Returns:** *Dict[str, Any]*


#### standard_to_numpy

`@staticmethod`

```python
@staticmethod
def standard_to_numpy(
image_data: Dict[str, Any]
)
 -> np.ndarray
```

Преобразует стандартный формат изображения в numpy массив

Args:
image_data: Данные изображения в стандартном формате

Returns:
Numpy массив изображения

**Parameters:**

- `image_data` (*Dict[str, Any]*)


**Returns:** *np.ndarray*


#### numpy_to_standard

`@staticmethod`

```python
@staticmethod
def numpy_to_standard(
signal_array: np.ndarray = 44100, sample_rate: int
)
 -> Dict[str, Any]
```

Преобразует numpy массив сигнала в стандартный формат

Args:
signal_array: Numpy массив аудиосигнала
sample_rate: Частота дискретизации

Returns:
Словарь в стандартном формате

**Parameters:**

- `signal_array` (*np.ndarray*)
- `sample_rate` (*int*)


**Returns:** *Dict[str, Any]*


#### standard_to_numpy

`@staticmethod`

```python
@staticmethod
def standard_to_numpy(
signal_data: Dict[str, Any]
)
 -> np.ndarray
```

Преобразует стандартный формат сигнала в numpy массив

Args:
signal_data: Данные сигнала в стандартном формате

Returns:
Numpy массив аудиосигнала

**Parameters:**

- `signal_data` (*Dict[str, Any]*)


**Returns:** *np.ndarray*


#### dict_to_standard

`@staticmethod`

```python
@staticmethod
def dict_to_standard(
config_dict: Dict[str, Any]
)
 -> Dict[str, Any]
```

Преобразует словарь конфигурации в стандартный формат

Args:
config_dict: Словарь с параметрами конфигурации

Returns:
Словарь в стандартном формате

**Parameters:**

- `config_dict` (*Dict[str, Any]*)


**Returns:** *Dict[str, Any]*


#### standard_to_dict

`@staticmethod`

```python
@staticmethod
def standard_to_dict(
config_data: Dict[str, Any]
)
 -> Dict[str, Any]
```

Преобразует стандартный формат конфигурации в словарь

Args:
config_data: Данные конфигурации в стандартном формате

Returns:
Словарь с параметрами конфигурации

**Parameters:**

- `config_data` (*Dict[str, Any]*)


**Returns:** *Dict[str, Any]*


#### dict_to_standard

`@staticmethod`

```python
@staticmethod
def dict_to_standard(
metrics_dict: Dict[str, Any], analysis_type: str
)
 -> Dict[str, Any]
```

Преобразует словарь метрик в стандартный формат отчета

Args:
metrics_dict: Словарь с метриками
analysis_type: Тип анализа

Returns:
Словарь в стандартном формате

**Parameters:**

- `metrics_dict` (*Dict[str, Any]*)
- `analysis_type` (*str*)


**Returns:** *Dict[str, Any]*


#### standard_to_dict

`@staticmethod`

```python
@staticmethod
def standard_to_dict(
report_data: Dict[str, Any]
)
 -> Dict[str, Any]
```

Преобразует стандартный формат отчета в словарь метрик

Args:
report_data: Данные отчета в стандартном формате

Returns:
Словарь с метриками

**Parameters:**

- `report_data` (*Dict[str, Any]*)


**Returns:** *Dict[str, Any]*


#### __init__

```python
def __init__(
self
)
```

Инициализирует менеджер обмена данными

**Parameters:**

- `self`


#### convert

```python
def convert(
self, data: Any, from_format: str, to_format: str
)
 -> Any
```

Конвертирует данные из одного формата в другой

Args:
data: Входные данные
from_format: Исходный формат
to_format: Целевой формат

Returns:
Сконвертированные данные

**Parameters:**

- `self`
- `data` (*Any*)
- `from_format` (*str*)
- `to_format` (*str*)


**Returns:** *Any*


#### validate

```python
def validate(
self, data: Any, format_type: str
)
 -> bool
```

Проверяет данные на соответствие формату

Args:
data: Данные для проверки
format_type: Тип формата

Returns:
True если данные соответствуют формату, иначе False

**Parameters:**

- `self`
- `data` (*Any*)
- `format_type` (*str*)


**Returns:** *bool*


#### get_supported_formats

```python
def get_supported_formats(
self
)
 -> List[str]
```

Возвращает список поддерживаемых форматов

Returns:
Список поддерживаемых форматов

**Parameters:**

- `self`


**Returns:** *List[str]*




## components\cpp-spm-hardware-sim\src\spm_simulator.py

### Module: spm_simulator

Модуль симулятора сканирующего зондового микроскопа (СЗМ)
Этот модуль содержит классы и функции для симуляции работы 
сканирующего зондового микроскопа.

### Classes

#### SurfaceModel

```python
class SurfaceModel
```

Класс для моделирования поверхности
Обрабатывает генерацию и загрузку данных о топографии поверхности.
Поддерживает как процедурную генерацию, так и загрузку на основе файлов.

#### ProbeModel

```python
class ProbeModel
```

Класс для модели зонда
Симулирует физическое движение зонда и взаимодействие с поверхностью, 
включая механизмы обратной связи.

#### SPMController

```python
class SPMController
```

Класс для контроллера СЗМ
Управляет общим процессом сканирования, координирует движение зонда 
и собирает данные.

### Functions

#### main

```python
def main(

)
```

Главная функция для демонстрации работы симулятора СЗМ

#### __init__

```python
def __init__(
self = 50, width: int = 50, height: int
)
```

Инициализирует модель поверхности

Args:
width: Ширина поверхности
height: Высота поверхности

**Parameters:**

- `self`
- `width` (*int*)
- `height` (*int*)


#### generate_surface

```python
def generate_surface(
self
)
```

Генерирует случайную поверхность с заданными характеристиками

**Parameters:**

- `self`


#### _add_craters

```python
def _add_craters(
self = 3, num_craters: int
)
```

Добавляет искусственные кратеры на поверхность

**Parameters:**

- `self`
- `num_craters` (*int*)


#### _add_mountains

```python
def _add_mountains(
self = 2, num_mountains: int
)
```

Добавляет искусственные горы на поверхность

**Parameters:**

- `self`
- `num_mountains` (*int*)


#### get_height

```python
def get_height(
self, x: int, y: int
)
 -> float
```

Получает высоту в заданной точке

Args:
x: Координата X
y: Координата Y

Returns:
Высота в точке (x,y)

**Parameters:**

- `self`
- `x` (*int*)
- `y` (*int*)


**Returns:** *float*


#### save_to_file

```python
def save_to_file(
self, filename: str
)
 -> bool
```

Сохраняет модель поверхности в файл

Args:
filename: Имя файла для сохранения

Returns:
bool: True если успешно сохранено, иначе False

**Parameters:**

- `self`
- `filename` (*str*)


**Returns:** *bool*


#### visualize

```python
def visualize(
self = 'Модель поверхности', title: str
)
```

Визуализирует модель поверхности

Args:
title: Заголовок графика

**Parameters:**

- `self`
- `title` (*str*)


#### __init__

```python
def __init__(
self
)
```

Инициализирует модель зонда

**Parameters:**

- `self`


#### set_position

```python
def set_position(
self, new_x: float, new_y: float, new_z: float
)
```

Устанавливает позицию зонда

Args:
new_x: Новая X-координата
new_y: Новая Y-координата
new_z: Новая Z-координата

**Parameters:**

- `self`
- `new_x` (*float*)
- `new_y` (*float*)
- `new_z` (*float*)


#### get_position

```python
def get_position(
self
)
 -> Tuple[float, float, float]
```

Получает текущую позицию зонда

Returns:
Кортеж с координатами (x, y, z)

**Parameters:**

- `self`


**Returns:** *Tuple[float, float, float]*


#### move_to

```python
def move_to(
self = None, target_x: float, target_y: float, target_z: float
)
```

Перемещает зонд к следующей позиции

Args:
target_x: Целевая X-координата
target_y: Целевая Y-координата
target_z: Целевая Z-координата (если None, используется адаптивная высота)

**Parameters:**

- `self`
- `target_x` (*float*)
- `target_y` (*float*)
- `target_z` (*float*)


#### adjust_to_surface

```python
def adjust_to_surface(
self, surface: SurfaceModel
)
 -> float
```

Адаптирует высоту зонда к поверхности

Args:
surface: Модель поверхности

Returns:
Адаптированная высота зонда

**Parameters:**

- `self`
- `surface` (*SurfaceModel*)


**Returns:** *float*


#### __init__

```python
def __init__(
self
)
```

Инициализирует контроллер СЗМ

**Parameters:**

- `self`


#### set_surface

```python
def set_surface(
self, surface: SurfaceModel
)
```

Устанавливает модель поверхности для сканирования

Args:
surface: Модель поверхности

**Parameters:**

- `self`
- `surface` (*SurfaceModel*)


#### scan_surface

```python
def scan_surface(
self
)
```

Выполняет сканирование всей поверхности

**Parameters:**

- `self`


#### save_scan_results

```python
def save_scan_results(
self, filename: str
)
 -> bool
```

Сохраняет результаты сканирования в файл

Args:
filename: Имя файла для сохранения

Returns:
bool: True если успешно сохранено, иначе False

**Parameters:**

- `self`
- `filename` (*str*)


**Returns:** *bool*


#### visualize_scan_results

```python
def visualize_scan_results(
self = 'Результаты сканирования', title: str
)
```

Визуализирует результаты сканирования

Args:
title: Заголовок графика

**Parameters:**

- `self`
- `title` (*str*)




## components\py-sstv-groundstation\src\sstv_decoder.py

### Module: sstv_decoder

Модуль декодирования SSTV
Этот модуль содержит функции для декодирования SSTV-сигналов
в изображения с использованием библиотеки pysstv.

### Classes

#### SSTVDecoder

```python
class SSTVDecoder
```

Класс для декодирования SSTV-сигналов
Обрабатывает декодирование SSTV-сигналов в изображения 
с использованием библиотеки pysstv.

### Functions

#### convert_audio_to_image

```python
def convert_audio_to_image(
audio_data: np.ndarray, sample_rate: int
)
 -> Optional[Image.Image]
```

Конвертирует аудиоданные в изображение

Args:
audio_data: Аудиоданные в формате numpy array
sample_rate: Частота дискретизации аудио

Returns:
Image.Image: Результативное изображение или None при ошибке

**Parameters:**

- `audio_data` (*np.ndarray*)
- `sample_rate` (*int*)


**Returns:** *Optional[Image.Image]*


#### detect_sstv_signal

```python
def detect_sstv_signal(
audio_data: np.ndarray, sample_rate: int
)
 -> Tuple[bool, float]
```

Обнаруживает SSTV-сигнал в аудиоданных

Args:
audio_data: Аудиоданные в формате numpy array
sample_rate: Частота дискретизации аудио

Returns:
Tuple[bool, float]: (True если найден сигнал, приблизительная частота начала)

**Parameters:**

- `audio_data` (*np.ndarray*)
- `sample_rate` (*int*)


**Returns:** *Tuple[bool, float]*


#### __init__

```python
def __init__(
self
)
```

Инициализирует декодер SSTV

**Parameters:**

- `self`


#### decode_from_audio

```python
def decode_from_audio(
self, audio_file: str
)
 -> Optional[Image.Image]
```

Декодирует SSTV-сигнал из аудиофайла

Args:
audio_file: Путь к аудиофайлу с SSTV-сигналом

Returns:
Image.Image: Декодированное изображение или None при ошибке

**Parameters:**

- `self`
- `audio_file` (*str*)


**Returns:** *Optional[Image.Image]*


#### save_decoded_image

```python
def save_decoded_image(
self, filepath: str
)
 -> bool
```

Сохраняет декодированное изображение в файл

Args:
filepath: Путь для сохранения изображения

Returns:
bool: True если изображение успешно сохранено, иначе False

**Parameters:**

- `self`
- `filepath` (*str*)


**Returns:** *bool*




## components\py-surface-image-analyzer\src\image_processor.py

### Module: image_processor

Модуль обработки изображений
Этот модуль содержит функции для загрузки, предварительной обработки 
и базовой манипуляции с изображениями поверхности.

### Classes

#### ImageProcessor

```python
class ImageProcessor
```

Класс для обработки изображений поверхности
Обрабатывает загрузку, предварительную обработку и базовую манипуляцию 
с изображениями поверхности.

### Functions

#### calculate_surface_roughness

```python
def calculate_surface_roughness(
image: np.ndarray
)
 -> float
```

Вычисляет шероховатость поверхности на основе статистики изображения

Args:
image: Входное изображение

Returns:
float: Значение шероховатости поверхности

**Parameters:**

- `image` (*np.ndarray*)


**Returns:** *float*


#### __init__

```python
def __init__(
self
)
```

Инициализирует процессор изображений

**Parameters:**

- `self`


#### load_image

```python
def load_image(
self, filepath: str
)
 -> bool
```

Загружает изображение из файла

Args:
filepath: Путь к файлу изображения

Returns:
bool: True если изображение успешно загружено, иначе False

**Parameters:**

- `self`
- `filepath` (*str*)


**Returns:** *bool*


#### apply_noise_reduction

```python
def apply_noise_reduction(
self = 'gaussian', method: str
)
 -> Optional[np.ndarray]
```

Применяет методы уменьшения шума к изображению

Args:
method: Метод фильтрации ("gaussian", "median", "bilateral")

Returns:
np.ndarray: Обработанное изображение или None при ошибке

**Parameters:**

- `self`
- `method` (*str*)


**Returns:** *Optional[np.ndarray]*


#### detect_edges

```python
def detect_edges(
self = 100, threshold1: int = 200, threshold2: int
)
 -> Optional[np.ndarray]
```

Обнаруживает края на изображении с помощью алгоритма Canny

Args:
threshold1: Первый порог для гистерезиса
threshold2: Второй порог для гистерезиса

Returns:
np.ndarray: Изображение с выделенными краями или None при ошибке

**Parameters:**

- `self`
- `threshold1` (*int*)
- `threshold2` (*int*)


**Returns:** *Optional[np.ndarray]*




## scripts\install-poetry.py

### Module: install-poetry

This script will install Poetry and its dependencies in an isolated fashion.

It will perform the following steps:
* Create a new virtual environment using the built-in venv module, or the virtualenv zipapp if venv is unavailable.
This will be created at a platform-specific path (or `$POETRY_HOME` if `$POETRY_HOME` is set:
- `~/Library/Application Support/pypoetry` on macOS
- `$XDG_DATA_HOME/pypoetry` on Linux/Unix (`$XDG_DATA_HOME` is `~/.local/share` if unset)
- `%APPDATA%\pypoetry` on Windows
* Update pip inside the virtual environment to avoid bugs in older versions.
* Install the latest (or a given) version of Poetry inside this virtual environment using pip.
* Install a `poetry` script into a platform-specific path (or `$POETRY_HOME/bin` if `$POETRY_HOME` is set):
- `~/.local/bin` on Unix
- `%APPDATA%\Python\Scripts` on Windows
* Attempt to inform the user if they need to add this bin directory to their `$PATH`, as well as how to do so.
* Upon failure, write an error log to `poetry-installer-error-<hash>.log and restore any previous environment.

This script performs minimal magic, and should be relatively stable. However, it is optimized for interactive developer
use and trivial pipelines. If you are considering using this script in production, you should consider manually-managed
installs, or use of pipx as alternatives to executing arbitrary, unversioned code from the internet. If you prefer this
script to alternatives, consider maintaining a local copy as part of your infrastructure.

For full documentation, visit https://python-poetry.org/docs/#installation.

### Functions

#### clear_line

```python
def clear_line(
self
)
 -> 'Cursor'
```

Clears all the output from the current line.

**Parameters:**

- `self`


**Returns:** *'Cursor'*


#### clear_line_after

```python
def clear_line_after(
self
)
 -> 'Cursor'
```

Clears all the output from the current line after the current position.

**Parameters:**

- `self`


**Returns:** *'Cursor'*


#### clear_output

```python
def clear_output(
self
)
 -> 'Cursor'
```

Clears all the output from the cursors' current position
to the end of the screen.

**Parameters:**

- `self`


**Returns:** *'Cursor'*


#### clear_screen

```python
def clear_screen(
self
)
 -> 'Cursor'
```

Clears the entire screen.

**Parameters:**

- `self`


**Returns:** *'Cursor'*


#### install

```python
def install(
self, version
)
```

Installs Poetry in $POETRY_HOME.

**Parameters:**

- `self`
- `version`




## security\auth_manager.py

### Module: auth_manager

Модуль управления аутентификацией для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет систему аутентификации и авторизации для API и компонентов проекта.

### Classes

#### AuthManager

```python
class AuthManager
```

Класс управления аутентификацией
Обеспечивает аутентификацию пользователей, генерацию токенов и проверку прав доступа к ресурсам.

### Functions

#### main

```python
def main(

)
```

Главная функция для демонстрации возможностей аутентификации

#### __init__

```python
def __init__(
self = 'auth.db', db_path: str = None, secret_key: str
)
```

Инициализирует менеджер аутентификации

Args:
db_path: Путь к базе данных аутентификации
secret_key: Секретный ключ для подписи токенов

**Parameters:**

- `self`
- `db_path` (*str*)
- `secret_key` (*str*)


#### init_database

```python
def init_database(
self
)
```

Инициализирует базу данных аутентификации

**Parameters:**

- `self`


#### create_default_admin

```python
def create_default_admin(
self
)
```

Создает пользователя администратора по умолчанию

**Parameters:**

- `self`


#### hash_password

```python
def hash_password(
self = None, password: str, salt: str
)
 -> tuple
```

Хэширует пароль с солью

Args:
password: Пароль для хэширования
salt: Соль (если None, генерируется новая)

Returns:
Кортеж (хэш_пароля, соль)

**Parameters:**

- `self`
- `password` (*str*)
- `salt` (*str*)


**Returns:** *tuple*


#### verify_password

```python
def verify_password(
self, password: str, stored_hash: str, salt: str
)
 -> bool
```

Проверяет пароль

Args:
password: Введенный пароль
stored_hash: Сохраненный хэш
salt: Соль

Returns:
True если пароль верен, иначе False

**Parameters:**

- `self`
- `password` (*str*)
- `stored_hash` (*str*)
- `salt` (*str*)


**Returns:** *bool*


#### register_user

```python
def register_user(
self = 'user', username: str, email: str, password: str, role: str
)
 -> bool
```

Регистрирует нового пользователя

Args:
username: Имя пользователя
email: Email пользователя
password: Пароль пользователя
role: Роль пользователя

Returns:
True если регистрация успешна, иначе False

**Parameters:**

- `self`
- `username` (*str*)
- `email` (*str*)
- `password` (*str*)
- `role` (*str*)


**Returns:** *bool*


#### user_exists

```python
def user_exists(
self, username: str
)
 -> bool
```

Проверяет существование пользователя

Args:
username: Имя пользователя

Returns:
True если пользователь существует, иначе False

**Parameters:**

- `self`
- `username` (*str*)


**Returns:** *bool*


#### authenticate_user

```python
def authenticate_user(
self, username: str, password: str
)
 -> Optional[Dict[str, Any]]
```

Аутентифицирует пользователя

Args:
username: Имя пользователя
password: Пароль

Returns:
Словарь с информацией о пользователе или None если аутентификация не удалась

**Parameters:**

- `self`
- `username` (*str*)
- `password` (*str*)


**Returns:** *Optional[Dict[str, Any]]*


#### update_last_login

```python
def update_last_login(
self, user_id: int
)
```

Обновляет время последнего входа

Args:
user_id: ID пользователя

**Parameters:**

- `self`
- `user_id` (*int*)


#### generate_token

```python
def generate_token(
self = 3600, user_id: int, expires_in: int
)
 -> str
```

Генерирует токен аутентификации

Args:
user_id: ID пользователя
expires_in: Время жизни токена в секундах

Returns:
Строка токена

**Parameters:**

- `self`
- `user_id` (*int*)
- `expires_in` (*int*)


**Returns:** *str*


#### store_token

```python
def store_token(
self, user_id: int, token: str, expires_in: int
)
```

Сохраняет токен в базе данных

Args:
user_id: ID пользователя
token: Токен
expires_in: Время жизни в секундах

**Parameters:**

- `self`
- `user_id` (*int*)
- `token` (*str*)
- `expires_in` (*int*)


#### verify_token

```python
def verify_token(
self, token: str
)
 -> Optional[Dict[str, Any]]
```

Проверяет токен аутентификации

Args:
token: Токен для проверки

Returns:
Словарь с информацией о пользователе или None если токен недействителен

**Parameters:**

- `self`
- `token` (*str*)


**Returns:** *Optional[Dict[str, Any]]*


#### token_exists_in_db

```python
def token_exists_in_db(
self, token: str
)
 -> bool
```

Проверяет существование токена в базе данных

Args:
token: Токен для проверки

Returns:
True если токен существует, иначе False

**Parameters:**

- `self`
- `token` (*str*)


**Returns:** *bool*


#### remove_expired_token

```python
def remove_expired_token(
self, token: str
)
```

Удаляет истекший токен из базы данных

Args:
token: Токен для удаления

**Parameters:**

- `self`
- `token` (*str*)


#### get_user_permissions

```python
def get_user_permissions(
self, user_id: int
)
 -> List[Dict[str, str]]
```

Получает права доступа пользователя

Args:
user_id: ID пользователя

Returns:
Список прав доступа

**Parameters:**

- `self`
- `user_id` (*int*)


**Returns:** *List[Dict[str, str]]*


#### has_permission

```python
def has_permission(
self, user_id: int, resource: str, action: str
)
 -> bool
```

Проверяет права доступа пользователя к ресурсу

Args:
user_id: ID пользователя
resource: Ресурс
action: Действие

Returns:
True если доступ разрешен, иначе False

**Parameters:**

- `self`
- `user_id` (*int*)
- `resource` (*str*)
- `action` (*str*)


**Returns:** *bool*


#### require_auth

```python
def require_auth(
self = None, resource: str = None, action: str
)
```

Декоратор для защиты маршрутов аутентификацией

Args:
resource: Ресурс для проверки прав
action: Действие для проверки прав

**Parameters:**

- `self`
- `resource` (*str*)
- `action` (*str*)




## security\data_encryption.py

### Module: data_encryption

Модуль шифрования данных для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет инструменты для шифрования и защиты чувствительных данных проекта.

### Classes

#### DataEncryption

```python
class DataEncryption
```

Класс для шифрования данных
Обеспечивает шифрование и дешифрование чувствительных данных проекта с использованием современных криптографических методов.

#### SecureDataManager

```python
class SecureDataManager
```

Класс безопасного управления данными
Обеспечивает шифрование и безопасное хранение 
конфиденциальных данных проекта.

#### SecurityValidator

```python
class SecurityValidator
```

Класс валидации безопасности
Обеспечивает проверку безопасности данных и 
защиту от распространенных угроз.

### Functions

#### main

```python
def main(

)
```

Главная функция для демонстрации возможностей шифрования

#### __init__

```python
def __init__(
self = None, key: bytes
)
```

Инициализирует шифровальщик данных

Args:
key: Ключ шифрования (если None, генерируется новый)

**Parameters:**

- `self`
- `key` (*bytes*)


#### encrypt_string

```python
def encrypt_string(
self, plaintext: str
)
 -> str
```

Шифрует строку

Args:
plaintext: Открытый текст для шифрования

Returns:
Зашифрованная строка в формате base64

**Parameters:**

- `self`
- `plaintext` (*str*)


**Returns:** *str*


#### decrypt_string

```python
def decrypt_string(
self, encrypted_data: str
)
 -> str
```

Дешифрует строку

Args:
encrypted_data: Зашифрованные данные в формате base64

Returns:
Расшифрованная строка

**Parameters:**

- `self`
- `encrypted_data` (*str*)


**Returns:** *str*


#### encrypt_bytes

```python
def encrypt_bytes(
self, data: bytes
)
 -> bytes
```

Шифрует байты

Args:
data: Байты для шифрования

Returns:
Зашифрованные байты

**Parameters:**

- `self`
- `data` (*bytes*)


**Returns:** *bytes*


#### decrypt_bytes

```python
def decrypt_bytes(
self, encrypted_data: bytes
)
 -> bytes
```

Дешифрует байты

Args:
encrypted_data: Зашифрованные байты

Returns:
Расшифрованные байты

**Parameters:**

- `self`
- `encrypted_data` (*bytes*)


**Returns:** *bytes*


#### encrypt_file

```python
def encrypt_file(
self, input_file: str, output_file: str
)
 -> bool
```

Шифрует содержимое файла

Args:
input_file: Входной файл
output_file: Выходной файл (зашифрованный)

Returns:
True если шифрование успешно, иначе False

**Parameters:**

- `self`
- `input_file` (*str*)
- `output_file` (*str*)


**Returns:** *bool*


#### decrypt_file

```python
def decrypt_file(
self, input_file: str, output_file: str
)
 -> bool
```

Дешифрует содержимое файла

Args:
input_file: Входной файл (зашифрованный)
output_file: Выходной файл (расшифрованный)

Returns:
True если дешифрование успешно, иначе False

**Parameters:**

- `self`
- `input_file` (*str*)
- `output_file` (*str*)


**Returns:** *bool*


#### generate_key_from_password

```python
def generate_key_from_password(
self = None, password: str, salt: bytes
)
 -> Tuple[bytes, bytes]
```

Генерирует ключ шифрования из пароля

Args:
password: Пароль
salt: Соль (если None, генерируется новая)

Returns:
Кортеж (ключ, соль)

**Parameters:**

- `self`
- `password` (*str*)
- `salt` (*bytes*)


**Returns:** *Tuple[bytes, bytes]*


#### __init__

```python
def __init__(
self = None, encryption_key: bytes
)
```

Инициализирует менеджер безопасных данных

Args:
encryption_key: Ключ шифрования

**Parameters:**

- `self`
- `encryption_key` (*bytes*)


#### store_secure_data

```python
def store_secure_data(
self = True, key: str, data: Union[str, dict, list], encrypt: bool
)
 -> bool
```

Сохраняет защищенные данные

Args:
key: Ключ для идентификации данных
data: Данные для сохранения
encrypt: Шифровать ли данные

Returns:
True если сохранение успешно, иначе False

**Parameters:**

- `self`
- `key` (*str*)
- `data` (*Union[str, dict, list]*)
- `encrypt` (*bool*)


**Returns:** *bool*


#### retrieve_secure_data

```python
def retrieve_secure_data(
self = True, key: str, decrypt: bool
)
 -> Optional[Union[str, dict, list]]
```

Получает защищенные данные

Args:
key: Ключ для идентификации данных
decrypt: Дешифровать ли данные

Returns:
Данные или None если не найдены

**Parameters:**

- `self`
- `key` (*str*)
- `decrypt` (*bool*)


**Returns:** *Optional[Union[str, dict, list]]*


#### secure_delete

```python
def secure_delete(
self, key: str
)
 -> bool
```

Безопасно удаляет защищенные данные

Args:
key: Ключ для идентификации данных

Returns:
True если удаление успешно, иначе False

**Parameters:**

- `self`
- `key` (*str*)


**Returns:** *bool*


#### store_sensitive_config

```python
def store_sensitive_config(
self, config_data: dict
)
 -> bool
```

Сохраняет конфигурацию с чувствительными данными

Args:
config_data: Данные конфигурации

Returns:
True если сохранение успешно, иначе False

**Parameters:**

- `self`
- `config_data` (*dict*)


**Returns:** *bool*


#### get_sensitive_config

```python
def get_sensitive_config(
self
)
 -> Optional[dict]
```

Получает конфигурацию с чувствительными данными

Returns:
Конфигурация или None если не найдена

**Parameters:**

- `self`


**Returns:** *Optional[dict]*


#### store_encrypted_file

```python
def store_encrypted_file(
self, filename: str, data: bytes
)
 -> bool
```

Сохраняет зашифрованный файл

Args:
filename: Имя файла
data: Данные для сохранения

Returns:
True если сохранение успешно, иначе False

**Parameters:**

- `self`
- `filename` (*str*)
- `data` (*bytes*)


**Returns:** *bool*


#### retrieve_encrypted_file

```python
def retrieve_encrypted_file(
self, filename: str
)
 -> Optional[bytes]
```

Получает зашифрованный файл

Args:
filename: Имя файла

Returns:
Данные файла или None если не найден

**Parameters:**

- `self`
- `filename` (*str*)


**Returns:** *Optional[bytes]*


#### validate_input

`@staticmethod`

```python
@staticmethod
def validate_input(
input_data: str = 1000, max_length: int
)
 -> bool
```

Проверяет безопасность входных данных

Args:
input_data: Входные данные для проверки
max_length: Максимальная длина

Returns:
True если данные безопасны, иначе False

**Parameters:**

- `input_data` (*str*)
- `max_length` (*int*)


**Returns:** *bool*


#### sanitize_input

`@staticmethod`

```python
@staticmethod
def sanitize_input(
input_data: str
)
 -> str
```

Санитизирует входные данные

Args:
input_data: Входные данные для санитизации

Returns:
Очищенные данные

**Parameters:**

- `input_data` (*str*)


**Returns:** *str*


#### validate_file_type

`@staticmethod`

```python
@staticmethod
def validate_file_type(
filename: str, allowed_extensions: list
)
 -> bool
```

Проверяет тип файла

Args:
filename: Имя файла
allowed_extensions: Разрешенные расширения

Returns:
True если тип файла допустим, иначе False

**Parameters:**

- `filename` (*str*)
- `allowed_extensions` (*list*)


**Returns:** *bool*


#### calculate_checksum

`@staticmethod`

```python
@staticmethod
def calculate_checksum(
data: Union[str, bytes]
)
 -> str
```

Вычисляет контрольную сумму данных

Args:
data: Данные для вычисления контрольной суммы

Returns:
Контрольная сумма в формате hex

**Parameters:**

- `data` (*Union[str, bytes]*)


**Returns:** *str*




## src\cli\dashboard.py

### Module: dashboard

Интерактивная панель управления проектом Лаборатория моделирования нанозонда
Этот модуль предоставляет графический интерфейс для управления всеми компонентами проекта.

### Classes

#### NanoprobeDashboard

```python
class NanoprobeDashboard
```

Класс интерактивной панели управления проектом
Предоставляет графический интерфейс для управления всеми компонентами проекта Лаборатории моделирования нанозонда.

### Functions

#### main

```python
def main(

)
```

Главная функция запуска панели управления

#### __init__

```python
def __init__(
self
)
```

Инициализирует панель управления

**Parameters:**

- `self`


#### create_widgets

```python
def create_widgets(
self
)
```

Создает виджеты интерфейса

**Parameters:**

- `self`


#### create_control_tab

```python
def create_control_tab(
self, parent
)
```

Создает вкладку управления

**Parameters:**

- `self`
- `parent`


#### create_visualization_tab

```python
def create_visualization_tab(
self, parent
)
```

Создает вкладку визуализации

**Parameters:**

- `self`
- `parent`


#### create_logs_tab

```python
def create_logs_tab(
self, parent
)
```

Создает вкладку логов

**Parameters:**

- `self`
- `parent`


#### create_settings_tab

```python
def create_settings_tab(
self, parent
)
```

Создает вкладку настроек

**Parameters:**

- `self`
- `parent`


#### update_status_info

```python
def update_status_info(
self
)
```

Обновляет информацию о состоянии

**Parameters:**

- `self`


#### run_spm_simulation

```python
def run_spm_simulation(
self
)
```

Запускает симуляцию СЗМ

**Parameters:**

- `self`


#### run_image_analysis

```python
def run_image_analysis(
self
)
```

Запускает анализ изображений

**Parameters:**

- `self`


#### run_sstv_decoding

```python
def run_sstv_decoding(
self
)
```

Запускает декодирование SSTV

**Parameters:**

- `self`


#### run_comprehensive_simulation

```python
def run_comprehensive_simulation(
self
)
```

Запускает комплексную симуляцию

**Parameters:**

- `self`


#### run_continuous_simulation

```python
def run_continuous_simulation(
self
)
```

Запускает непрерывную симуляцию

**Parameters:**

- `self`


#### stop_simulation

```python
def stop_simulation(
self
)
```

Останавливает симуляцию

**Parameters:**

- `self`


#### load_and_visualize_data

```python
def load_and_visualize_data(
self
)
```

Загружает и визуализирует данные

**Parameters:**

- `self`


#### create_visualization_report

```python
def create_visualization_report(
self
)
```

Создает отчет визуализации

**Parameters:**

- `self`


#### refresh_logs

```python
def refresh_logs(
self
)
```

Обновляет отображение логов

**Parameters:**

- `self`


#### save_settings

```python
def save_settings(
self
)
```

Сохраняет настройки

**Parameters:**

- `self`


#### run

```python
def run(
self
)
```

Запускает панель управления

**Parameters:**

- `self`




## src\cli\main.py

### Module: main

Главная консольная утилита проекта Лаборатория моделирования нанозонда
Этот скрипт предоставляет интерактивный интерфейс для запуска 
всех компонентов проекта и управления ими.

### Functions

#### show_header

```python
def show_header(

)
```

Отображает заголовок программы

#### show_project_overview

```python
def show_project_overview(

)
```

Отображает обзор проекта

#### show_menu

```python
def show_menu(

)
```

Отображает главное меню

#### run_spm_simulator

```python
def run_spm_simulator(

)
```

Запускает симулятор СЗМ

#### run_surface_analyzer

```python
def run_surface_analyzer(

)
```

Запускает анализатор изображений

#### run_sstv_groundstation

```python
def run_sstv_groundstation(

)
```

Запускает наземную станцию SSTV

#### show_project_info

```python
def show_project_info(

)
```

Показывает информацию о проекте

#### show_license

```python
def show_license(

)
```

Показывает информацию о лицензии

#### clean_project_cache

```python
def clean_project_cache(

)
```

Очищает кэш проекта

#### auto_cleanup_on_exit

```python
def auto_cleanup_on_exit(

)
```

Автоматическая очистка кэша при завершении программы

#### main

```python
def main(

)
```

Главная функция программы



## src\cli\project_manager.py

### Module: project_manager

Менеджер проекта для Лаборатории моделирования нанозонда
Этот скрипт предоставляет унифицированный интерфейс для управления 
всеми компонентами проекта: симулятором СЗМ, анализатором изображений 
и наземной станцией SSTV.

### Classes

#### ProjectManager

```python
class ProjectManager
```

Класс для управления всем проектом Лаборатории моделирования нанозонда
Обеспечивает унифицированный интерфейс для всех компонентов проекта.

### Functions

#### main

```python
def main(

)
```

Главная функция запуска менеджера проекта

#### __init__

```python
def __init__(
self = None, config_file: str
)
```

Инициализирует менеджер проекта

Args:
config_file: Путь к файлу конфигурации проекта

**Parameters:**

- `self`
- `config_file` (*str*)


#### load_config

```python
def load_config(
self
)
 -> Dict
```

Загружает конфигурацию проекта из JSON-файла

Returns:
Словарь с конфигурацией проекта

**Parameters:**

- `self`


**Returns:** *Dict*


#### run_spm_simulator

```python
def run_spm_simulator(
self = True, use_python: bool
)
```

Запускает симулятор СЗМ

Args:
use_python: Использовать Python-реализацию вместо C++

**Parameters:**

- `self`
- `use_python` (*bool*)


#### run_surface_analyzer

```python
def run_surface_analyzer(
self
)
```

Запускает анализатор изображений поверхности

**Parameters:**

- `self`


#### create_sample_analyzer

```python
def create_sample_analyzer(
self
)
```

Создает пример скрипта анализатора изображений

**Parameters:**

- `self`


#### run_sstv_station

```python
def run_sstv_station(
self
)
```

Запускает наземную станцию SSTV

**Parameters:**

- `self`


#### create_sample_sstv_station

```python
def create_sample_sstv_station(
self
)
```

Создает пример скрипта наземной станции SSTV

**Parameters:**

- `self`


#### build_cpp_components

```python
def build_cpp_components(
self
)
```

Собирает C++ компоненты проекта

**Parameters:**

- `self`


#### clean_cache

```python
def clean_cache(
self
)
```

Очищает кэш проекта

**Parameters:**

- `self`


#### _auto_cleanup_on_exit

```python
def _auto_cleanup_on_exit(
self
)
```

Внутренняя функция автоматической очистки при завершении

**Parameters:**

- `self`


#### show_project_info

```python
def show_project_info(
self
)
```

Показывает информацию о проекте

**Parameters:**

- `self`


#### show_menu

```python
def show_menu(
self
)
```

Отображает главное меню проекта

**Parameters:**

- `self`


#### run_interactive

```python
def run_interactive(
self
)
```

Запускает интерактивный режим менеджера проекта

**Parameters:**

- `self`




## src\web\web_dashboard.py

### Module: web_dashboard

Веб-панель управления для Лаборатории моделирования нанозонда
Этот скрипт создает веб-интерфейс для управления всеми компонентами проекта.

### Classes

#### WebDashboard

```python
class WebDashboard
```

Класс веб-панели управления
Обеспечивает веб-интерфейс для управления проектом.

### Functions

#### main

```python
def main(

)
```

Главная функция запуска веб-панели

#### __init__

```python
def __init__(
self = '127.0.0.1', host: str = 5000, port: int
)
```

Инициализирует веб-панель

Args:
host: Хост для запуска сервера
port: Порт для запуска сервера

**Parameters:**

- `self`
- `host` (*str*)
- `port` (*int*)


#### _setup_routes

```python
def _setup_routes(
self
)
```

Настройка маршрутов Flask

**Parameters:**

- `self`


#### _setup_socketio

```python
def _setup_socketio(
self
)
```

Настройка SocketIO для реального времени

**Parameters:**

- `self`


#### _get_project_info

```python
def _get_project_info(
self
)
 -> Dict[str, Any]
```

Получает информацию о проекте

**Parameters:**

- `self`


**Returns:** *Dict[str, Any]*


#### _get_system_metrics

```python
def _get_system_metrics(
self
)
 -> Dict[str, Any]
```

Получает системные метрики

**Parameters:**

- `self`


**Returns:** *Dict[str, Any]*


#### _get_cache_info

```python
def _get_cache_info(
self
)
 -> Dict[str, Any]
```

Получает информацию о кэше

**Parameters:**

- `self`


**Returns:** *Dict[str, Any]*


#### _get_running_processes

```python
def _get_running_processes(
self
)
 -> Dict[str, Any]
```

Получает информацию о запущенных процессах

**Parameters:**

- `self`


**Returns:** *Dict[str, Any]*


#### _get_components_info

```python
def _get_components_info(
self
)
 -> List[Dict[str, Any]]
```

Получает информацию о компонентах проекта

**Parameters:**

- `self`


**Returns:** *List[Dict[str, Any]]*


#### _execute_action

```python
def _execute_action(
self, action: str, data: Dict[str, Any]
)
 -> Dict[str, Any]
```

Выполняет действие

**Parameters:**

- `self`
- `action` (*str*)
- `data` (*Dict[str, Any]*)


**Returns:** *Dict[str, Any]*


#### _start_component

```python
def _start_component(
self, component_name: str
)
 -> Dict[str, Any]
```

Запускает компонент

**Parameters:**

- `self`
- `component_name` (*str*)


**Returns:** *Dict[str, Any]*


#### _stop_component

```python
def _stop_component(
self, component_name: str
)
 -> Dict[str, Any]
```

Останавливает компонент

**Parameters:**

- `self`
- `component_name` (*str*)


**Returns:** *Dict[str, Any]*


#### _get_recent_logs

```python
def _get_recent_logs(
self = 50, limit: int
)
 -> List[Dict[str, Any]]
```

Получает последние записи логов

**Parameters:**

- `self`
- `limit` (*int*)


**Returns:** *List[Dict[str, Any]]*


#### _get_realtime_status

```python
def _get_realtime_status(
self
)
 -> Dict[str, Any]
```

Получает статус в реальном времени

**Parameters:**

- `self`


**Returns:** *Dict[str, Any]*


#### _auto_cleanup_on_exit

```python
def _auto_cleanup_on_exit(
self
)
```

Внутренняя функция автоматической очистки при завершении

**Parameters:**

- `self`


#### start_server

```python
def start_server(
self = True, open_browser: bool
)
```

Запускает веб-сервер

Args:
open_browser: Открыть браузер автоматически

**Parameters:**

- `self`
- `open_browser` (*bool*)


#### index

`@self.app.route('/')`

```python
@self.app.route('/')
def index(

)
```

Главная страница

#### api_status

`@self.app.route('/api/status')`

```python
@self.app.route('/api/status')
def api_status(

)
```

API для получения статуса системы

#### api_components

`@self.app.route('/api/components')`

```python
@self.app.route('/api/components')
def api_components(

)
```

API для получения информации о компонентах

#### api_actions

`@self.app.route('/api/actions/<action>', methods=['POST'])`

```python
@self.app.route('/api/actions/<action>', methods=['POST'])
def api_actions(
action
)
```

API для выполнения действий

**Parameters:**

- `action`


#### api_logs

`@self.app.route('/api/logs')`

```python
@self.app.route('/api/logs')
def api_logs(

)
```

API для получения логов

#### handle_connect

`@self.socketio.on('connect')`

```python
@self.socketio.on('connect')
def handle_connect(

)
```

Обработка подключения клиента

#### handle_disconnect

`@self.socketio.on('disconnect')`

```python
@self.socketio.on('disconnect')
def handle_disconnect(

)
```

Обработка отключения клиента

#### handle_update_request

`@self.socketio.on('request_update')`

```python
@self.socketio.on('request_update')
def handle_update_request(

)
```

Обработка запроса на обновление данных



## start.py

### Module: start

Main entry point for Nanoprobe Simulation Lab
This script provides access to all project components through a unified interface.

### Functions

#### show_help

```python
def show_help(

)
```

Display help information

#### main

```python
def main(

)
```

Main entry point



## utils\analytics.py

### Module: analytics

Модуль аналитики для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет инструменты для анализа данных 
и машинного обучения для результатов симуляции.

### Classes

#### SurfaceAnalytics

```python
class SurfaceAnalytics
```

Класс для анализа данных поверхности
Обеспечивает статистический анализ, кластеризацию и 
предсказательное моделирование для данных поверхности.

#### ImageAnalytics

```python
class ImageAnalytics
```

Класс для анализа изображений
Обеспечивает анализ характеристик изображений и 
обнаружение паттернов в данных изображений.

#### SSTVAnalytics

```python
class SSTVAnalytics
```

Класс для анализа SSTV данных
Обеспечивает анализ характеристик сигналов и 
качество декодирования SSTV.

#### ProjectAnalytics

```python
class ProjectAnalytics
```

Центральный класс аналитики проекта
Объединяет все аналитические модули и предоставляет 
комплексный анализ данных из всех компонентов проекта.

### Functions

#### main

```python
def main(

)
```

Главная функция для демонстрации возможностей аналитического модуля

#### __init__

```python
def __init__(
self
)
```

Инициализирует аналитический модуль для поверхности

**Parameters:**

- `self`


#### calculate_surface_properties

```python
def calculate_surface_properties(
self, surface_data: np.ndarray
)
 -> Dict[str, float]
```

Вычисляет свойства поверхности из данных

Args:
surface_data: Данные поверхности в виде numpy массива

Returns:
Словарь с вычисленными свойствами поверхности

**Parameters:**

- `self`
- `surface_data` (*np.ndarray*)


**Returns:** *Dict[str, float]*


#### _calculate_surface_area

```python
def _calculate_surface_area(
self, surface_data: np.ndarray
)
 -> float
```

Вычисляет площадь поверхности с учетом рельефа

Args:
surface_data: Данные поверхности

Returns:
Площадь поверхности

**Parameters:**

- `self`
- `surface_data` (*np.ndarray*)


**Returns:** *float*


#### cluster_surface_regions

```python
def cluster_surface_regions(
self = 3, surface_data: np.ndarray, n_clusters: int
)
 -> np.ndarray
```

Кластеризует области поверхности по высоте

Args:
surface_data: Данные поверхности
n_clusters: Количество кластеров

Returns:
Массив с метками кластеров

**Parameters:**

- `self`
- `surface_data` (*np.ndarray*)
- `n_clusters` (*int*)


**Returns:** *np.ndarray*


#### dimensionality_reduction

```python
def dimensionality_reduction(
self, surface_data: np.ndarray
)
 -> Tuple[np.ndarray, np.ndarray]
```

Выполняет понижение размерности поверхности с помощью PCA

Args:
surface_data: Данные поверхности

Returns:
Кортеж с преобразованными данными и объясненной дисперсией

**Parameters:**

- `self`
- `surface_data` (*np.ndarray*)


**Returns:** *Tuple[np.ndarray, np.ndarray]*


#### predict_surface_properties

```python
def predict_surface_properties(
self = 'roughness', features: np.ndarray, target_property: str
)
 -> np.ndarray
```

Предсказывает свойства поверхности на основе признаков

Args:
features: Массив признаков
target_property: Целевое свойство для предсказания

Returns:
Предсказанные значения

**Parameters:**

- `self`
- `features` (*np.ndarray*)
- `target_property` (*str*)


**Returns:** *np.ndarray*


#### __init__

```python
def __init__(
self
)
```

Инициализирует аналитический модуль для изображений

**Parameters:**

- `self`


#### calculate_image_features

```python
def calculate_image_features(
self, image_data: np.ndarray
)
 -> Dict[str, float]
```

Вычисляет признаки изображения

Args:
image_data: Данные изображения в виде numpy массива

Returns:
Словарь с вычисленными признаками изображения

**Parameters:**

- `self`
- `image_data` (*np.ndarray*)


**Returns:** *Dict[str, float]*


#### _calculate_entropy

```python
def _calculate_entropy(
self, data: np.ndarray
)
 -> float
```

Вычисляет энтропию изображения

**Parameters:**

- `self`
- `data` (*np.ndarray*)


**Returns:** *float*


#### _calculate_homogeneity

```python
def _calculate_homogeneity(
self, image: np.ndarray
)
 -> float
```

Вычисляет однородность изображения

**Parameters:**

- `self`
- `image` (*np.ndarray*)


**Returns:** *float*


#### detect_patterns

```python
def detect_patterns(
self, image_data: np.ndarray
)
 -> Dict[str, Any]
```

Обнаруживает паттерны в изображении

Args:
image_data: Данные изображения

Returns:
Словарь с обнаруженными паттернами

**Parameters:**

- `self`
- `image_data` (*np.ndarray*)


**Returns:** *Dict[str, Any]*


#### _detect_regions

```python
def _detect_regions(
self, image: np.ndarray
)
 -> int
```

Обнаруживает регионы с похожими характеристиками

**Parameters:**

- `self`
- `image` (*np.ndarray*)


**Returns:** *int*


#### __init__

```python
def __init__(
self
)
```

Инициализирует аналитический модуль для SSTV

**Parameters:**

- `self`


#### analyze_signal_quality

```python
def analyze_signal_quality(
self = 44100, signal_data: np.ndarray, sample_rate: int
)
 -> Dict[str, float]
```

Анализирует качество SSTV сигнала

Args:
signal_data: Данные аудиосигнала
sample_rate: Частота дискретизации

Returns:
Словарь с метриками качества сигнала

**Parameters:**

- `self`
- `signal_data` (*np.ndarray*)
- `sample_rate` (*int*)


**Returns:** *Dict[str, float]*


#### _calculate_zero_crossing_rate

```python
def _calculate_zero_crossing_rate(
self, signal: np.ndarray
)
 -> float
```

Вычисляет скорость пересечения нуля

**Parameters:**

- `self`
- `signal` (*np.ndarray*)


**Returns:** *float*


#### _calculate_spectral_centroid

```python
def _calculate_spectral_centroid(
self, signal: np.ndarray, sample_rate: int
)
 -> float
```

Вычисляет спектроцентроид сигнала

**Parameters:**

- `self`
- `signal` (*np.ndarray*)
- `sample_rate` (*int*)


**Returns:** *float*


#### evaluate_decoding_quality

```python
def evaluate_decoding_quality(
self, original_image: np.ndarray, decoded_image: np.ndarray
)
 -> Dict[str, float]
```

Оценивает качество декодирования SSTV

Args:
original_image: Оригинальное изображение
decoded_image: Декодированное изображение

Returns:
Словарь с метриками качества декодирования

**Parameters:**

- `self`
- `original_image` (*np.ndarray*)
- `decoded_image` (*np.ndarray*)


**Returns:** *Dict[str, float]*


#### __init__

```python
def __init__(
self
)
```

Инициализирует центральный аналитический модуль

**Parameters:**

- `self`


#### generate_comprehensive_report

```python
def generate_comprehensive_report(
self = None, surface_data: Optional[np.ndarray] = None, image_data: Optional[np.ndarray] = None, signal_data: Optional[np.ndarray] = 44100, sample_rate: int
)
 -> Dict[str, Any]
```

Генерирует комплексный аналитический отчет

Args:
surface_data: Данные поверхности
image_data: Данные изображения
signal_data: Данные аудиосигнала
sample_rate: Частота дискретизации сигнала

Returns:
Словарь с комплексным аналитическим отчетом

**Parameters:**

- `self`
- `surface_data` (*Optional[np.ndarray]*)
- `image_data` (*Optional[np.ndarray]*)
- `signal_data` (*Optional[np.ndarray]*)
- `sample_rate` (*int*)


**Returns:** *Dict[str, Any]*


#### visualize_analytics

```python
def visualize_analytics(
self = 'analytics_report.png', analytics_report: Dict[str, Any], output_path: str
)
```

Визуализирует аналитический отчет

Args:
analytics_report: Отчет с аналитикой
output_path: Путь для сохранения визуализации

**Parameters:**

- `self`
- `analytics_report` (*Dict[str, Any]*)
- `output_path` (*str*)


#### save_analytics_report

```python
def save_analytics_report(
self = 'analytics_report.json', report: Dict[str, Any], filename: str
)
```

Сохраняет аналитический отчет в файл

Args:
report: Аналитический отчет
filename: Имя файла для сохранения

**Parameters:**

- `self`
- `report` (*Dict[str, Any]*)
- `filename` (*str*)




## utils\backup_manager.py

### Module: backup_manager

Модуль управления резервным копированием для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет инструменты для создания, 
управления и восстановления резервных копий данных проекта.

### Classes

#### BackupManager

```python
class BackupManager
```

Класс управления резервным копированием
Обеспечивает создание, хранение и восстановление 
резервных копий проекта и его данных.

### Functions

#### main

```python
def main(

)
```

Главная функция для демонстрации возможностей менеджера резервного копирования

#### __init__

```python
def __init__(
self = None, config_manager: Optional[ConfigManager]
)
```

Инициализирует менеджер резервного копирования

Args:
config_manager: Менеджер конфигурации (опционально)

**Parameters:**

- `self`
- `config_manager` (*Optional[ConfigManager]*)


#### _load_metadata

```python
def _load_metadata(
self
)
 -> Dict
```

Загружает метаданные резервных копий

**Parameters:**

- `self`


**Returns:** *Dict*


#### _save_metadata

```python
def _save_metadata(
self
)
```

Сохраняет метаданные резервных копий

**Parameters:**

- `self`


#### create_backup

```python
def create_backup(
self = None, backup_name: str = True, include_outputs: bool = True, compress: bool = False, encrypt: bool = None, encryption_key: bytes
)
 -> Optional[str]
```

Создает резервную копию проекта

Args:
backup_name: Имя резервной копии (если None, генерируется автоматически)
include_outputs: Включать ли выходные данные
compress: Сжимать ли резервную копию
encrypt: Шифровать ли резервную копию
encryption_key: Ключ шифрования (если None, генерируется новый)

Returns:
Путь к созданной резервной копии или None при ошибке

**Parameters:**

- `self`
- `backup_name` (*str*)
- `include_outputs` (*bool*)
- `compress` (*bool*)
- `encrypt` (*bool*)
- `encryption_key` (*bytes*)


**Returns:** *Optional[str]*


#### _create_zip_archive

```python
def _create_zip_archive(
self, source_dir: Path, archive_path: Path
)
```

Создает ZIP архив

Args:
source_dir: Исходная директория
archive_path: Путь к архиву

**Parameters:**

- `self`
- `source_dir` (*Path*)
- `archive_path` (*Path*)


#### _encrypt_file

```python
def _encrypt_file(
self = None, input_file: Path, output_file: Path, key: bytes
)
 -> bool
```

Шифрует файл

Args:
input_file: Входной файл
output_file: Выходной файл
key: Ключ шифрования

Returns:
True если шифрование успешно, иначе False

**Parameters:**

- `self`
- `input_file` (*Path*)
- `output_file` (*Path*)
- `key` (*bytes*)


**Returns:** *bool*


#### restore_backup

```python
def restore_backup(
self = None, backup_name: str = None, restore_path: str, decrypt_key: bytes
)
 -> bool
```

Восстанавливает резервную копию

Args:
backup_name: Имя резервной копии
restore_path: Путь для восстановления (по умолчанию текущая директория)
decrypt_key: Ключ для дешифрования (если резервная копия зашифрована)

Returns:
True если восстановление успешно, иначе False

**Parameters:**

- `self`
- `backup_name` (*str*)
- `restore_path` (*str*)
- `decrypt_key` (*bytes*)


**Returns:** *bool*


#### _decrypt_file

```python
def _decrypt_file(
self, input_file: Path, output_file: Path, key: bytes
)
 -> bool
```

Дешифрует файл

Args:
input_file: Входной файл
output_file: Выходной файл
key: Ключ шифрования

Returns:
True если дешифрование успешно, иначе False

**Parameters:**

- `self`
- `input_file` (*Path*)
- `output_file` (*Path*)
- `key` (*bytes*)


**Returns:** *bool*


#### list_backups

```python
def list_backups(
self
)
 -> List[Dict]
```

Возвращает список резервных копий

Returns:
Список словарей с информацией о резервных копиях

**Parameters:**

- `self`


**Returns:** *List[Dict]*


#### delete_backup

```python
def delete_backup(
self, backup_name: str
)
 -> bool
```

Удаляет резервную копию

Args:
backup_name: Имя резервной копии

Returns:
True если удаление успешно, иначе False

**Parameters:**

- `self`
- `backup_name` (*str*)


**Returns:** *bool*


#### verify_backup_integrity

```python
def verify_backup_integrity(
self, backup_name: str
)
 -> Tuple[bool, str]
```

Проверяет целостность резервной копии

Args:
backup_name: Имя резервной копии

Returns:
Кортеж (успешно, сообщение)

**Parameters:**

- `self`
- `backup_name` (*str*)


**Returns:** *Tuple[bool, str]*


#### _get_file_size

```python
def _get_file_size(
self, path: Path
)
 -> int
```

Получает размер файла или директории

Args:
path: Путь к файлу или директории

Returns:
Размер в байтах

**Parameters:**

- `self`
- `path` (*Path*)


**Returns:** *int*


#### cleanup_old_backups

```python
def cleanup_old_backups(
self = 30, keep_days: int = 5, keep_count: int
)
 -> int
```

Удаляет старые резервные копии

Args:
keep_days: Хранить резервные копии не менее указанного количества дней
keep_count: Оставлять не менее указанного количества резервных копий

Returns:
Количество удаленных резервных копий

**Parameters:**

- `self`
- `keep_days` (*int*)
- `keep_count` (*int*)


**Returns:** *int*




## utils\cache_manager.py

### Module: cache_manager

Модуль управления кэшем для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет инструменты для автоматической 
очистки и управления кэшем проекта.

### Classes

#### CacheInfo

`@dataclass`

```python
class CacheInfo
```

Информация о кэше

#### CacheManager

```python
class CacheManager
```

Класс менеджера кэша
Обеспечивает автоматическую очистку и 
управление кэшем проекта.

### Functions

#### main

```python
def main(

)
```

Главная функция для демонстрации возможностей менеджера кэша

#### __init__

```python
def __init__(
self = '.', project_root: str = 'cache_config.json', config_file: str
)
```

Инициализирует менеджер кэша

Args:
project_root: Корневая директория проекта
config_file: Файл конфигурации кэша

**Parameters:**

- `self`
- `project_root` (*str*)
- `config_file` (*str*)


#### _load_config

```python
def _load_config(
self
)
 -> Dict
```

Загружает конфигурацию кэша

Returns:
Словарь с конфигурацией кэша

**Parameters:**

- `self`


**Returns:** *Dict*


#### _get_cache_directories

```python
def _get_cache_directories(
self
)
 -> List[Path]
```

Получает список директорий кэша

Returns:
Список путей к директориям кэша

**Parameters:**

- `self`


**Returns:** *List[Path]*


#### analyze_cache

```python
def analyze_cache(
self
)
 -> List[CacheInfo]
```

Анализирует кэш проекта

Returns:
Список информации о кэше

**Parameters:**

- `self`


**Returns:** *List[CacheInfo]*


#### _determine_cache_type

```python
def _determine_cache_type(
self, cache_path: Path
)
 -> str
```

Определяет тип кэша по пути

Args:
cache_path: Путь к директории кэша

Returns:
Тип кэша

**Parameters:**

- `self`
- `cache_path` (*Path*)


**Returns:** *str*


#### cleanup_cache

```python
def cleanup_cache(
self = None, max_age_days: Optional[int] = None, max_size_mb: Optional[int] = False, force: bool
)
 -> Dict[str, Union[int, List[str]]]
```

Очищает кэш проекта

Args:
max_age_days: Максимальный возраст файлов в днях
max_size_mb: Максимальный размер кэша в мегабайтах
force: Принудительная очистка без проверок

Returns:
Словарь с результатами очистки

**Parameters:**

- `self`
- `max_age_days` (*Optional[int]*)
- `max_size_mb` (*Optional[int]*)
- `force` (*bool*)


**Returns:** *Dict[str, Union[int, List[str]]]*


#### _cleanup_python_cache

```python
def _cleanup_python_cache(
self
)
```

Очищает системный кэш Python

**Parameters:**

- `self`


#### auto_cleanup

```python
def auto_cleanup(
self
)
 -> Dict[str, Union[int, List[str]]]
```

Автоматическая очистка кэша по расписанию

Returns:
Словарь с результатами очистки

**Parameters:**

- `self`


**Returns:** *Dict[str, Union[int, List[str]]]*


#### get_cache_statistics

```python
def get_cache_statistics(
self
)
 -> Dict[str, Union[int, float, str]]
```

Получает статистику кэша

Returns:
Словарь со статистикой кэша

**Parameters:**

- `self`


**Returns:** *Dict[str, Union[int, float, str]]*


#### optimize_memory_usage

```python
def optimize_memory_usage(
self
)
 -> Dict[str, Union[int, float]]
```

Оптимизирует использование памяти

Returns:
Словарь с результатами оптимизации

**Parameters:**

- `self`


**Returns:** *Dict[str, Union[int, float]]*


#### generate_cleanup_report

```python
def generate_cleanup_report(
self = None, output_path: str
)
 -> str
```

Генерирует отчет об очистке кэша

Args:
output_path: Путь для сохранения отчета (если None, генерируется автоматически)

Returns:
Путь к созданному отчету

**Parameters:**

- `self`
- `output_path` (*str*)


**Returns:** *str*




## utils\code_analyzer.py

### Module: code_analyzer

Модуль анализа кода для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет инструменты для статического анализа кода,
обнаружения потенциальных проблем и автоматического рефакторинга.

### Classes

#### CodeIssue

`@dataclass`

```python
class CodeIssue
```

Проблема в коде

#### CodeMetrics

`@dataclass`

```python
class CodeMetrics
```

Метрики кода

#### CodeAnalyzer

```python
class CodeAnalyzer
```

Класс анализатора кода
Обеспечивает статический анализ кода, обнаружение проблем
и оценку качества кода проекта.

### Functions

#### main

```python
def main(

)
```

Главная функция для демонстрации возможностей анализатора кода

#### __init__

```python
def __init__(
self = '.', project_root: str
)
```

Инициализирует анализатор кода

Args:
project_root: Корневая директория проекта

**Parameters:**

- `self`
- `project_root` (*str*)


#### analyze_project

```python
def analyze_project(
self = None, include_patterns: List[str] = None, exclude_patterns: List[str]
)
 -> Dict[str, Any]
```

Анализирует весь проект

Args:
include_patterns: Паттерны для включения файлов
exclude_patterns: Паттерны для исключения файлов

Returns:
Результаты анализа проекта

**Parameters:**

- `self`
- `include_patterns` (*List[str]*)
- `exclude_patterns` (*List[str]*)


**Returns:** *Dict[str, Any]*


#### analyze_file

```python
def analyze_file(
self, file_path: Path
)
 -> List[CodeIssue]
```

Анализирует отдельный файл

Args:
file_path: Путь к файлу

Returns:
Список обнаруженных проблем

**Parameters:**

- `self`
- `file_path` (*Path*)


**Returns:** *List[CodeIssue]*


#### _calculate_metrics

```python
def _calculate_metrics(
self, file_path: Path, content: str
)
 -> CodeMetrics
```

Вычисляет метрики кода

**Parameters:**

- `self`
- `file_path` (*Path*)
- `content` (*str*)


**Returns:** *CodeMetrics*


#### _check_naming_conventions

```python
def _check_naming_conventions(
self, file_path: Path, content: str
)
 -> List[CodeIssue]
```

Проверяет соблюдение соглашений об именовании

**Parameters:**

- `self`
- `file_path` (*Path*)
- `content` (*str*)


**Returns:** *List[CodeIssue]*


#### _check_code_complexity

```python
def _check_code_complexity(
self, file_path: Path, content: str
)
 -> List[CodeIssue]
```

Проверяет сложность кода

**Parameters:**

- `self`
- `file_path` (*Path*)
- `content` (*str*)


**Returns:** *List[CodeIssue]*


#### _check_best_practices

```python
def _check_best_practices(
self, file_path: Path, content: str
)
 -> List[CodeIssue]
```

Проверяет соблюдение лучших практик

**Parameters:**

- `self`
- `file_path` (*Path*)
- `content` (*str*)


**Returns:** *List[CodeIssue]*


#### _check_security_issues

```python
def _check_security_issues(
self, file_path: Path, content: str
)
 -> List[CodeIssue]
```

Проверяет потенциальные проблемы безопасности

**Parameters:**

- `self`
- `file_path` (*Path*)
- `content` (*str*)


**Returns:** *List[CodeIssue]*


#### _check_performance_issues

```python
def _check_performance_issues(
self, file_path: Path, content: str
)
 -> List[CodeIssue]
```

Проверяет потенциальные проблемы производительности

**Parameters:**

- `self`
- `file_path` (*Path*)
- `content` (*str*)


**Returns:** *List[CodeIssue]*


#### _generate_analysis_report

```python
def _generate_analysis_report(
self
)
 -> Dict[str, Any]
```

Генерирует отчет по анализу кода

**Parameters:**

- `self`


**Returns:** *Dict[str, Any]*


#### save_report

```python
def save_report(
self = 'code_analysis_report.json', report: Dict[str, Any], output_path: str
)
```

Сохраняет отчет в файл

**Parameters:**

- `self`
- `report` (*Dict[str, Any]*)
- `output_path` (*str*)




## utils\config_manager.py

### Module: config_manager

Модуль управления конфигурацией для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет централизованное управление конфигурацией 
для всех компонентов проекта.

### Classes

#### ConfigManager

```python
class ConfigManager
```

Класс для управления конфигурацией проекта
Обеспечивает централизованное хранение и доступ к параметрам конфигурации 
для всех компонентов проекта.

### Functions

#### main

```python
def main(

)
```

Главная функция для демонстрации работы менеджера конфигурации

#### __init__

```python
def __init__(
self = 'config.json', config_file: str
)
```

Инициализирует менеджер конфигурации

Args:
config_file: Путь к файлу конфигурации

**Parameters:**

- `self`
- `config_file` (*str*)


#### load_config

```python
def load_config(
self
)
 -> Dict[str, Any]
```

Загружает конфигурацию из файла

Returns:
Словарь с параметрами конфигурации

**Parameters:**

- `self`


**Returns:** *Dict[str, Any]*


#### save_config

```python
def save_config(
self
)
 -> bool
```

Сохраняет текущую конфигурацию в файл

Returns:
bool: True если успешно сохранено, иначе False

**Parameters:**

- `self`


**Returns:** *bool*


#### get_default_config

```python
def get_default_config(
self
)
 -> Dict[str, Any]
```

Возвращает стандартную конфигурацию проекта

Returns:
Словарь со стандартными параметрами конфигурации

**Parameters:**

- `self`


**Returns:** *Dict[str, Any]*


#### create_default_config

```python
def create_default_config(
self
)
```

Создает стандартный файл конфигурации

**Parameters:**

- `self`


#### get

```python
def get(
self = None, key_path: str, default: Any
)
 -> Any
```

Получает значение конфигурации по пути ключа

Args:
key_path: Путь к ключу в формате 'section.subsection.key'
default: Значение по умолчанию, если ключ не найден

Returns:
Значение конфигурации или значение по умолчанию

**Parameters:**

- `self`
- `key_path` (*str*)
- `default` (*Any*)


**Returns:** *Any*


#### set

```python
def set(
self, key_path: str, value: Any
)
 -> bool
```

Устанавливает значение конфигурации по пути ключа

Args:
key_path: Путь к ключу в формате 'section.subsection.key'
value: Новое значение

Returns:
bool: True если успешно установлено, иначе False

**Parameters:**

- `self`
- `key_path` (*str*)
- `value` (*Any*)


**Returns:** *bool*


#### update_component_config

```python
def update_component_config(
self, component_name: str, new_config: Dict[str, Any]
)
 -> bool
```

Обновляет конфигурацию компонента

Args:
component_name: Название компонента
new_config: Новая конфигурация компонента

Returns:
bool: True если успешно обновлено, иначе False

**Parameters:**

- `self`
- `component_name` (*str*)
- `new_config` (*Dict[str, Any]*)


**Returns:** *bool*


#### get_component_config

```python
def get_component_config(
self, component_name: str
)
 -> Optional[Dict[str, Any]]
```

Получает конфигурацию компонента

Args:
component_name: Название компонента

Returns:
Конфигурация компонента или None если компонент не найден

**Parameters:**

- `self`
- `component_name` (*str*)


**Returns:** *Optional[Dict[str, Any]]*


#### validate_config

```python
def validate_config(
self
)
 -> bool
```

Проверяет валидность конфигурации

Returns:
bool: True если конфигурация валидна, иначе False

**Parameters:**

- `self`


**Returns:** *bool*




## utils\config_optimizer.py

### Module: config_optimizer

Модуль оптимизации конфигурации для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет инструменты для оптимизации 
конфигурации и параметров проекта.

### Classes

#### OptimizationParams

`@dataclass`

```python
class OptimizationParams
```

Параметры оптимизации

#### ConfigOptimizer

```python
class ConfigOptimizer
```

Класс оптимизатора конфигурации
Обеспечивает оптимизацию параметров конфигурации 
на основе текущего состояния системы.

#### AdaptiveConfigManager

```python
class AdaptiveConfigManager
```

Класс адаптивного управления конфигурацией
Обеспечивает динамическую адаптацию параметров 
конфигурации в зависимости от условий выполнения.

### Functions

#### main

```python
def main(

)
```

Главная функция для демонстрации возможностей оптимизатора конфигурации

#### __init__

```python
def __init__(
self = 'config.json', config_path: str
)
```

Инициализирует оптимизатор конфигурации

Args:
config_path: Путь к файлу конфигурации

**Parameters:**

- `self`
- `config_path` (*str*)


#### load_config

```python
def load_config(
self
)
 -> bool
```

Загружает конфигурацию из файла

Returns:
True если загрузка успешна, иначе False

**Parameters:**

- `self`


**Returns:** *bool*


#### save_config

```python
def save_config(
self = None, config: Dict[str, Any], output_path: str
)
 -> bool
```

Сохраняет конфигурацию в файл

Args:
config: Конфигурация для сохранения
output_path: Путь для сохранения (если None, используется исходный путь)

Returns:
True если сохранение успешно, иначе False

**Parameters:**

- `self`
- `config` (*Dict[str, Any]*)
- `output_path` (*str*)


**Returns:** *bool*


#### collect_system_metrics

```python
def collect_system_metrics(
self
)
 -> Dict[str, Any]
```

Собирает метрики системы

Returns:
Словарь с метриками системы

**Parameters:**

- `self`


**Returns:** *Dict[str, Any]*


#### optimize_for_performance

```python
def optimize_for_performance(
self
)
 -> Dict[str, Any]
```

Оптимизирует конфигурацию для производительности

Returns:
Оптимизированная конфигурация

**Parameters:**

- `self`


**Returns:** *Dict[str, Any]*


#### optimize_for_resource_efficiency

```python
def optimize_for_resource_efficiency(
self
)
 -> Dict[str, Any]
```

Оптимизирует конфигурацию для эффективного использования ресурсов

Returns:
Оптимизированная конфигурация

**Parameters:**

- `self`


**Returns:** *Dict[str, Any]*


#### optimize_for_stability

```python
def optimize_for_stability(
self
)
 -> Dict[str, Any]
```

Оптимизирует конфигурацию для стабильности

Returns:
Оптимизированная конфигурация

**Parameters:**

- `self`


**Returns:** *Dict[str, Any]*


#### optimize_config

```python
def optimize_config(
self = 'balanced', strategy: str
)
 -> Dict[str, Any]
```

Оптимизирует конфигурацию по заданной стратегии

Args:
strategy: Стратегия оптимизации ("performance", "efficiency", "stability", "balanced")

Returns:
Оптимизированная конфигурация

**Parameters:**

- `self`
- `strategy` (*str*)


**Returns:** *Dict[str, Any]*


#### apply_optimization

```python
def apply_optimization(
self = 'balanced', strategy: str = True, save_to_file: bool
)
 -> bool
```

Применяет оптимизацию к конфигурации

Args:
strategy: Стратегия оптимизации
save_to_file: Сохранять ли результат в файл

Returns:
True если применение успешно, иначе False

**Parameters:**

- `self`
- `strategy` (*str*)
- `save_to_file` (*bool*)


**Returns:** *bool*


#### get_optimization_report

```python
def get_optimization_report(
self
)
 -> Dict[str, Any]
```

Генерирует отчет об оптимизации

Returns:
Словарь с отчетом об оптимизации

**Parameters:**

- `self`


**Returns:** *Dict[str, Any]*


#### auto_optimize

```python
def auto_optimize(
self = 'balanced', strategy: str
)
 -> Dict[str, Any]
```

Автоматически оптимизирует конфигурацию

Args:
strategy: Стратегия оптимизации

Returns:
Отчет об оптимизации

**Parameters:**

- `self`
- `strategy` (*str*)


**Returns:** *Dict[str, Any]*


#### optimize_multiple_configs

```python
def optimize_multiple_configs(
self = 'balanced', config_paths: List[str], strategy: str
)
 -> Dict[str, Any]
```

Оптимизирует несколько конфигурационных файлов

Args:
config_paths: Список путей к конфигурационным файлам
strategy: Стратегия оптимизации

Returns:
Словарь с результатами оптимизации для каждого файла

**Parameters:**

- `self`
- `config_paths` (*List[str]*)
- `strategy` (*str*)


**Returns:** *Dict[str, Any]*


#### __init__

```python
def __init__(
self = 'config.json', base_config_path: str
)
```

Инициализирует адаптивный менеджер конфигурации

Args:
base_config_path: Базовый путь к конфигурации

**Parameters:**

- `self`
- `base_config_path` (*str*)


#### start_adaptive_monitoring

```python
def start_adaptive_monitoring(
self = 60.0, interval: float = 'balanced', strategy: str
)
```

Запускает адаптивный мониторинг и оптимизацию

Args:
interval: Интервал между проверками (в секундах)
strategy: Стратегия оптимизации

**Parameters:**

- `self`
- `interval` (*float*)
- `strategy` (*str*)


#### stop_adaptive_monitoring

```python
def stop_adaptive_monitoring(
self
)
```

Останавливает адаптивный мониторинг

**Parameters:**

- `self`


#### _monitoring_loop

```python
def _monitoring_loop(
self, interval: float, strategy: str
)
```

Цикл адаптивного мониторинга

Args:
interval: Интервал между проверками
strategy: Стратегия оптимизации

**Parameters:**

- `self`
- `interval` (*float*)
- `strategy` (*str*)


#### should_adapt

```python
def should_adapt(
self
)
 -> bool
```

Проверяет, нужно ли применять адаптацию

Returns:
True если адаптация необходима, иначе False

**Parameters:**

- `self`


**Returns:** *bool*


#### set_adaptation_callback

```python
def set_adaptation_callback(
self, callback: callable
)
```

Устанавливает callback для уведомлений об адаптации

Args:
callback: Функция обратного вызова

**Parameters:**

- `self`
- `callback` (*callable*)


#### get_adaptation_history

```python
def get_adaptation_history(
self
)
 -> List[Dict[str, Any]]
```

Возвращает историю адаптаций

Returns:
Список записей истории адаптаций

**Parameters:**

- `self`


**Returns:** *List[Dict[str, Any]]*


#### clear_adaptation_history

```python
def clear_adaptation_history(
self
)
```

Очищает историю адаптаций

**Parameters:**

- `self`




## utils\config_validator.py

### Module: config_validator

Модуль валидации конфигурации для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет инструменты для проверки 
и валидации конфигурационных файлов проекта.

### Classes

#### ConfigValidator

```python
class ConfigValidator
```

Класс валидации конфигурации
Обеспечивает проверку корректности конфигурационных 
файлов и параметров проекта.

#### OptimizationAdvisor

```python
class OptimizationAdvisor
```

Класс советника по оптимизации
Предоставляет рекомендации по оптимизации 
производительности и ресурсов проекта.

### Functions

#### main

```python
def main(

)
```

Главная функция для демонстрации возможностей валидации конфигурации

#### __init__

```python
def __init__(
self
)
```

Инициализирует валидатор конфигурации

**Parameters:**

- `self`


#### validate_json_config

```python
def validate_json_config(
self = None, config_path: str, schema: Optional[Dict]
)
 -> Dict[str, Any]
```

Валидирует JSON конфигурационный файл

Args:
config_path: Путь к конфигурационному файлу
schema: JSON схема для валидации (если None, используется схема по умолчанию)

Returns:
Словарь с результатами валидации

**Parameters:**

- `self`
- `config_path` (*str*)
- `schema` (*Optional[Dict]*)


**Returns:** *Dict[str, Any]*


#### validate_yaml_config

```python
def validate_yaml_config(
self = None, config_path: str, schema: Optional[Dict]
)
 -> Dict[str, Any]
```

Валидирует YAML конфигурационный файл

Args:
config_path: Путь к конфигурационному файлу
schema: JSON схема для валидации (если None, используется схема по умолчанию)

Returns:
Словарь с результатами валидации

**Parameters:**

- `self`
- `config_path` (*str*)
- `schema` (*Optional[Dict]*)


**Returns:** *Dict[str, Any]*


#### get_default_config_schema

```python
def get_default_config_schema(
self
)
 -> Dict[str, Any]
```

Возвращает схему по умолчанию для валидации конфигурации проекта

Returns:
Словарь с JSON схемой

**Parameters:**

- `self`


**Returns:** *Dict[str, Any]*


#### validate_config_against_schema

```python
def validate_config_against_schema(
self, config: Dict[str, Any], schema: Dict[str, Any]
)
 -> Dict[str, Any]
```

Валидирует конфигурацию по заданной схеме

Args:
config: Конфигурация для валидации
schema: Схема валидации

Returns:
Словарь с результатами валидации

**Parameters:**

- `self`
- `config` (*Dict[str, Any]*)
- `schema` (*Dict[str, Any]*)


**Returns:** *Dict[str, Any]*


#### validate_project_structure

```python
def validate_project_structure(
self = '.', project_root: str
)
 -> Dict[str, Any]
```

Валидирует структуру проекта

Args:
project_root: Корневая директория проекта

Returns:
Словарь с результатами валидации

**Parameters:**

- `self`
- `project_root` (*str*)


**Returns:** *Dict[str, Any]*


#### validate_dependencies

```python
def validate_dependencies(
self = 'requirements.txt', requirements_path: str
)
 -> Dict[str, Any]
```

Валидирует зависимости проекта

Args:
requirements_path: Путь к файлу зависимостей

Returns:
Словарь с результатами валидации

**Parameters:**

- `self`
- `requirements_path` (*str*)


**Returns:** *Dict[str, Any]*


#### generate_validation_report

```python
def generate_validation_report(
self = None, validation_results: Dict[str, Any], output_path: str
)
 -> str
```

Генерирует отчет о валидации

Args:
validation_results: Результаты валидации
output_path: Путь для сохранения отчета (если None, генерируется автоматически)

Returns:
Путь к созданному отчету

**Parameters:**

- `self`
- `validation_results` (*Dict[str, Any]*)
- `output_path` (*str*)


**Returns:** *str*


#### _generate_validation_summary

```python
def _generate_validation_summary(
self, validation_results: Dict[str, Any]
)
 -> Dict[str, Any]
```

Генерирует сводку по результатам валидации

Args:
validation_results: Результаты валидации

Returns:
Словарь со сводкой

**Parameters:**

- `self`
- `validation_results` (*Dict[str, Any]*)


**Returns:** *Dict[str, Any]*


#### __init__

```python
def __init__(
self
)
```

Инициализирует советника по оптимизации

**Parameters:**

- `self`


#### analyze_performance_data

```python
def analyze_performance_data(
self, performance_metrics: Dict[str, Any]
)
 -> List[Dict[str, str]]
```

Анализирует данные о производительности и дает рекомендации

Args:
performance_metrics: Метрики производительности

Returns:
Список рекомендаций по оптимизации

**Parameters:**

- `self`
- `performance_metrics` (*Dict[str, Any]*)


**Returns:** *List[Dict[str, str]]*


#### analyze_code_complexity

```python
def analyze_code_complexity(
self, code_path: str
)
 -> List[Dict[str, str]]
```

Анализирует сложность кода и дает рекомендации

Args:
code_path: Путь к файлу кода

Returns:
Список рекомендаций по оптимизации

**Parameters:**

- `self`
- `code_path` (*str*)


**Returns:** *List[Dict[str, str]]*


#### generate_optimization_report

```python
def generate_optimization_report(
self = None, recommendations: List[Dict[str, str]], output_path: str
)
 -> str
```

Генерирует отчет по оптимизации

Args:
recommendations: Рекомендации по оптимизации
output_path: Путь для сохранения отчета (если None, генерируется автоматически)

Returns:
Путь к созданному отчету

**Parameters:**

- `self`
- `recommendations` (*List[Dict[str, str]]*)
- `output_path` (*str*)


**Returns:** *str*


#### _categorize_recommendations

```python
def _categorize_recommendations(
self, recommendations: List[Dict[str, str]]
)
 -> Dict[str, int]
```

Категоризирует рекомендации по типам

Args:
recommendations: Список рекомендаций

Returns:
Словарь с количеством рекомендаций по категориям

**Parameters:**

- `self`
- `recommendations` (*List[Dict[str, str]]*)


**Returns:** *Dict[str, int]*




## utils\data_integrity.py

### Module: data_integrity

Модуль проверки целостности данных для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет инструменты для проверки 
целостности и корректности данных проекта.

### Classes

#### DataIntegrityChecker

```python
class DataIntegrityChecker
```

Класс проверки целостности данных
Обеспечивает проверку целостности, корректности и 
валидность данных проекта.

#### IntegrityReportGenerator

```python
class IntegrityReportGenerator
```

Класс генерации отчетов о целостности
Создает отчеты о проверке целостности данных проекта.

### Functions

#### main

```python
def main(

)
```

Главная функция для демонстрации возможностей проверки целостности данных

#### __init__

```python
def __init__(
self
)
```

Инициализирует проверяльщик целостности данных

**Parameters:**

- `self`


#### calculate_checksum

```python
def calculate_checksum(
self, data: bytes
)
 -> str
```

Вычисляет контрольную сумму данных

Args:
data: Данные для вычисления контрольной суммы

Returns:
Контрольная сумма в формате hex

**Parameters:**

- `self`
- `data` (*bytes*)


**Returns:** *str*


#### calculate_file_checksum

```python
def calculate_file_checksum(
self, file_path: str
)
 -> Optional[str]
```

Вычисляет контрольную сумму файла

Args:
file_path: Путь к файлу

Returns:
Контрольная сумма в формате hex или None при ошибке

**Parameters:**

- `self`
- `file_path` (*str*)


**Returns:** *Optional[str]*


#### verify_file_integrity

```python
def verify_file_integrity(
self, file_path: str, expected_checksum: str
)
 -> bool
```

Проверяет целостность файла

Args:
file_path: Путь к файлу
expected_checksum: Ожидаемая контрольная сумма

Returns:
True если файл цел, иначе False

**Parameters:**

- `self`
- `file_path` (*str*)
- `expected_checksum` (*str*)


**Returns:** *bool*


#### check_numpy_array_integrity

```python
def check_numpy_array_integrity(
self, array: np.ndarray
)
 -> Dict[str, any]
```

Проверяет целостность numpy массива

Args:
array: Numpy массив для проверки

Returns:
Словарь с результатами проверки

**Parameters:**

- `self`
- `array` (*np.ndarray*)


**Returns:** *Dict[str, any]*


#### check_csv_integrity

```python
def check_csv_integrity(
self, file_path: str
)
 -> Dict[str, any]
```

Проверяет целостность CSV файла

Args:
file_path: Путь к CSV файлу

Returns:
Словарь с результатами проверки

**Parameters:**

- `self`
- `file_path` (*str*)


**Returns:** *Dict[str, any]*


#### check_json_integrity

```python
def check_json_integrity(
self, file_path: str
)
 -> Dict[str, any]
```

Проверяет целостность JSON файла

Args:
file_path: Путь к JSON файлу

Returns:
Словарь с результатами проверки

**Parameters:**

- `self`
- `file_path` (*str*)


**Returns:** *Dict[str, any]*


#### generate_data_fingerprint

```python
def generate_data_fingerprint(
self, data: any
)
 -> str
```

Генерирует уникальный отпечаток данных

Args:
data: Данные для генерации отпечатка

Returns:
Строка отпечатка

**Parameters:**

- `self`
- `data` (*any*)


**Returns:** *str*


#### create_data_manifest

```python
def create_data_manifest(
self = True, directory: str, recursive: bool
)
 -> Dict[str, any]
```

Создает манифест данных для директории

Args:
directory: Директория для сканирования
recursive: Рекурсивно сканировать поддиректории

Returns:
Словарь с манифестом данных

**Parameters:**

- `self`
- `directory` (*str*)
- `recursive` (*bool*)


**Returns:** *Dict[str, any]*


#### verify_data_manifest

```python
def verify_data_manifest(
self, manifest: Dict[str, any]
)
 -> Dict[str, any]
```

Проверяет манифест данных

Args:
manifest: Манифест данных для проверки

Returns:
Словарь с результатами проверки

**Parameters:**

- `self`
- `manifest` (*Dict[str, any]*)


**Returns:** *Dict[str, any]*


#### check_simulation_data_integrity

```python
def check_simulation_data_integrity(
self, data_dict: Dict[str, any]
)
 -> Dict[str, any]
```

Проверяет целостность данных симуляции

Args:
data_dict: Словарь с данными симуляции

Returns:
Словарь с результатами проверки

**Parameters:**

- `self`
- `data_dict` (*Dict[str, any]*)


**Returns:** *Dict[str, any]*


#### __init__

```python
def __init__(
self
)
```

Инициализирует генератор отчетов о целостности

**Parameters:**

- `self`


#### generate_integrity_report

```python
def generate_integrity_report(
self = None, check_results: Dict[str, any], output_path: str
)
 -> str
```

Генерирует отчет о целостности данных

Args:
check_results: Результаты проверки целостности
output_path: Путь для сохранения отчета (если None, генерируется автоматически)

Returns:
Путь к созданному отчету

**Parameters:**

- `self`
- `check_results` (*Dict[str, any]*)
- `output_path` (*str*)


**Returns:** *str*


#### _generate_summary

```python
def _generate_summary(
self, check_results: Dict[str, any]
)
 -> Dict[str, any]
```

Генерирует сводку по результатам проверки

Args:
check_results: Результаты проверки целостности

Returns:
Словарь со сводкой

**Parameters:**

- `self`
- `check_results` (*Dict[str, any]*)


**Returns:** *Dict[str, any]*




## utils\data_manager.py

### Module: data_manager

Модуль управления данными для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет централизованное управление данными 
для всех компонентов проекта.

### Classes

#### DataManager

```python
class DataManager
```

Класс для управления данными проекта
Обеспечивает централизованное хранение, загрузку и сохранение 
данных для всех компонентов проекта.

### Functions

#### main

```python
def main(

)
```

Главная функция для демонстрации работы менеджера данных

#### __init__

```python
def __init__(
self = 'data', data_dir: str = 'output', output_dir: str
)
```

Инициализирует менеджер данных

Args:
data_dir: Директория для входных данных
output_dir: Директория для выходных данных

**Parameters:**

- `self`
- `data_dir` (*str*)
- `output_dir` (*str*)


#### save_surface_data

```python
def save_surface_data(
self, surface_data: np.ndarray, filename: str
)
 -> bool
```

Сохраняет данные поверхности

Args:
surface_data: Данные поверхности в виде numpy массива
filename: Имя файла для сохранения

Returns:
bool: True если успешно сохранено, иначе False

**Parameters:**

- `self`
- `surface_data` (*np.ndarray*)
- `filename` (*str*)


**Returns:** *bool*


#### load_surface_data

```python
def load_surface_data(
self, filename: str
)
 -> Optional[np.ndarray]
```

Загружает данные поверхности

Args:
filename: Имя файла для загрузки

Returns:
Numpy массив с данными поверхности или None при ошибке

**Parameters:**

- `self`
- `filename` (*str*)


**Returns:** *Optional[np.ndarray]*


#### save_scan_results

```python
def save_scan_results(
self, scan_data: np.ndarray, filename: str
)
 -> bool
```

Сохраняет результаты сканирования

Args:
scan_data: Данные сканирования в виде numpy массива
filename: Имя файла для сохранения

Returns:
bool: True если успешно сохранено, иначе False

**Parameters:**

- `self`
- `scan_data` (*np.ndarray*)
- `filename` (*str*)


**Returns:** *bool*


#### load_scan_results

```python
def load_scan_results(
self, filename: str
)
 -> Optional[np.ndarray]
```

Загружает результаты сканирования

Args:
filename: Имя файла для загрузки

Returns:
Numpy массив с результатами сканирования или None при ошибке

**Parameters:**

- `self`
- `filename` (*str*)


**Returns:** *Optional[np.ndarray]*


#### save_image_analysis_results

```python
def save_image_analysis_results(
self, results: Dict[str, Any], filename: str
)
 -> bool
```

Сохраняет результаты анализа изображений

Args:
results: Словарь с результатами анализа
filename: Имя файла для сохранения

Returns:
bool: True если успешно сохранено, иначе False

**Parameters:**

- `self`
- `results` (*Dict[str, Any]*)
- `filename` (*str*)


**Returns:** *bool*


#### load_image_analysis_results

```python
def load_image_analysis_results(
self, filename: str
)
 -> Optional[Dict[str, Any]]
```

Загружает результаты анализа изображений

Args:
filename: Имя файла для загрузки

Returns:
Словарь с результатами анализа или None при ошибке

**Parameters:**

- `self`
- `filename` (*str*)


**Returns:** *Optional[Dict[str, Any]]*


#### save_sstv_results

```python
def save_sstv_results(
self, image_data, filename: str
)
 -> bool
```

Сохраняет результаты SSTV декодирования

Args:
image_data: Данные изображения
filename: Имя файла для сохранения

Returns:
bool: True если успешно сохранено, иначе False

**Parameters:**

- `self`
- `image_data`
- `filename` (*str*)


**Returns:** *bool*


#### save_simulation_metadata

```python
def save_simulation_metadata(
self = 'simulation_metadata.json', metadata: Dict[str, Any], filename: str
)
 -> bool
```

Сохраняет метаданные симуляции

Args:
metadata: Словарь с метаданными
filename: Имя файла для сохранения

Returns:
bool: True если успешно сохранено, иначе False

**Parameters:**

- `self`
- `metadata` (*Dict[str, Any]*)
- `filename` (*str*)


**Returns:** *bool*


#### load_simulation_metadata

```python
def load_simulation_metadata(
self = 'simulation_metadata.json', filename: str
)
 -> Optional[Dict[str, Any]]
```

Загружает метаданные симуляции

Args:
filename: Имя файла для загрузки

Returns:
Словарь с метаданными или None при ошибке

**Parameters:**

- `self`
- `filename` (*str*)


**Returns:** *Optional[Dict[str, Any]]*


#### export_to_csv

```python
def export_to_csv(
self, data: Union[np.ndarray, pd.DataFrame], filename: str
)
 -> bool
```

Экспортирует данные в CSV формат

Args:
data: Данные для экспорта
filename: Имя файла для экспорта

Returns:
bool: True если успешно экспортировано, иначе False

**Parameters:**

- `self`
- `data` (*Union[np.ndarray, pd.DataFrame]*)
- `filename` (*str*)


**Returns:** *bool*


#### get_recent_files

```python
def get_recent_files(
self = '', extension: str = 5, count: int
)
 -> List[Path]
```

Получает список последних файлов с заданным расширением

Args:
extension: Расширение файлов (например, '.txt', '.csv')
count: Количество файлов для возврата

Returns:
Список путей к файлам

**Parameters:**

- `self`
- `extension` (*str*)
- `count` (*int*)


**Returns:** *List[Path]*


#### cleanup_old_files

```python
def cleanup_old_files(
self = 30, days_old: int
)
 -> int
```

Удаляет старые файлы из директорий данных

Args:
days_old: Файлы старше этого количества дней будут удалены

Returns:
Количество удаленных файлов

**Parameters:**

- `self`
- `days_old` (*int*)


**Returns:** *int*




## utils\data_validator.py

### Module: data_validator

Модуль валидации данных для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет инструменты для проверки, 
валидации и обеспечения качества данных проекта.

### Classes

#### ValidationLevel

```python
class ValidationLevel(Enum)
```

Уровни валидации

#### ValidationResult

`@dataclass`

```python
class ValidationResult
```

Результат валидации

#### DataValidator

```python
class DataValidator
```

Класс валидатора данных
Обеспечивает проверку, валидацию и 
обеспечение качества данных проекта.

### Functions

#### validate_data

```python
def validate_data(
validation_level: ValidationLevel = ValidationLevel.STANDARD
)
```

Декоратор для валидации данных

Args:
validation_level: Уровень строгости валидации

**Parameters:**

- `validation_level` (*ValidationLevel*)


#### main

```python
def main(

)
```

Главная функция для демонстрации возможностей валидатора данных

#### __init__

```python
def __init__(
self = ValidationLevel.STANDARD, validation_level: ValidationLevel
)
```

Инициализирует валидатор данных

Args:
validation_level: Уровень строгости валидации

**Parameters:**

- `self`
- `validation_level` (*ValidationLevel*)


#### add_validation_rule

```python
def add_validation_rule(
self = None, field_name: str = False, validator_func: Callable, error_message: str, warning: bool
)
```

Добавляет правило валидации

Args:
field_name: Имя поля для валидации
validator_func: Функция валидации
error_message: Сообщение об ошибке
warning: Является ли предупреждением вместо ошибки

**Parameters:**

- `self`
- `field_name` (*str*)
- `validator_func` (*Callable*)
- `error_message` (*str*)
- `warning` (*bool*)


#### validate_numeric_field

```python
def validate_numeric_field(
self = None, value: Any = None, min_val: float = True, max_val: float, allow_nan: bool
)
 -> bool
```

Валидирует числовое поле

Args:
value: Значение для валидации
min_val: Минимальное значение
max_val: Максимальное значение
allow_nan: Разрешать ли NaN значения

Returns:
True если значение валидно, иначе False

**Parameters:**

- `self`
- `value` (*Any*)
- `min_val` (*float*)
- `max_val` (*float*)
- `allow_nan` (*bool*)


**Returns:** *bool*


#### validate_string_field

```python
def validate_string_field(
self = 1, value: Any = None, min_length: int = None, max_length: int = None, pattern: str, allowed_values: List[str]
)
 -> bool
```

Валидирует строковое поле

Args:
value: Значение для валидации
min_length: Минимальная длина
max_length: Максимальная длина
pattern: Регулярное выражение для проверки
allowed_values: Список разрешенных значений

Returns:
True если значение валидно, иначе False

**Parameters:**

- `self`
- `value` (*Any*)
- `min_length` (*int*)
- `max_length` (*int*)
- `pattern` (*str*)
- `allowed_values` (*List[str]*)


**Returns:** *bool*


#### validate_array_field

```python
def validate_array_field(
self = 0, arr: Any = None, min_length: int = None, max_length: int = True, element_validator: Callable, allow_empty: bool
)
 -> bool
```

Валидирует массив

Args:
arr: Массив для валидации
min_length: Минимальная длина массива
max_length: Максимальная длина массива
element_validator: Валидатор элементов массива
allow_empty: Разрешать ли пустые массивы

Returns:
True если массив валидный, иначе False

**Parameters:**

- `self`
- `arr` (*Any*)
- `min_length` (*int*)
- `max_length` (*int*)
- `element_validator` (*Callable*)
- `allow_empty` (*bool*)


**Returns:** *bool*


#### validate_dataframe

```python
def validate_dataframe(
self, df: pd.DataFrame, schema: Dict[str, Dict[str, Any]]
)
 -> ValidationResult
```

Валидирует DataFrame согласно схеме

Args:
df: DataFrame для валидации
schema: Схема валидации

Returns:
Результат валидации

**Parameters:**

- `self`
- `df` (*pd.DataFrame*)
- `schema` (*Dict[str, Dict[str, Any]]*)


**Returns:** *ValidationResult*


#### validate_numpy_array

```python
def validate_numpy_array(
self = None, arr: np.ndarray = None, shape: tuple = None, dtype: str = None, min_val: float = True, max_val: float, allow_nan: bool
)
 -> ValidationResult
```

Валидирует numpy массив

Args:
arr: Массив для валидации
shape: Ожидаемая форма массива
dtype: Ожидаемый тип данных
min_val: Минимальное значение
max_val: Максимальное значение
allow_nan: Разрешать ли NaN значения

Returns:
Результат валидации

**Parameters:**

- `self`
- `arr` (*np.ndarray*)
- `shape` (*tuple*)
- `dtype` (*str*)
- `min_val` (*float*)
- `max_val` (*float*)
- `allow_nan` (*bool*)


**Returns:** *ValidationResult*


#### calculate_data_quality_score

```python
def calculate_data_quality_score(
self, data: Union[pd.DataFrame, np.ndarray, Dict]
)
 -> Dict[str, float]
```

Рассчитывает оценку качества данных

Args:
data: Данные для оценки

Returns:
Словарь с метриками качества данных

**Parameters:**

- `self`
- `data` (*Union[pd.DataFrame, np.ndarray, Dict]*)


**Returns:** *Dict[str, float]*


#### generate_data_report

```python
def generate_data_report(
self = None, data: Union[pd.DataFrame, np.ndarray], output_path: str
)
 -> str
```

Генерирует отчет о данных

Args:
data: Данные для анализа
output_path: Путь для сохранения отчета (если None, генерируется автоматически)

Returns:
Путь к созданному отчету

**Parameters:**

- `self`
- `data` (*Union[pd.DataFrame, np.ndarray]*)
- `output_path` (*str*)


**Returns:** *str*


#### validate_file_integrity

```python
def validate_file_integrity(
self = None, file_path: str, expected_hash: str
)
 -> ValidationResult
```

Проверяет целостность файла

Args:
file_path: Путь к файлу
expected_hash: Ожидаемый хеш файла

Returns:
Результат валидации

**Parameters:**

- `self`
- `file_path` (*str*)
- `expected_hash` (*str*)


**Returns:** *ValidationResult*


#### validate_json_schema

```python
def validate_json_schema(
self, data: Dict, schema: Dict
)
 -> ValidationResult
```

Валидирует JSON данные по схеме

Args:
data: JSON данные
schema: Схема валидации

Returns:
Результат валидации

**Parameters:**

- `self`
- `data` (*Dict*)
- `schema` (*Dict*)


**Returns:** *ValidationResult*




## utils\deployment_manager.py

### Module: deployment_manager

Модуль управления развертыванием для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет инструменты для управления развертыванием,
контейнеризацией и оркестрацией проекта.

### Classes

#### DeploymentConfig

`@dataclass`

```python
class DeploymentConfig
```

Конфигурация развертывания

#### DeploymentManager

```python
class DeploymentManager
```

Класс менеджера развертывания
Обеспечивает управление развертыванием, контейнеризацией
и оркестрацией проекта.

### Functions

#### main

```python
def main(

)
```

Главная функция для демонстрации менеджера развертывания

#### __init__

```python
def __init__(
self = '.', project_root: str
)
```

Инициализирует менеджер развертывания

Args:
project_root: Корневая директория проекта

**Parameters:**

- `self`
- `project_root` (*str*)


#### _check_docker

```python
def _check_docker(
self
)
 -> bool
```

Проверяет доступность Docker

**Parameters:**

- `self`


**Returns:** *bool*


#### create_virtual_environment

```python
def create_virtual_environment(
self = 'venv', env_name: str = '3.9', python_version: str
)
 -> bool
```

Создает виртуальное окружение

Args:
env_name: Имя виртуального окружения
python_version: Версия Python

Returns:
Успешность создания

**Parameters:**

- `self`
- `env_name` (*str*)
- `python_version` (*str*)


**Returns:** *bool*


#### generate_dockerfile

```python
def generate_dockerfile(
self = None, output_path: str
)
 -> str
```

Генерирует Dockerfile для проекта

Args:
output_path: Путь для сохранения Dockerfile

Returns:
Путь к сгенерированному Dockerfile

**Parameters:**

- `self`
- `output_path` (*str*)


**Returns:** *str*


#### generate_docker_compose

```python
def generate_docker_compose(
self = None, output_path: str
)
 -> str
```

Генерирует docker-compose.yml для проекта

Args:
output_path: Путь для сохранения docker-compose.yml

Returns:
Путь к сгенерированному docker-compose.yml

**Parameters:**

- `self`
- `output_path` (*str*)


**Returns:** *str*


#### build_docker_image

```python
def build_docker_image(
self = 'nanoprobe-lab', image_name: str = 'latest', tag: str
)
 -> bool
```

Собирает Docker образ

Args:
image_name: Имя образа
tag: Тег образа

Returns:
Успешность сборки

**Parameters:**

- `self`
- `image_name` (*str*)
- `tag` (*str*)


**Returns:** *bool*


#### run_container

```python
def run_container(
self = 'nanoprobe-lab', image_name: str = 'latest', tag: str = None, ports: List[str]
)
 -> bool
```

Запускает контейнер

Args:
image_name: Имя образа
tag: Тег образа
ports: Список портов для проброса

Returns:
Успешность запуска

**Parameters:**

- `self`
- `image_name` (*str*)
- `tag` (*str*)
- `ports` (*List[str]*)


**Returns:** *bool*


#### generate_systemd_service

```python
def generate_systemd_service(
self = 'nanoprobe-lab', service_name: str = 'root', user: str = None, output_path: str
)
 -> str
```

Генерирует systemd service файл

Args:
service_name: Имя сервиса
user: Пользователь для запуска
output_path: Путь для сохранения service файла

Returns:
Путь к сгенерированному service файлу

**Parameters:**

- `self`
- `service_name` (*str*)
- `user` (*str*)
- `output_path` (*str*)


**Returns:** *str*


#### create_deployment_package

```python
def create_deployment_package(
self = None, package_name: str
)
 -> str
```

Создает пакет развертывания

Args:
package_name: Имя пакета

Returns:
Путь к созданному пакету

**Parameters:**

- `self`
- `package_name` (*str*)


**Returns:** *str*


#### generate_kubernetes_manifests

```python
def generate_kubernetes_manifests(
self = None, output_dir: str
)
 -> str
```

Генерирует манифесты Kubernetes

Args:
output_dir: Директория для сохранения манифестов

Returns:
Путь к директории с манифестами

**Parameters:**

- `self`
- `output_dir` (*str*)


**Returns:** *str*


#### get_deployment_status

```python
def get_deployment_status(
self
)
 -> Dict[str, Any]
```

Получает статус развертывания

**Parameters:**

- `self`


**Returns:** *Dict[str, Any]*




## utils\documentation_generator.py

### Module: documentation_generator

Модуль автоматической документации для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет инструменты для автоматической генерации 
документации из исходного кода и комментариев.

### Classes

#### DocItem

`@dataclass`

```python
class DocItem
```

Элемент документации

#### DocumentationGenerator

```python
class DocumentationGenerator
```

Класс генератора документации
Обеспечивает автоматическую генерацию документации 
из исходного кода проекта.

### Functions

#### main

```python
def main(

)
```

Главная функция для демонстрации генератора документации

#### __init__

```python
def __init__(
self = '.', project_root: str
)
```

Инициализирует генератор документации

Args:
project_root: Корневая директория проекта

**Parameters:**

- `self`
- `project_root` (*str*)


#### analyze_project_structure

```python
def analyze_project_structure(
self
)
 -> Dict[str, Any]
```

Анализирует структуру проекта

Returns:
Информация о структуре проекта

**Parameters:**

- `self`


**Returns:** *Dict[str, Any]*


#### extract_docstrings

```python
def extract_docstrings(
self = None, include_patterns: List[str] = None, exclude_patterns: List[str]
)
 -> List[DocItem]
```

Извлекает docstring из проекта

Args:
include_patterns: Паттерны для включения файлов
exclude_patterns: Паттерны для исключения файлов

Returns:
Список элементов документации

**Parameters:**

- `self`
- `include_patterns` (*List[str]*)
- `exclude_patterns` (*List[str]*)


**Returns:** *List[DocItem]*


#### _extract_from_file

```python
def _extract_from_file(
self, file_path: Path
)
```

Извлекает документацию из файла

**Parameters:**

- `self`
- `file_path` (*Path*)


#### _extract_class_doc

```python
def _extract_class_doc(
self, node: ast.ClassDef, file_path: Path, content: str
)
```

Извлекает документацию класса

**Parameters:**

- `self`
- `node` (*ast.ClassDef*)
- `file_path` (*Path*)
- `content` (*str*)


#### _extract_function_doc

```python
def _extract_function_doc(
self = False, node: ast.FunctionDef, file_path: Path, content: str, is_method: bool
)
```

Извлекает документацию функции/метода

**Parameters:**

- `self`
- `node` (*ast.FunctionDef*)
- `file_path` (*Path*)
- `content` (*str*)
- `is_method` (*bool*)


#### _build_function_signature

```python
def _build_function_signature(
self, node: ast.FunctionDef, is_method: bool
)
 -> str
```

Строит сигнатуру функции

**Parameters:**

- `self`
- `node` (*ast.FunctionDef*)
- `is_method` (*bool*)


**Returns:** *str*


#### generate_markdown_documentation

```python
def generate_markdown_documentation(
self = 'docs/api_reference.md', output_path: str
)
 -> str
```

Генерирует документацию в формате Markdown

Args:
output_path: Путь для сохранения документации

Returns:
Путь к сгенерированному файлу

**Parameters:**

- `self`
- `output_path` (*str*)


**Returns:** *str*


#### generate_html_documentation

```python
def generate_html_documentation(
self = 'docs/api_reference.html', output_path: str
)
 -> str
```

Генерирует документацию в формате HTML

Args:
output_path: Путь для сохранения документации

Returns:
Путь к сгенерированному файлу

**Parameters:**

- `self`
- `output_path` (*str*)


**Returns:** *str*


#### generate_json_documentation

```python
def generate_json_documentation(
self = 'docs/api_reference.json', output_path: str
)
 -> str
```

Генерирует документацию в формате JSON

Args:
output_path: Путь для сохранения документации

Returns:
Путь к сгенерированному файлу

**Parameters:**

- `self`
- `output_path` (*str*)


**Returns:** *str*


#### _format_docstring

```python
def _format_docstring(
self, docstring: str
)
 -> str
```

Форматирует docstring для отображения

**Parameters:**

- `self`
- `docstring` (*str*)


**Returns:** *str*


#### get_statistics

```python
def get_statistics(
self
)
 -> Dict[str, int]
```

Получает статистику по документации

**Parameters:**

- `self`


**Returns:** *Dict[str, int]*




## utils\error_handler.py

### Module: error_handler

Модуль обработки ошибок для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет систему обработки ошибок, 
логирования и восстановления для всего проекта.

### Classes

#### ErrorSeverity

```python
class ErrorSeverity(Enum)
```

Перечисление уровней важности ошибок

#### ErrorInfo

`@dataclass`

```python
class ErrorInfo
```

Информация об ошибке

#### ErrorHandler

```python
class ErrorHandler
```

Класс обработки ошибок
Обеспечивает централизованную обработку, 
логирование и восстановление после ошибок.

#### RecoveryManager

```python
class RecoveryManager
```

Класс менеджера восстановления
Обеспечивает восстановление после ошибок 
и управление состоянием системы.

#### SafeExecutor

```python
class SafeExecutor
```

Класс безопасного исполнителя
Обеспечивает безопасное выполнение кода 
с перехватом и обработкой исключений.

### Functions

#### main

```python
def main(

)
```

Главная функция для демонстрации возможностей обработчика ошибок

#### __init__

```python
def __init__(
self = 'error_log.json', log_file: str = 1000, max_log_size: int
)
```

Инициализирует обработчик ошибок

Args:
log_file: Файл для логирования ошибок
max_log_size: Максимальный размер лога (количество записей)

**Parameters:**

- `self`
- `log_file` (*str*)
- `max_log_size` (*int*)


#### load_error_history

```python
def load_error_history(
self
)
```

Загружает историю ошибок из файла

**Parameters:**

- `self`


#### save_error_history

```python
def save_error_history(
self
)
```

Сохраняет историю ошибок в файл

**Parameters:**

- `self`


#### log_error

```python
def log_error(
self = None, message: str = 'Unknown', exception: Exception = ErrorSeverity.ERROR, component: str = None, severity: ErrorSeverity, user_context: Optional[Dict[str, Any]]
)
 -> ErrorInfo
```

Логирует ошибку

Args:
message: Сообщение об ошибке
exception: Объект исключения (если есть)
component: Компонент, в котором произошла ошибка
severity: Уровень важности ошибки
user_context: Контекст пользователя (опционально)

Returns:
Объект информации об ошибке

**Parameters:**

- `self`
- `message` (*str*)
- `exception` (*Exception*)
- `component` (*str*)
- `severity` (*ErrorSeverity*)
- `user_context` (*Optional[Dict[str, Any]]*)


**Returns:** *ErrorInfo*


#### handle_exception

```python
def handle_exception(
self = 'Unknown', func: Callable = None, component: str = False, fallback_return: Any, suppress_exception: bool
)
 -> Callable
```

Декоратор для обработки исключений в функциях

Args:
func: Функция для оборачивания
component: Компонент, в котором происходит вызов
fallback_return: Значение по умолчанию при ошибке
suppress_exception: Подавлять ли исключение (возвращать fallback_return)

Returns:
Обернутая функция с обработкой исключений

**Parameters:**

- `self`
- `func` (*Callable*)
- `component` (*str*)
- `fallback_return` (*Any*)
- `suppress_exception` (*bool*)


**Returns:** *Callable*


#### get_recent_errors

```python
def get_recent_errors(
self = 10, count: int
)
 -> list
```

Возвращает последние ошибки

Args:
count: Количество ошибок для возврата

Returns:
Список последних ошибок

**Parameters:**

- `self`
- `count` (*int*)


**Returns:** *list*


#### get_errors_by_severity

```python
def get_errors_by_severity(
self, severity: ErrorSeverity
)
 -> list
```

Возвращает ошибки по уровню важности

Args:
severity: Уровень важности

Returns:
Список ошибок с указанным уровнем важности

**Parameters:**

- `self`
- `severity` (*ErrorSeverity*)


**Returns:** *list*


#### get_errors_by_component

```python
def get_errors_by_component(
self, component: str
)
 -> list
```

Возвращает ошибки по компоненту

Args:
component: Название компонента

Returns:
Список ошибок для указанного компонента

**Parameters:**

- `self`
- `component` (*str*)


**Returns:** *list*


#### clear_error_history

```python
def clear_error_history(
self
)
```

Очищает историю ошибок

**Parameters:**

- `self`


#### export_error_report

```python
def export_error_report(
self = None, output_path: str
)
 -> str
```

Экспортирует отчет об ошибках

Args:
output_path: Путь для сохранения отчета (если None, генерируется автоматически)

Returns:
Путь к экспортированному отчету

**Parameters:**

- `self`
- `output_path` (*str*)


**Returns:** *str*


#### __init__

```python
def __init__(
self, error_handler: ErrorHandler
)
```

Инициализирует менеджер восстановления

Args:
error_handler: Обработчик ошибок

**Parameters:**

- `self`
- `error_handler` (*ErrorHandler*)


#### create_state_backup

```python
def create_state_backup(
self, state_id: str, state_data: Any
)
```

Создает резервную копию состояния

Args:
state_id: Идентификатор состояния
state_data: Данные состояния

**Parameters:**

- `self`
- `state_id` (*str*)
- `state_data` (*Any*)


#### restore_state

```python
def restore_state(
self, state_id: str
)
 -> Optional[Any]
```

Восстанавливает состояние из резервной копии

Args:
state_id: Идентификатор состояния

Returns:
Восстановленные данные состояния или None если не найдено

**Parameters:**

- `self`
- `state_id` (*str*)


**Returns:** *Optional[Any]*


#### register_recovery_strategy

```python
def register_recovery_strategy(
self, error_type: str, strategy: Callable
)
```

Регистрирует стратегию восстановления для типа ошибки

Args:
error_type: Тип ошибки
strategy: Функция стратегии восстановления

**Parameters:**

- `self`
- `error_type` (*str*)
- `strategy` (*Callable*)


#### attempt_recovery

```python
def attempt_recovery(
self, error_info: ErrorInfo
)
 -> bool
```

Пытается восстановиться после ошибки

Args:
error_info: Информация об ошибке

Returns:
True если восстановление прошло успешно, иначе False

**Parameters:**

- `self`
- `error_info` (*ErrorInfo*)


**Returns:** *bool*


#### cleanup_old_backups

```python
def cleanup_old_backups(
self = 24, retention_hours: int
)
```

Удаляет старые резервные копии состояния

Args:
retention_hours: Время хранения в часах

**Parameters:**

- `self`
- `retention_hours` (*int*)


#### __init__

```python
def __init__(
self, error_handler: ErrorHandler
)
```

Инициализирует безопасный исполнитель

Args:
error_handler: Обработчик ошибок

**Parameters:**

- `self`
- `error_handler` (*ErrorHandler*)


#### execute_with_retry

```python
def execute_with_retry(
self = 3, func: Callable = 1.0, max_retries: int = 'SafeExecutor', retry_delay: float, component: str
)
 -> Any
```

Выполняет функцию с повторными попытками при ошибках

Args:
func: Функция для выполнения
max_retries: Максимальное количество попыток
retry_delay: Задержка между попытками в секундах
component: Компонент для логирования

Returns:
Результат выполнения функции

**Parameters:**

- `self`
- `func` (*Callable*)
- `max_retries` (*int*)
- `retry_delay` (*float*)
- `component` (*str*)


**Returns:** *Any*


#### execute_with_timeout

```python
def execute_with_timeout(
self = None, func: Callable = 'SafeExecutor', timeout: float, fallback_return: Any, component: str
)
 -> Any
```

Выполняет функцию с таймаутом

Args:
func: Функция для выполнения
timeout: Таймаут в секундах
fallback_return: Значение по умолчанию при таймауте
component: Компонент для логирования

Returns:
Результат выполнения функции или fallback_return при таймауте

**Parameters:**

- `self`
- `func` (*Callable*)
- `timeout` (*float*)
- `fallback_return` (*Any*)
- `component` (*str*)


**Returns:** *Any*




## utils\logger.py

### Module: logger

Модуль ведения логов для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет централизованную систему ведения логов 
для всех компонентов проекта.

### Classes

#### LoggerSetup

```python
class LoggerSetup
```

Класс для настройки системы ведения логов
Обеспечивает централизованную настройку логирования для всех 
компонентов проекта.

#### NanoprobeLogger

```python
class NanoprobeLogger
```

Класс для централизованного ведения логов проекта
Предоставляет удобный интерфейс для логирования событий 
в различных компонентах проекта.

### Functions

#### setup_project_logging

```python
def setup_project_logging(
config_manager = None
)
 -> NanoprobeLogger
```

Настраивает централизованное логирование для проекта

Args:
config_manager: Экземпляр менеджера конфигурации (опционально)

Returns:
Настроенный экземпляр NanoprobeLogger

**Parameters:**

- `config_manager`


**Returns:** *NanoprobeLogger*


#### main

```python
def main(

)
```

Главная функция для демонстрации работы системы логирования

#### __init__

```python
def __init__(
self = 'logs', log_dir: str = 'INFO', log_level: str
)
```

Инициализирует настройщик логов

Args:
log_dir: Директория для файлов логов
log_level: Уровень логирования

**Parameters:**

- `self`
- `log_dir` (*str*)
- `log_level` (*str*)


#### setup_logging_directory

```python
def setup_logging_directory(
self
)
```

Создает директорию для логов если она не существует

**Parameters:**

- `self`


#### create_logger

```python
def create_logger(
self = None, name: str, log_file: Optional[str]
)
 -> logging.Logger
```

Создает экземпляр логгера с заданными параметрами

Args:
name: Имя логгера
log_file: Имя файла для логирования (опционально)

Returns:
Настроенный экземпляр логгера

**Parameters:**

- `self`
- `name` (*str*)
- `log_file` (*Optional[str]*)


**Returns:** *logging.Logger*


#### __init__

```python
def __init__(
self = None, config_manager
)
```

Инициализирует логгер проекта

Args:
config_manager: Экземпляр менеджера конфигурации (опционально)

**Parameters:**

- `self`
- `config_manager`


#### get_logger

```python
def get_logger(
self, name: str
)
 -> logging.Logger
```

Получает или создает логгер с заданным именем

Args:
name: Имя логгера

Returns:
Экземпляр логгера

**Parameters:**

- `self`
- `name` (*str*)


**Returns:** *logging.Logger*


#### log_spm_event

```python
def log_spm_event(
self = 'INFO', message: str, level: str
)
```

Логирует событие связанное с СЗМ симулятором

Args:
message: Сообщение для логирования
level: Уровень логирования

**Parameters:**

- `self`
- `message` (*str*)
- `level` (*str*)


#### log_analyzer_event

```python
def log_analyzer_event(
self = 'INFO', message: str, level: str
)
```

Логирует событие связанное с анализатором изображений

Args:
message: Сообщение для логирования
level: Уровень логирования

**Parameters:**

- `self`
- `message` (*str*)
- `level` (*str*)


#### log_sstv_event

```python
def log_sstv_event(
self = 'INFO', message: str, level: str
)
```

Логирует событие связанное с SSTV станцией

Args:
message: Сообщение для логирования
level: Уровень логирования

**Parameters:**

- `self`
- `message` (*str*)
- `level` (*str*)


#### log_system_event

```python
def log_system_event(
self = 'INFO', message: str, level: str
)
```

Логирует системное событие проекта

Args:
message: Сообщение для логирования
level: Уровень логирования

**Parameters:**

- `self`
- `message` (*str*)
- `level` (*str*)


#### log_simulation_event

```python
def log_simulation_event(
self = 'INFO', message: str, level: str
)
```

Логирует событие связанное с симуляцией

Args:
message: Сообщение для логирования
level: Уровень логирования

**Parameters:**

- `self`
- `message` (*str*)
- `level` (*str*)




## utils\machine_learning.py

### Module: machine_learning

Модуль машинного обучения для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет инструменты для построения 
предсказательных моделей на основе данных симуляции.

### Classes

#### SurfacePredictionModel

```python
class SurfacePredictionModel
```

Класс для построения предсказательных моделей для данных поверхности
Обеспечивает обучение моделей для предсказания свойств поверхности 
на основе параметров симуляции.

#### ImageAnalysisPredictor

```python
class ImageAnalysisPredictor
```

Класс для предсказания характеристик изображений
Обучает модели для предсказания качества изображений 
и обнаруженных паттернов.

#### SSTVPredictor

```python
class SSTVPredictor
```

Класс для предсказания качества SSTV декодирования
Обучает модели для предсказания качества декодирования 
на основе характеристик сигнала.

#### ProjectMLPipeline

```python
class ProjectMLPipeline
```

Центральный класс ML пайплайна проекта
Объединяет все ML компоненты и предоставляет 
единый интерфейс для машинного обучения.

### Functions

#### main

```python
def main(

)
```

Главная функция для демонстрации возможностей ML модуля

#### __init__

```python
def __init__(
self
)
```

Инициализирует модель предсказания поверхности

**Parameters:**

- `self`


#### prepare_features

```python
def prepare_features(
self, surface_data: np.ndarray
)
 -> np.ndarray
```

Подготавливает признаки из данных поверхности

Args:
surface_data: Данные поверхности в виде numpy массива

Returns:
Массив признаков

**Parameters:**

- `self`
- `surface_data` (*np.ndarray*)


**Returns:** *np.ndarray*


#### train_regression_model

```python
def train_regression_model(
self, X: np.ndarray, y: np.ndarray
)
 -> Dict[str, float]
```

Обучает регрессионную модель

Args:
X: Признаки (матрица объекты-признаки)
y: Целевые значения (вектор)

Returns:
Словарь с метриками качества модели

**Parameters:**

- `self`
- `X` (*np.ndarray*)
- `y` (*np.ndarray*)


**Returns:** *Dict[str, float]*


#### train_classification_model

```python
def train_classification_model(
self, X: np.ndarray, y_labels: np.ndarray
)
 -> Dict[str, Any]
```

Обучает классификационную модель

Args:
X: Признаки (матрица объекты-признаки)
y_labels: Целевые метки (вектор)

Returns:
Словарь с метриками качества модели

**Parameters:**

- `self`
- `X` (*np.ndarray*)
- `y_labels` (*np.ndarray*)


**Returns:** *Dict[str, Any]*


#### predict

```python
def predict(
self = 'regression', surface_data: np.ndarray, task_type: str
)
 -> np.ndarray
```

Делает предсказание для новых данных поверхности

Args:
surface_data: Новые данные поверхности
task_type: Тип задачи ('regression' или 'classification')

Returns:
Предсказанные значения

**Parameters:**

- `self`
- `surface_data` (*np.ndarray*)
- `task_type` (*str*)


**Returns:** *np.ndarray*


#### save_model

```python
def save_model(
self, filepath: str
)
```

Сохраняет обученную модель

Args:
filepath: Путь для сохранения модели

**Parameters:**

- `self`
- `filepath` (*str*)


#### load_model

```python
def load_model(
self, filepath: str
)
```

Загружает обученную модель

Args:
filepath: Путь для загрузки модели

**Parameters:**

- `self`
- `filepath` (*str*)


#### __init__

```python
def __init__(
self
)
```

Инициализирует предиктор анализа изображений

**Parameters:**

- `self`


#### prepare_image_features

```python
def prepare_image_features(
self, image_data: np.ndarray
)
 -> np.ndarray
```

Подготавливает признаки из данных изображения

Args:
image_data: Данные изображения в виде numpy массива

Returns:
Массив признаков

**Parameters:**

- `self`
- `image_data` (*np.ndarray*)


**Returns:** *np.ndarray*


#### train

```python
def train(
self, X: np.ndarray, y: np.ndarray
)
 -> Dict[str, float]
```

Обучает модель на признаках и целевых значениях

Args:
X: Матрица признаков
y: Целевые значения

Returns:
Словарь с метриками качества

**Parameters:**

- `self`
- `X` (*np.ndarray*)
- `y` (*np.ndarray*)


**Returns:** *Dict[str, float]*


#### predict_quality_score

```python
def predict_quality_score(
self, image_data: np.ndarray
)
 -> float
```

Предсказывает оценку качества изображения

Args:
image_data: Данные изображения

Returns:
Предсказанная оценка качества (0-1)

**Parameters:**

- `self`
- `image_data` (*np.ndarray*)


**Returns:** *float*


#### __init__

```python
def __init__(
self
)
```

Инициализирует предиктор SSTV

**Parameters:**

- `self`


#### prepare_signal_features

```python
def prepare_signal_features(
self = 44100, signal_data: np.ndarray, sample_rate: int
)
 -> np.ndarray
```

Подготавливает признаки из аудиосигнала SSTV

Args:
signal_data: Данные аудиосигнала
sample_rate: Частота дискретизации

Returns:
Массив признаков

**Parameters:**

- `self`
- `signal_data` (*np.ndarray*)
- `sample_rate` (*int*)


**Returns:** *np.ndarray*


#### train_quality_model

```python
def train_quality_model(
self, X: np.ndarray, y_quality: np.ndarray
)
 -> Dict[str, float]
```

Обучает модель для предсказания качества декодирования

Args:
X: Признаки сигнала
y_quality: Целевые значения качества (0-1)

Returns:
Словарь с метриками качества

**Parameters:**

- `self`
- `X` (*np.ndarray*)
- `y_quality` (*np.ndarray*)


**Returns:** *Dict[str, float]*


#### predict_decoding_quality

```python
def predict_decoding_quality(
self, signal_data: np.ndarray
)
 -> float
```

Предсказывает качество декодирования SSTV

Args:
signal_data: Данные аудиосигнала

Returns:
Предсказанное качество декодирования (0-1)

**Parameters:**

- `self`
- `signal_data` (*np.ndarray*)


**Returns:** *float*


#### __init__

```python
def __init__(
self
)
```

Инициализирует ML пайплайн проекта

**Parameters:**

- `self`


#### train_all_models

```python
def train_all_models(
self, training_data: Dict[str, Any]
)
 -> Dict[str, Any]
```

Обучает все модели в пайплайне

Args:
training_data: Словарь с обучающими данными для каждой модели

Returns:
Словарь с метриками качества всех моделей

**Parameters:**

- `self`
- `training_data` (*Dict[str, Any]*)


**Returns:** *Dict[str, Any]*


#### make_predictions

```python
def make_predictions(
self, input_data: Dict[str, Any]
)
 -> Dict[str, Any]
```

Делает предсказания всеми моделями

Args:
input_data: Словарь с входными данными для каждой модели

Returns:
Словарь с предсказаниями всех моделей

**Parameters:**

- `self`
- `input_data` (*Dict[str, Any]*)


**Returns:** *Dict[str, Any]*


#### save_all_models

```python
def save_all_models(
self, directory: str
)
```

Сохраняет все обученные модели

Args:
directory: Директория для сохранения моделей

**Parameters:**

- `self`
- `directory` (*str*)


#### load_all_models

```python
def load_all_models(
self, directory: str
)
```

Загружает все модели

Args:
directory: Директория с сохраненными моделями

**Parameters:**

- `self`
- `directory` (*str*)




## utils\model_trainer.py

### Module: model_trainer

Модуль обучения моделей машинного обучения для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет инструменты для обучения, 
оценки и оптимизации моделей машинного обучения.

### Classes

#### ModelResult

`@dataclass`

```python
class ModelResult
```

Результат обучения модели

#### ModelTrainer

```python
class ModelTrainer
```

Класс тренера моделей
Обеспечивает обучение, оценку и 
оптимизацию моделей машинного обучения.

### Functions

#### model_training_pipeline

```python
def model_training_pipeline(
func
)
```

Декоратор для создания пайплайна обучения модели

Args:
func: Функция для декорирования

**Parameters:**

- `func`


#### main

```python
def main(

)
```

Главная функция для демонстрации возможностей тренера моделей

#### __init__

```python
def __init__(
self = 'models', output_dir: str
)
```

Инициализирует тренер моделей

Args:
output_dir: Директория для сохранения моделей

**Parameters:**

- `self`
- `output_dir` (*str*)


#### prepare_data

```python
def prepare_data(
self = 0.2, X: Union[np.ndarray, pd.DataFrame] = True, y: Union[np.ndarray, pd.Series] = True, test_size: float, scale_features: bool, encode_labels: bool
)
 -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
```

Подготавливает данные для обучения

Args:
X: Признаки
y: Целевые значения
test_size: Размер тестовой выборки
scale_features: Нормализовать ли признаки
encode_labels: Кодировать ли метки

Returns:
Кортеж (X_train, X_test, y_train, y_test)

**Parameters:**

- `self`
- `X` (*Union[np.ndarray, pd.DataFrame]*)
- `y` (*Union[np.ndarray, pd.Series]*)
- `test_size` (*float*)
- `scale_features` (*bool*)
- `encode_labels` (*bool*)


**Returns:** *Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]*


#### train_regression_model

```python
def train_regression_model(
self = 'random_forest', X_train: np.ndarray, y_train: np.ndarray, model_type: str
)
 -> Any
```

Обучает модель регрессии

Args:
X_train: Обучающие признаки
y_train: Обучающие целевые значения
model_type: Тип модели

Returns:
Обученную модель

**Parameters:**

- `self`
- `X_train` (*np.ndarray*)
- `y_train` (*np.ndarray*)
- `model_type` (*str*)


**Returns:** *Any*


#### train_classification_model

```python
def train_classification_model(
self = 'random_forest', X_train: np.ndarray, y_train: np.ndarray, model_type: str
)
 -> Any
```

Обучает модель классификации

Args:
X_train: Обучающие признаки
y_train: Обучающие целевые значения
model_type: Тип модели

Returns:
Обученную модель

**Parameters:**

- `self`
- `X_train` (*np.ndarray*)
- `y_train` (*np.ndarray*)
- `model_type` (*str*)


**Returns:** *Any*


#### evaluate_model

```python
def evaluate_model(
self = 'regression', model: Any, X_test: np.ndarray, y_test: np.ndarray, model_type: str
)
 -> Dict[str, float]
```

Оценивает модель

Args:
model: Обученная модель
X_test: Тестовые признаки
y_test: Тестовые целевые значения
model_type: Тип модели

Returns:
Словарь с метриками

**Parameters:**

- `self`
- `model` (*Any*)
- `X_test` (*np.ndarray*)
- `y_test` (*np.ndarray*)
- `model_type` (*str*)


**Returns:** *Dict[str, float]*


#### hyperparameter_tuning

```python
def hyperparameter_tuning(
self = 'random_forest', X_train: np.ndarray = 5, y_train: np.ndarray, model_type: str, cv_folds: int
)
 -> Tuple[Any, Dict[str, Any]]
```

Подбирает гиперпараметры модели

Args:
X_train: Обучающие признаки
y_train: Обучающие целевые значения
model_type: Тип модели
cv_folds: Количество фолдов для кросс-валидации

Returns:
Кортеж (лучшая модель, результаты подбора)

**Parameters:**

- `self`
- `X_train` (*np.ndarray*)
- `y_train` (*np.ndarray*)
- `model_type` (*str*)
- `cv_folds` (*int*)


**Returns:** *Tuple[Any, Dict[str, Any]]*


#### train_and_evaluate

```python
def train_and_evaluate(
self = 'random_forest', X: Union[np.ndarray, pd.DataFrame] = 'auto', y: Union[np.ndarray, pd.Series], model_type: str, problem_type: str
)
 -> ModelResult
```

Обучает и оценивает модель

Args:
X: Признаки
y: Целевые значения
model_type: Тип модели
problem_type: Тип задачи ('regression', 'classification', 'auto')

Returns:
Результат обучения модели

**Parameters:**

- `self`
- `X` (*Union[np.ndarray, pd.DataFrame]*)
- `y` (*Union[np.ndarray, pd.Series]*)
- `model_type` (*str*)
- `problem_type` (*str*)


**Returns:** *ModelResult*


#### save_model

```python
def save_model(
self = None, model: Any, model_name: str, metadata: Dict[str, Any]
)
 -> str
```

Сохраняет модель

Args:
model: Обученная модель
model_name: Имя модели
metadata: Метаданные модели

Returns:
Путь к сохраненной модели

**Parameters:**

- `self`
- `model` (*Any*)
- `model_name` (*str*)
- `metadata` (*Dict[str, Any]*)


**Returns:** *str*


#### load_model

```python
def load_model(
self, model_name: str
)
 -> Tuple[Any, Any, Optional[Dict]]
```

Загружает модель

Args:
model_name: Имя модели

Returns:
Кортеж (модель, скалер, метаданные)

**Parameters:**

- `self`
- `model_name` (*str*)


**Returns:** *Tuple[Any, Any, Optional[Dict]]*


#### cross_validate_model

```python
def cross_validate_model(
self = 'random_forest', X: np.ndarray = 5, y: np.ndarray, model_type: str, cv_folds: int
)
 -> Dict[str, Any]
```

Проводит кросс-валидацию модели

Args:
X: Признаки
y: Целевые значения
model_type: Тип модели
cv_folds: Количество фолдов

Returns:
Словарь с результатами кросс-валидации

**Parameters:**

- `self`
- `X` (*np.ndarray*)
- `y` (*np.ndarray*)
- `model_type` (*str*)
- `cv_folds` (*int*)


**Returns:** *Dict[str, Any]*


#### plot_feature_importance

```python
def plot_feature_importance(
self = None, model: Any = 10, feature_names: List[str] = None, top_n: int, output_path: str
)
 -> str
```

Строит график важности признаков

Args:
model: Обученная модель
feature_names: Названия признаков
top_n: Количество признаков для отображения
output_path: Путь для сохранения графика

Returns:
Путь к сохраненному графику

**Parameters:**

- `self`
- `model` (*Any*)
- `feature_names` (*List[str]*)
- `top_n` (*int*)
- `output_path` (*str*)


**Returns:** *str*


#### plot_predictions_vs_actual

```python
def plot_predictions_vs_actual(
self = None, y_true: np.ndarray, y_pred: np.ndarray, output_path: str
)
 -> str
```

Строит график предсказанных vs фактических значений

Args:
y_true: Фактические значения
y_pred: Предсказанные значения
output_path: Путь для сохранения графика

Returns:
Путь к сохраненному графику

**Parameters:**

- `self`
- `y_true` (*np.ndarray*)
- `y_pred` (*np.ndarray*)
- `output_path` (*str*)


**Returns:** *str*




## utils\performance_monitor.py

### Module: performance_monitor

Модуль мониторинга производительности для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет инструменты для мониторинга 
производительности и эффективности симуляций.

### Classes

#### PerformanceMonitor

```python
class PerformanceMonitor
```

Класс для мониторинга производительности
Отслеживает использование CPU, памяти, диска и других 
ресурсов во время выполнения симуляций.

#### SimulationProfiler

```python
class SimulationProfiler
```

Класс для профилирования симуляций
Обеспечивает детальное профилирование производительности 
различных этапов симуляции.

### Functions

#### main

```python
def main(

)
```

Главная функция для демонстрации возможностей монитора производительности

#### __init__

```python
def __init__(
self
)
```

Инициализирует монитор производительности

**Parameters:**

- `self`


#### start_monitoring

```python
def start_monitoring(
self = 1.0, interval: float
)
```

Запускает мониторинг ресурсов

Args:
interval: Интервал между измерениями в секундах

**Parameters:**

- `self`
- `interval` (*float*)


#### stop_monitoring

```python
def stop_monitoring(
self
)
```

Останавливает мониторинг ресурсов

**Parameters:**

- `self`


#### _monitor_loop

```python
def _monitor_loop(
self, interval: float
)
```

Основной цикл мониторинга

Args:
interval: Интервал между измерениями

**Parameters:**

- `self`
- `interval` (*float*)


#### get_current_metrics

```python
def get_current_metrics(
self
)
 -> Dict[str, float]
```

Получает текущие метрики производительности

Returns:
Словарь с текущими метриками

**Parameters:**

- `self`


**Returns:** *Dict[str, float]*


#### get_average_metrics

```python
def get_average_metrics(
self
)
 -> Dict[str, float]
```

Получает усредненные метрики за период мониторинга

Returns:
Словарь с усредненными метриками

**Parameters:**

- `self`


**Returns:** *Dict[str, float]*


#### measure_function_performance

```python
def measure_function_performance(
self, func: Callable, *args, **kwargs
)
 -> Dict[str, Any]
```

Измеряет производительность выполнения функции

Args:
func: Функция для измерения
*args: Аргументы функции
**kwargs: Ключевые аргументы функции

Returns:
Словарь с метриками производительности

**Parameters:**

- `self`
- `func` (*Callable*)
- `*args`
- `**kwargs`


**Returns:** *Dict[str, Any]*


#### visualize_performance

```python
def visualize_performance(
self = None, output_path: Optional[str]
)
```

Визуализирует метрики производительности

Args:
output_path: Путь для сохранения графика (опционально)

**Parameters:**

- `self`
- `output_path` (*Optional[str]*)


#### save_performance_report

```python
def save_performance_report(
self, output_path: str
)
```

Сохраняет отчет о производительности

Args:
output_path: Путь для сохранения отчета

**Parameters:**

- `self`
- `output_path` (*str*)


#### __init__

```python
def __init__(
self
)
```

Инициализирует профилировщик симуляций

**Parameters:**

- `self`


#### profile_simulation_stage

```python
def profile_simulation_stage(
self, stage_name: str, func: Callable, *args, **kwargs
)
 -> Dict[str, Any]
```

Профилирует отдельный этап симуляции

Args:
stage_name: Название этапа
func: Функция этапа
*args: Аргументы функции
**kwargs: Ключевые аргументы

Returns:
Словарь с результатами профилирования

**Parameters:**

- `self`
- `stage_name` (*str*)
- `func` (*Callable*)
- `*args`
- `**kwargs`


**Returns:** *Dict[str, Any]*


#### profile_full_simulation

```python
def profile_full_simulation(
self, simulation_func: Callable, *args, **kwargs
)
 -> Dict[str, Any]
```

Профилирует полную симуляцию

Args:
simulation_func: Функция симуляции
*args: Аргументы функции
**kwargs: Ключевые аргументы

Returns:
Словарь с результатами профилирования

**Parameters:**

- `self`
- `simulation_func` (*Callable*)
- `*args`
- `**kwargs`


**Returns:** *Dict[str, Any]*


#### get_optimization_recommendations

```python
def get_optimization_recommendations(
self
)
 -> List[str]
```

Получает рекомендации по оптимизации

Returns:
Список рекомендаций

**Parameters:**

- `self`


**Returns:** *List[str]*


#### generate_performance_report

```python
def generate_performance_report(
self = 'performance_report.json', output_path: str
)
```

Генерирует полный отчет о производительности

Args:
output_path: Путь для сохранения отчета

**Parameters:**

- `self`
- `output_path` (*str*)


#### test_function

```python
def test_function(

)
```

Тестовая функция для измерения производительности



## utils\profiler.py

### Module: profiler

Модуль профилирования производительности для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет инструменты для профилирования, 
анализа производительности и оптимизации кода проекта.

### Classes

#### PerformanceMetric

`@dataclass`

```python
class PerformanceMetric
```

Метрика производительности

#### Profiler

```python
class Profiler
```

Класс профилировщика
Обеспечивает профилирование, анализ 
производительности и оптимизацию кода.

### Functions

#### profile_performance

```python
def profile_performance(
func: Callable
)
 -> Callable
```

Декоратор для профилирования производительности функции

Args:
func: Функция для профилирования

Returns:
Обернутая функция с профилированием

**Parameters:**

- `func` (*Callable*)


**Returns:** *Callable*


#### benchmark_function

```python
def benchmark_function(
iterations: int = 100
)
```

Декоратор для бенчмаркинга функции

Args:
iterations: Количество итераций для бенчмаркинга

Returns:
Декоратор для бенчмаркинга

**Parameters:**

- `iterations` (*int*)


#### performance_monitor

`@contextmanager`

```python
@contextmanager
def performance_monitor(
name: str = 'Operation'
)
```

Контекстный менеджер для мониторинга производительности

Args:
name: Название операции для мониторинга

**Parameters:**

- `name` (*str*)


#### main

```python
def main(

)
```

Главная функция для демонстрации возможностей профилировщика

#### __init__

```python
def __init__(
self = 'profiles', output_dir: str
)
```

Инициализирует профилировщик

Args:
output_dir: Директория для сохранения результатов профилирования

**Parameters:**

- `self`
- `output_dir` (*str*)


#### start_cpu_monitoring

```python
def start_cpu_monitoring(
self
)
```

Запускает мониторинг CPU

**Parameters:**

- `self`


#### stop_cpu_monitoring

```python
def stop_cpu_monitoring(
self
)
```

Останавливает мониторинг CPU

**Parameters:**

- `self`


#### profile_function

```python
def profile_function(
self, func: Callable, *args, **kwargs
)
 -> Dict[str, Any]
```

Профилирует выполнение функции

Args:
func: Функция для профилирования
*args: Аргументы функции
**kwargs: Именованные аргументы функции

Returns:
Словарь с результатами профилирования

**Parameters:**

- `self`
- `func` (*Callable*)
- `*args`
- `**kwargs`


**Returns:** *Dict[str, Any]*


#### profile_memory_usage

```python
def profile_memory_usage(
self, func: Callable, *args, **kwargs
)
 -> Dict[str, Any]
```

Профилирует использование памяти функцией

Args:
func: Функция для профилирования
*args: Аргументы функции
**kwargs: Именованные аргументы функции

Returns:
Словарь с результатами профилирования памяти

**Parameters:**

- `self`
- `func` (*Callable*)
- `*args`
- `**kwargs`


**Returns:** *Dict[str, Any]*


#### profile_line_by_line

```python
def profile_line_by_line(
self, func: Callable, *args, **kwargs
)
 -> Dict[str, Any]
```

Профилирует функцию построчно

Args:
func: Функция для профилирования
*args: Аргументы функции
**kwargs: Именованные аргументы функции

Returns:
Словарь с результатами построчного профилирования

**Parameters:**

- `self`
- `func` (*Callable*)
- `*args`
- `**kwargs`


**Returns:** *Dict[str, Any]*


#### benchmark_function

```python
def benchmark_function(
self = 100, func: Callable, iterations: int, *args, **kwargs
)
 -> Dict[str, Any]
```

Бенчмаркинг функции

Args:
func: Функция для бенчмаркинга
iterations: Количество итераций
*args: Аргументы функции
**kwargs: Именованные аргументы функции

Returns:
Словарь с результатами бенчмаркинга

**Parameters:**

- `self`
- `func` (*Callable*)
- `iterations` (*int*)
- `*args`
- `**kwargs`


**Returns:** *Dict[str, Any]*


#### analyze_system_resources

```python
def analyze_system_resources(
self
)
 -> Dict[str, Any]
```

Анализирует системные ресурсы

Returns:
Словарь с информацией о системных ресурсах

**Parameters:**

- `self`


**Returns:** *Dict[str, Any]*


#### generate_performance_report

```python
def generate_performance_report(
self = None, profile_data: Dict[str, Any], output_path: str
)
 -> str
```

Генерирует отчет о производительности

Args:
profile_data: Данные профилирования
output_path: Путь для сохранения отчета (если None, генерируется автоматически)

Returns:
Путь к созданному отчету

**Parameters:**

- `self`
- `profile_data` (*Dict[str, Any]*)
- `output_path` (*str*)


**Returns:** *str*


#### _generate_summary

```python
def _generate_summary(
self, profile_data: Dict[str, Any]
)
 -> Dict[str, Any]
```

Генерирует сводку по данным профилирования

Args:
profile_data: Данные профилирования

Returns:
Сводка по профилированию

**Parameters:**

- `self`
- `profile_data` (*Dict[str, Any]*)


**Returns:** *Dict[str, Any]*


#### visualize_performance_data

```python
def visualize_performance_data(
self = None, profile_data: Dict[str, Any], output_path: str
)
 -> str
```

Визуализирует данные производительности

Args:
profile_data: Данные профилирования
output_path: Путь для сохранения графика (если None, генерируется автоматически)

Returns:
Путь к созданному графику

**Parameters:**

- `self`
- `profile_data` (*Dict[str, Any]*)
- `output_path` (*str*)


**Returns:** *str*


#### sample_function

```python
def sample_function(
n: int = 10000
)
```

Пример функции для профилирования

**Parameters:**

- `n` (*int*)




## utils\report_generator.py

### Module: report_generator

Модуль генерации отчетов для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет инструменты для создания 
комплексных отчетов о симуляциях и анализах.

### Classes

#### ReportGenerator

```python
class ReportGenerator
```

Класс для генерации отчетов
Создает комплексные отчеты о симуляциях, анализах и 
результатах работы всех компонентов проекта.

### Functions

#### main

```python
def main(

)
```

Главная функция для демонстрации возможностей генератора отчетов

#### __init__

```python
def __init__(
self = 'reports', output_dir: str
)
```

Инициализирует генератор отчетов

Args:
output_dir: Директория для сохранения отчетов

**Parameters:**

- `self`
- `output_dir` (*str*)


#### generate_simulation_report

```python
def generate_simulation_report(
self = 'Отчет о симуляции', simulation_data: Dict[str, Any] = True, title: str, include_charts: bool
)
 -> str
```

Генерирует отчет о симуляции

Args:
simulation_data: Данные симуляции
title: Заголовок отчета
include_charts: Включать ли диаграммы

Returns:
Путь к созданному отчету

**Parameters:**

- `self`
- `simulation_data` (*Dict[str, Any]*)
- `title` (*str*)
- `include_charts` (*bool*)


**Returns:** *str*


#### generate_analytics_report

```python
def generate_analytics_report(
self = 'Аналитический отчет', analytics_data: Dict[str, Any], title: str
)
 -> str
```

Генерирует аналитический отчет

Args:
analytics_data: Данные аналитики
title: Заголовок отчета

Returns:
Путь к созданному отчету

**Parameters:**

- `self`
- `analytics_data` (*Dict[str, Any]*)
- `title` (*str*)


**Returns:** *str*


#### generate_comparison_report

```python
def generate_comparison_report(
self = 'Сравнительный отчет', reports_data: List[Dict[str, Any]], title: str
)
 -> str
```

Генерирует сравнительный отчет

Args:
reports_data: Список данных для сравнения
title: Заголовок отчета

Returns:
Путь к созданному отчету

**Parameters:**

- `self`
- `reports_data` (*List[Dict[str, Any]]*)
- `title` (*str*)


**Returns:** *str*


#### create_summary_statistics

```python
def create_summary_statistics(
self, data_list: List[Dict[str, Any]]
)
 -> Dict[str, Any]
```

Создает сводную статистику из списка данных

Args:
data_list: Список словарей с данными

Returns:
Словарь с сводной статистикой

**Parameters:**

- `self`
- `data_list` (*List[Dict[str, Any]]*)


**Returns:** *Dict[str, Any]*




## utils\simulator_orchestrator.py

### Module: simulator_orchestrator

Модуль оркестратора симуляции для проекта Лаборатория моделирования нанозонда
Этот модуль координирует работу всех компонентов проекта 
для комплексной симуляции.

### Classes

#### SimulationOrchestrator

```python
class SimulationOrchestrator
```

Класс оркестратора симуляции
Координирует работу всех компонентов проекта для комплексной симуляции 
процессов, происходящих в нанозондовом микроскопе и связанных системах.

### Functions

#### main

```python
def main(

)
```

Главная функция для демонстрации возможностей оркестратора

#### __init__

```python
def __init__(
self = None, config_manager: Optional[ConfigManager]
)
```

Инициализирует оркестратор симуляции

Args:
config_manager: Экземпляр менеджера конфигурации (опционально)

**Parameters:**

- `self`
- `config_manager` (*Optional[ConfigManager]*)


#### initialize_components

```python
def initialize_components(
self
)
```

Инициализирует все компоненты проекта

**Parameters:**

- `self`


#### create_simulation_surface

```python
def create_simulation_surface(
self = (50, 50), size: tuple
)
 -> 'SurfaceModel'
```

Создает поверхность для симуляции

Args:
size: Размер поверхности (ширина, высота)

Returns:
Экземпляр модели поверхности

**Parameters:**

- `self`
- `size` (*tuple*)


**Returns:** *'SurfaceModel'*


#### run_spm_simulation

```python
def run_spm_simulation(
self = 10.0, surface: 'SurfaceModel', duration: float
)
 -> Dict[str, Any]
```

Запускает симуляцию сканирования поверхности СЗМ

Args:
surface: Модель поверхности для сканирования
duration: Продолжительность симуляции в секундах

Returns:
Словарь с результатами симуляции

**Parameters:**

- `self`
- `surface` (*'SurfaceModel'*)
- `duration` (*float*)


**Returns:** *Dict[str, Any]*


#### run_image_analysis

```python
def run_image_analysis(
self, image_path: str
)
 -> Dict[str, Any]
```

Запускает анализ изображения

Args:
image_path: Путь к изображению для анализа

Returns:
Словарь с результатами анализа

**Parameters:**

- `self`
- `image_path` (*str*)


**Returns:** *Dict[str, Any]*


#### run_sstv_decoding

```python
def run_sstv_decoding(
self, audio_file: str
)
 -> Dict[str, Any]
```

Запускает декодирование SSTV сигнала

Args:
audio_file: Путь к аудиофайлу с SSTV сигналом

Returns:
Словарь с результатами декодирования

**Parameters:**

- `self`
- `audio_file` (*str*)


**Returns:** *Dict[str, Any]*


#### coordinate_multi_component_simulation

```python
def coordinate_multi_component_simulation(
self = (50, 50), surface_size: tuple
)
 -> Dict[str, Any]
```

Координирует симуляцию с участием нескольких компонентов

Args:
surface_size: Размер поверхности для симуляции

Returns:
Словарь с результатами комплексной симуляции

**Parameters:**

- `self`
- `surface_size` (*tuple*)


**Returns:** *Dict[str, Any]*


#### run_continuous_simulation

```python
def run_continuous_simulation(
self = 10, duration_minutes: int
)
```

Запускает непрерывную симуляцию в течение заданного времени

Args:
duration_minutes: Продолжительность симуляции в минутах

**Parameters:**

- `self`
- `duration_minutes` (*int*)


#### stop_simulation

```python
def stop_simulation(
self
)
```

Останавливает текущую симуляцию

**Parameters:**

- `self`


#### start_background_simulation

```python
def start_background_simulation(
self = 10, duration_minutes: int
)
```

Запускает симуляцию в фоновом потоке

Args:
duration_minutes: Продолжительность симуляции в минутах

**Parameters:**

- `self`
- `duration_minutes` (*int*)


#### get_simulation_status

```python
def get_simulation_status(
self
)
 -> Dict[str, Any]
```

Возвращает статус текущей симуляции

Returns:
Словарь с информацией о статусе симуляции

**Parameters:**

- `self`


**Returns:** *Dict[str, Any]*




## utils\system_monitor.py

### Module: system_monitor

Модуль мониторинга системы для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет инструменты для мониторинга 
состояния системы и производительности проекта.

### Classes

#### SystemMonitor

```python
class SystemMonitor
```

Класс мониторинга системы
Обеспечивает мониторинг состояния системы, 
ресурсов и производительности проекта.

#### MonitoringDashboard

```python
class MonitoringDashboard
```

Класс интерактивной панели мониторинга
Предоставляет графический интерфейс для мониторинга 
состояния системы и производительности.

#### HealthCheckManager

```python
class HealthCheckManager
```

Класс менеджера проверки здоровья
Обеспечивает комплексную проверку состояния 
системы и компонентов проекта.

### Functions

#### main

```python
def main(

)
```

Главная функция для демонстрации возможностей мониторинга системы

#### __init__

```python
def __init__(
self = 1.0, update_interval: float
)
```

Инициализирует монитор системы

Args:
update_interval: Интервал обновления в секундах

**Parameters:**

- `self`
- `update_interval` (*float*)


#### start_monitoring

```python
def start_monitoring(
self
)
```

Запускает мониторинг системы

**Parameters:**

- `self`


#### stop_monitoring

```python
def stop_monitoring(
self
)
```

Останавливает мониторинг системы

**Parameters:**

- `self`


#### _monitor_loop

```python
def _monitor_loop(
self
)
```

Основной цикл мониторинга

**Parameters:**

- `self`


#### get_current_metrics

```python
def get_current_metrics(
self
)
 -> Dict[str, Any]
```

Получает текущие метрики системы

Returns:
Словарь с текущими метриками

**Parameters:**

- `self`


**Returns:** *Dict[str, Any]*


#### get_system_health

```python
def get_system_health(
self
)
 -> Dict[str, Any]
```

Оценивает здоровье системы

Returns:
Словарь с оценкой здоровья системы

**Parameters:**

- `self`


**Returns:** *Dict[str, Any]*


#### get_resource_usage_trend

```python
def get_resource_usage_trend(
self
)
 -> Dict[str, List[float]]
```

Получает тренды использования ресурсов

Returns:
Словарь с трендами использования ресурсов

**Parameters:**

- `self`


**Returns:** *Dict[str, List[float]]*


#### generate_report

```python
def generate_report(
self = None, output_path: str
)
 -> str
```

Генерирует отчет о состоянии системы

Args:
output_path: Путь для сохранения отчета (если None, генерируется автоматически)

Returns:
Путь к созданному отчету

**Parameters:**

- `self`
- `output_path` (*str*)


**Returns:** *str*


#### __init__

```python
def __init__(
self, system_monitor: SystemMonitor
)
```

Инициализирует панель мониторинга

Args:
system_monitor: Экземпляр SystemMonitor

**Parameters:**

- `self`
- `system_monitor` (*SystemMonitor*)


#### create_gui

```python
def create_gui(
self
)
```

Создает графический интерфейс панели мониторинга

**Parameters:**

- `self`


#### start_monitoring

```python
def start_monitoring(
self
)
```

Запускает мониторинг

**Parameters:**

- `self`


#### stop_monitoring

```python
def stop_monitoring(
self
)
```

Останавливает мониторинг

**Parameters:**

- `self`


#### animate_plots

```python
def animate_plots(
self, frame
)
```

Анимирует графики

**Parameters:**

- `self`
- `frame`


#### update_display

```python
def update_display(
self
)
```

Обновляет отображение метрик

**Parameters:**

- `self`


#### generate_report

```python
def generate_report(
self
)
```

Генерирует отчет о мониторинге

**Parameters:**

- `self`


#### run

```python
def run(
self
)
```

Запускает панель мониторинга

**Parameters:**

- `self`


#### __init__

```python
def __init__(
self, system_monitor: SystemMonitor
)
```

Инициализирует менеджер проверки здоровья

Args:
system_monitor: Экземпляр SystemMonitor

**Parameters:**

- `self`
- `system_monitor` (*SystemMonitor*)


#### run_comprehensive_health_check

```python
def run_comprehensive_health_check(
self
)
 -> Dict[str, Any]
```

Запускает комплексную проверку здоровья системы

Returns:
Словарь с результатами проверки

**Parameters:**

- `self`


**Returns:** *Dict[str, Any]*


#### _check_system_health

```python
def _check_system_health(
self
)
 -> Dict[str, Any]
```

Проверяет здоровье системы

**Parameters:**

- `self`


**Returns:** *Dict[str, Any]*


#### _check_resource_usage

```python
def _check_resource_usage(
self
)
 -> Dict[str, Any]
```

Проверяет использование ресурсов

**Parameters:**

- `self`


**Returns:** *Dict[str, Any]*


#### _check_project_directories

```python
def _check_project_directories(
self
)
 -> Dict[str, Any]
```

Проверяет директории проекта

**Parameters:**

- `self`


**Returns:** *Dict[str, Any]*


#### _check_project_configuration

```python
def _check_project_configuration(
self
)
 -> Dict[str, Any]
```

Проверяет конфигурацию проекта

**Parameters:**

- `self`


**Returns:** *Dict[str, Any]*


#### _generate_recommendations

```python
def _generate_recommendations(
self, results: Dict[str, Any]
)
 -> List[str]
```

Генерирует рекомендации на основе результатов проверки

**Parameters:**

- `self`
- `results` (*Dict[str, Any]*)


**Returns:** *List[str]*


#### generate_health_report

```python
def generate_health_report(
self = None, output_path: str
)
 -> str
```

Генерирует отчет о здоровье системы

Args:
output_path: Путь для сохранения отчета

Returns:
Путь к созданному отчету

**Parameters:**

- `self`
- `output_path` (*str*)


**Returns:** *str*




## utils\test_framework.py

### Module: test_framework

Модуль тестовой платформы для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет инструменты для комплексного 
тестирования и обеспечения качества кода проекта.

### Classes

#### TestFramework

```python
class TestFramework
```

Класс тестовой платформы
Обеспечивает комплексное тестирование, 
покрытие кода и обеспечение качества проекта.

#### QualityAssurance

```python
class QualityAssurance
```

Класс обеспечения качества
Обеспечивает статический анализ кода, 
проверку стиля и другие аспекты качества.

### Functions

#### main

```python
def main(

)
```

Главная функция для демонстрации возможностей тестовой платформы

#### __init__

```python
def __init__(
self = '.', project_root: str
)
```

Инициализирует тестовую платформу

Args:
project_root: Корневая директория проекта

**Parameters:**

- `self`
- `project_root` (*str*)


#### discover_tests

```python
def discover_tests(
self = 'tests', test_directory: str
)
 -> List[str]
```

Находит все тестовые файлы в проекте

Args:
test_directory: Директория с тестами

Returns:
Список путей к тестовым файлам

**Parameters:**

- `self`
- `test_directory` (*str*)


**Returns:** *List[str]*


#### run_unittests

```python
def run_unittests(
self = 'test_*.py', test_pattern: str
)
 -> Dict[str, Any]
```

Запускает тесты с использованием unittest

Args:
test_pattern: Паттерн для поиска тестов

Returns:
Результаты выполнения тестов

**Parameters:**

- `self`
- `test_pattern` (*str*)


**Returns:** *Dict[str, Any]*


#### run_pytest

```python
def run_pytest(
self = 'tests', test_directory: str = True, coverage_report: bool
)
 -> Dict[str, Any]
```

Запускает тесты с использованием pytest

Args:
test_directory: Директория с тестами
coverage_report: Создавать ли отчет о покрытии

Returns:
Результаты выполнения тестов

**Parameters:**

- `self`
- `test_directory` (*str*)
- `coverage_report` (*bool*)


**Returns:** *Dict[str, Any]*


#### measure_code_coverage

```python
def measure_code_coverage(
self = '.', source_directory: str = 'tests', test_directory: str
)
 -> Dict[str, Any]
```

Измеряет покрытие кода тестами

Args:
source_directory: Директория с исходным кодом
test_directory: Директория с тестами

Returns:
Результаты измерения покрытия

**Parameters:**

- `self`
- `source_directory` (*str*)
- `test_directory` (*str*)


**Returns:** *Dict[str, Any]*


#### run_performance_tests

```python
def run_performance_tests(
self = 10, test_functions: List[Callable], iterations: int
)
 -> Dict[str, Any]
```

Запускает тесты производительности

Args:
test_functions: Список функций для тестирования производительности
iterations: Количество итераций для каждого теста

Returns:
Результаты тестов производительности

**Parameters:**

- `self`
- `test_functions` (*List[Callable]*)
- `iterations` (*int*)


**Returns:** *Dict[str, Any]*


#### run_stress_tests

```python
def run_stress_tests(
self = 60, target_function: Callable = 10, duration: int, concurrency: int
)
 -> Dict[str, Any]
```

Запускает стресс-тесты

Args:
target_function: Функция для тестирования
duration: Продолжительность теста в секундах
concurrency: Количество одновременных вызовов

Returns:
Результаты стресс-теста

**Parameters:**

- `self`
- `target_function` (*Callable*)
- `duration` (*int*)
- `concurrency` (*int*)


**Returns:** *Dict[str, Any]*


#### run_integration_tests

```python
def run_integration_tests(
self
)
 -> Dict[str, Any]
```

Запускает интеграционные тесты

Returns:
Результаты интеграционных тестов

**Parameters:**

- `self`


**Returns:** *Dict[str, Any]*


#### _test_spm_component

```python
def _test_spm_component(
self
)
 -> Dict[str, Any]
```

Тестирует компонент СЗМ

**Parameters:**

- `self`


**Returns:** *Dict[str, Any]*


#### _test_image_analyzer_component

```python
def _test_image_analyzer_component(
self
)
 -> Dict[str, Any]
```

Тестирует компонент анализатора изображений

**Parameters:**

- `self`


**Returns:** *Dict[str, Any]*


#### _test_sstv_component

```python
def _test_sstv_component(
self
)
 -> Dict[str, Any]
```

Тестирует компонент SSTV

**Parameters:**

- `self`


**Returns:** *Dict[str, Any]*


#### _test_data_exchange

```python
def _test_data_exchange(
self
)
 -> Dict[str, Any]
```

Тестирует обмен данными между компонентами

**Parameters:**

- `self`


**Returns:** *Dict[str, Any]*


#### _test_api_integration

```python
def _test_api_integration(
self
)
 -> Dict[str, Any]
```

Тестирует интеграцию через API

**Parameters:**

- `self`


**Returns:** *Dict[str, Any]*


#### generate_test_report

```python
def generate_test_report(
self = None, output_path: str
)
 -> str
```

Генерирует отчет о тестировании

Args:
output_path: Путь для сохранения отчета (если None, генерируется автоматически)

Returns:
Путь к созданному отчету

**Parameters:**

- `self`
- `output_path` (*str*)


**Returns:** *str*


#### _generate_test_summary

```python
def _generate_test_summary(
self
)
 -> Dict[str, Any]
```

Генерирует сводку по результатам тестирования

Returns:
Сводка по результатам тестирования

**Parameters:**

- `self`


**Returns:** *Dict[str, Any]*


#### run_continuous_integration_pipeline

```python
def run_continuous_integration_pipeline(
self
)
 -> Dict[str, Any]
```

Запускает полный CI/CD пайплайн

Returns:
Результаты выполнения CI/CD пайплайна

**Parameters:**

- `self`


**Returns:** *Dict[str, Any]*


#### __init__

```python
def __init__(
self = '.', project_root: str
)
```

Инициализирует систему обеспечения качества

Args:
project_root: Корневая директория проекта

**Parameters:**

- `self`
- `project_root` (*str*)


#### run_pylint_analysis

```python
def run_pylint_analysis(
self
)
 -> Dict[str, Any]
```

Запускает анализ кода с помощью pylint

Returns:
Результаты анализа pylint

**Parameters:**

- `self`


**Returns:** *Dict[str, Any]*


#### run_flake8_analysis

```python
def run_flake8_analysis(
self
)
 -> Dict[str, Any]
```

Запускает анализ кода с помощью flake8

Returns:
Результаты анализа flake8

**Parameters:**

- `self`


**Returns:** *Dict[str, Any]*


#### run_black_formatter_check

```python
def run_black_formatter_check(
self
)
 -> Dict[str, Any]
```

Проверяет форматирование кода с помощью black

Returns:
Результаты проверки форматирования

**Parameters:**

- `self`


**Returns:** *Dict[str, Any]*


#### run_mypy_analysis

```python
def run_mypy_analysis(
self
)
 -> Dict[str, Any]
```

Запускает статический анализ типов с помощью mypy

Returns:
Результаты анализа mypy

**Parameters:**

- `self`


**Returns:** *Dict[str, Any]*


#### generate_quality_report

```python
def generate_quality_report(
self = None, output_path: str
)
 -> str
```

Генерирует отчет о качестве кода

Args:
output_path: Путь для сохранения отчета (если None, генерируется автоматически)

Returns:
Путь к созданному отчету

**Parameters:**

- `self`
- `output_path` (*str*)


**Returns:** *str*


#### _generate_quality_summary

```python
def _generate_quality_summary(
self
)
 -> Dict[str, Any]
```

Генерирует сводку по качеству кода

Returns:
Сводка по качеству кода

**Parameters:**

- `self`


**Returns:** *Dict[str, Any]*


#### _calculate_quality_score

```python
def _calculate_quality_score(
self, pylint_result: Dict, flake8_result: Dict, black_result: Dict, mypy_result: Dict
)
 -> float
```

Рассчитывает общий балл качества кода

Args:
pylint_result: Результаты pylint анализа
flake8_result: Результаты flake8 анализа
black_result: Результаты black проверки
mypy_result: Результаты mypy анализа

Returns:
Общий балл качества (0-100)

**Parameters:**

- `self`
- `pylint_result` (*Dict*)
- `flake8_result` (*Dict*)
- `black_result` (*Dict*)
- `mypy_result` (*Dict*)


**Returns:** *float*




## utils\visualizer.py

### Module: visualizer

Модуль визуализации для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет средства для визуализации данных 
из всех компонентов проекта.

### Classes

#### SurfaceVisualizer

```python
class SurfaceVisualizer
```

Класс для визуализации поверхностей
Обеспечивает 2D и 3D визуализацию данных поверхности 
из симулятора СЗМ.

#### ImageAnalyzerVisualizer

```python
class ImageAnalyzerVisualizer
```

Класс для визуализации результатов анализа изображений
Обеспечивает визуализацию результатов фильтрации и анализа 
изображений поверхности.

#### SSTVVisualizer

```python
class SSTVVisualizer
```

Класс для визуализации результатов SSTV
Обеспечивает визуализацию декодированных изображений и 
анализ сигналов SSTV.

#### ProjectVisualizer

```python
class ProjectVisualizer
```

Центральный класс визуализации проекта
Объединяет все визуализаторы и предоставляет единый интерфейс 
для визуализации данных из всех компонентов проекта.

### Functions

#### main

```python
def main(

)
```

Главная функция для демонстрации возможностей визуализатора

#### __init__

```python
def __init__(
self = (12, 8), figsize: Tuple[int, int]
)
```

Инициализирует визуализатор поверхности

Args:
figsize: Размер фигуры для отображения

**Parameters:**

- `self`
- `figsize` (*Tuple[int, int]*)


#### plot_surface_2d

```python
def plot_surface_2d(
self = 'Поверхность 2D', surface_data: np.ndarray = None, title: str, save_path: Optional[str]
)
 -> plt.Figure
```

Создает 2D визуализацию поверхности

Args:
surface_data: Данные поверхности в виде numpy массива
title: Заголовок графика
save_path: Путь для сохранения изображения (опционально)

Returns:
Объект matplotlib Figure

**Parameters:**

- `self`
- `surface_data` (*np.ndarray*)
- `title` (*str*)
- `save_path` (*Optional[str]*)


**Returns:** *plt.Figure*


#### plot_surface_3d

```python
def plot_surface_3d(
self = 'Поверхность 3D', surface_data: np.ndarray = None, title: str, save_path: Optional[str]
)
 -> plt.Figure
```

Создает 3D визуализацию поверхности

Args:
surface_data: Данные поверхности в виде numpy массива
title: Заголовок графика
save_path: Путь для сохранения изображения (опционально)

Returns:
Объект matplotlib Figure

**Parameters:**

- `self`
- `surface_data` (*np.ndarray*)
- `title` (*str*)
- `save_path` (*Optional[str]*)


**Returns:** *plt.Figure*


#### animate_scan_process

```python
def animate_scan_process(
self = 'Процесс сканирования', scan_data_list: list = None, title: str, save_path: Optional[str]
)
 -> animation.FuncAnimation
```

Создает анимацию процесса сканирования

Args:
scan_data_list: Список данных сканирования на разных этапах
title: Заголовок анимации
save_path: Путь для сохранения анимации (опционально)

Returns:
Объект matplotlib Animation

**Parameters:**

- `self`
- `scan_data_list` (*list*)
- `title` (*str*)
- `save_path` (*Optional[str]*)


**Returns:** *animation.FuncAnimation*


#### __init__

```python
def __init__(
self = (12, 8), figsize: Tuple[int, int]
)
```

Инициализирует визуализатор анализа изображений

Args:
figsize: Размер фигуры для отображения

**Parameters:**

- `self`
- `figsize` (*Tuple[int, int]*)


#### plot_comparison

```python
def plot_comparison(
self = 'Оригинальное изображение', original: np.ndarray = 'Обработанное изображение', processed: np.ndarray = None, title_original: str, title_processed: str, save_path: Optional[str]
)
 -> plt.Figure
```

Сравнивает оригинальное и обработанное изображения

Args:
original: Оригинальное изображение
processed: Обработанное изображение
title_original: Заголовок для оригинального изображения
title_processed: Заголовок для обработанного изображения
save_path: Путь для сохранения изображения (опционально)

Returns:
Объект matplotlib Figure

**Parameters:**

- `self`
- `original` (*np.ndarray*)
- `processed` (*np.ndarray*)
- `title_original` (*str*)
- `title_processed` (*str*)
- `save_path` (*Optional[str]*)


**Returns:** *plt.Figure*


#### plot_histograms

```python
def plot_histograms(
self = 'Гистограммы изображений', original: np.ndarray = None, processed: np.ndarray, title: str, save_path: Optional[str]
)
 -> plt.Figure
```

Строит гистограммы для оригинального и обработанного изображений

Args:
original: Оригинальное изображение
processed: Обработанное изображение
title: Заголовок графика
save_path: Путь для сохранения изображения (опционально)

Returns:
Объект matplotlib Figure

**Parameters:**

- `self`
- `original` (*np.ndarray*)
- `processed` (*np.ndarray*)
- `title` (*str*)
- `save_path` (*Optional[str]*)


**Returns:** *plt.Figure*


#### highlight_defects

```python
def highlight_defects(
self = 'Обнаруженные дефекты', image: np.ndarray = None, defects_coords: list, title: str, save_path: Optional[str]
)
 -> plt.Figure
```

Выделяет обнаруженные дефекты на изображении

Args:
image: Исходное изображение
defects_coords: Список координат дефектов [(x1, y1), (x2, y2), ...]
title: Заголовок графика
save_path: Путь для сохранения изображения (опционально)

Returns:
Объект matplotlib Figure

**Parameters:**

- `self`
- `image` (*np.ndarray*)
- `defects_coords` (*list*)
- `title` (*str*)
- `save_path` (*Optional[str]*)


**Returns:** *plt.Figure*


#### __init__

```python
def __init__(
self = (12, 8), figsize: Tuple[int, int]
)
```

Инициализирует визуализатор SSTV

Args:
figsize: Размер фигура для отображения

**Parameters:**

- `self`
- `figsize` (*Tuple[int, int]*)


#### plot_decoded_image

```python
def plot_decoded_image(
self = 'Декодированное SSTV изображение', image_data = None, title: str, save_path: Optional[str]
)
 -> plt.Figure
```

Отображает декодированное SSTV изображение

Args:
image_data: Данные изображения (numpy array или PIL Image)
title: Заголовок графика
save_path: Путь для сохранения изображения (опционально)

Returns:
Объект matplotlib Figure

**Parameters:**

- `self`
- `image_data`
- `title` (*str*)
- `save_path` (*Optional[str]*)


**Returns:** *plt.Figure*


#### plot_signal_spectrum

```python
def plot_signal_spectrum(
self = 44100, signal: np.ndarray = 'Спектр сигнала SSTV', sample_rate: int = None, title: str, save_path: Optional[str]
)
 -> plt.Figure
```

Отображает спектр сигнала SSTV

Args:
signal: Аудиосигнал
sample_rate: Частота дискретизации
title: Заголовок графика
save_path: Путь для сохранения изображения (опционально)

Returns:
Объект matplotlib Figure

**Parameters:**

- `self`
- `signal` (*np.ndarray*)
- `sample_rate` (*int*)
- `title` (*str*)
- `save_path` (*Optional[str]*)


**Returns:** *plt.Figure*


#### __init__

```python
def __init__(
self
)
```

Инициализирует центральный визуализатор проекта

**Parameters:**

- `self`


#### visualize_all_for_report

```python
def visualize_all_for_report(
self = None, surface_data: Optional[np.ndarray] = None, original_image: Optional[np.ndarray] = None, processed_image: Optional[np.ndarray] = None, sstv_image = 'output', output_dir: str
)
 -> bool
```

Создает полный отчет визуализации для всех компонентов

Args:
surface_data: Данные поверхности
original_image: Оригинальное изображение
processed_image: Обработанное изображение
sstv_image: Декодированное SSTV изображение
output_dir: Директория для сохранения визуализаций

Returns:
bool: True если успешно создан отчет, иначе False

**Parameters:**

- `self`
- `surface_data` (*Optional[np.ndarray]*)
- `original_image` (*Optional[np.ndarray]*)
- `processed_image` (*Optional[np.ndarray]*)
- `sstv_image`
- `output_dir` (*str*)


**Returns:** *bool*



