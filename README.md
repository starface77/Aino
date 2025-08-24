# 🤖 AION - AI Self-Improvement System

Интеллектуальная система самоулучшения с веб-интерфейсом, API, мониторингом и тестированием.

## 🚀 Установка

```bash
# Установка зависимостей
pip install -r requirements.txt

# Запуск системы
python aion.py --help
```

## 📋 Основные команды

### Анализ и управление:
```bash
python aion.py analyze                    # Анализ проекта и GitHub
python aion.py status                     # Статус системы  
python aion.py issues                     # Найденные проблемы
python aion.py projects                   # Изученные проекты
python aion.py clean                      # Очистить логи
python aion.py fix                        # Исправить проблемы
```

### AI команды:
```bash
python aion.py ai --action analyze        # AI анализ кода
python aion.py ai --action plan           # План улучшений
python aion.py ai --action improve        # Применение улучшений
python aion.py ai --provider inference    # Использовать Gemma 27B
```

### Веб-интерфейс и API:
```bash
python aion.py web                        # Веб-интерфейс (порт 5000)
python aion.py web --host 0.0.0.0 --port 8080  # Кастомные настройки
python aion.py api                        # REST API (порт 8000)
python aion.py api --host 0.0.0.0 --port 9000  # Кастомный API
```

### Мониторинг:
```bash
python aion.py monitor                    # Статус системы
python aion.py monitor --action status    # Детальный статус
python aion.py monitor --action start     # Фоновый мониторинг
python aion.py monitor --action dashboard # Интерактивный дашборд
```

### Тестирование:
```bash
python aion.py test                       # Все тесты
python aion.py test --action specific --name core  # Конкретный тест
python aion.py test --action history      # История тестирования
python aion.py test --action clean        # Очистка тестовых данных
```

## 🌐 Веб-интерфейс

Современный веб-интерфейс с функциями:
- 📊 Дашборд с метриками в реальном времени
- 🔧 Управление анализом и исправлениями
- 💻 Встроенный терминал
- 📈 Визуализация данных
- 🤖 AI интерфейс

**Доступ:** `http://localhost:5000`

## 🚀 REST API

Полноценный REST API для интеграции:

### Основные эндпоинты:
- `GET /api/v1/health` - Проверка здоровья
- `GET /api/v1/status` - Статус системы
- `POST /api/v1/analyze` - Запуск анализа
- `POST /api/v1/ai` - AI операции
- `POST /api/v1/fix` - Исправления
- `GET /api/v1/issues` - Список проблем
- `GET /api/v1/projects` - Изученные проекты
- `GET /api/v1/logs` - Системные логи

### Примеры запросов:

```bash
# Статус системы
curl http://localhost:8000/api/v1/status

# Запуск анализа
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"async": true}'

# AI анализ
curl -X POST http://localhost:8000/api/v1/ai \
  -H "Content-Type: application/json" \
  -d '{"action": "analyze", "provider": "inference"}'
```

## 📊 Мониторинг

Система мониторинга отслеживает:
- 🖥️ CPU, память, диск
- ⚡ Метрики AION (анализы, AI запросы, ошибки)
- 📝 Лог событий
- 🏥 Общее "здоровье" системы
- ⏱️ Время работы

## 🧪 Тестирование

Комплексная система тестирования:
- **CoreFunctionalityTests** - Основной функционал
- **FileOperationsTests** - Файловые операции
- **ConfigurationTests** - Конфигурация
- **APITests** - AI функциональность
- **MonitoringTests** - Мониторинг
- **IntegrationTests** - Интеграционные тесты

## 🎯 Функции

- ✅ Анализ кода проекта
- ✅ Изучение популярных GitHub проектов  
- ✅ Автоматическое исправление проблем
- ✅ AI анализ с Gemma 27B
- ✅ Веб-интерфейс с дашбордом
- ✅ REST API для интеграции
- ✅ Система мониторинга производительности
- ✅ Комплексное тестирование
- ✅ Логирование и отчетность

## 🤖 AI Интеграция

Система использует Gemma 27B через inference.net для:
- Анализа качества кода
- Генерации планов улучшений
- Автоматического применения исправлений
- Обнаружения сложных проблем

## 🔗 Архитектура

```
AION System
├── aion.py          # Основная CLI система
├── ai_core.py       # AI провайдеры и логика
├── web_interface.py # Веб-интерфейс Flask
├── api_server.py    # REST API сервер  
├── monitoring.py    # Система мониторинга
├── testing.py       # Система тестирования
├── terminal.py      # Интерактивный терминал
└── requirements.txt # Зависимости
```

## 📦 Требования

Основные зависимости:
- `Flask` - веб-фреймворк
- `flask-cors` - CORS поддержка
- `psutil` - системный мониторинг
- `requests` - HTTP клиент

Все зависимости в `requirements.txt`

## 🚀 Быстрый старт

```bash
# 1. Установка
pip install -r requirements.txt

# 2. Анализ
python aion.py analyze

# 3. Веб-интерфейс
python aion.py web

# 4. API сервер
python aion.py api

# 5. Мониторинг
python aion.py monitor

# 6. Тесты
python aion.py test
```

**Полнофункциональная система для самоанализа и улучшения ИИ проектов!**