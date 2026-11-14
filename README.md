# Прогноз дефолта по кредитной карте

Учебный проект по полному циклу ML: проверка данных, подготовка признаков, подбор модели, отслеживание экспериментов, API и мониторинг. Используется [UCI Default of Credit Card Clients Dataset](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients). Целевая переменная `default.payment.next.month`: 1 — дефолт, 0 — нет.

## Структура

- `UCI_Credit_Card.csv` — исходные данные (хранятся в Git).
- `src/data/` — проверка CSV через Pandera, признаки и подготовка выборок.
- `src/models/` — sklearn Pipeline, GridSearchCV, оценка и MLflow.
- `src/api/app.py` — FastAPI с `POST /predict`.
- `src/monitor.py` — запросы к API и PSI для важных признаков.
- `tests/` — простые unit-тесты.
- `dvc.yaml` — стадии `prepare` и `train`.
- `.github/workflows/ci.yml` — форматирование, линтер, валидация и тесты.

## Быстрый запуск

Требуется Python 3.11 или новее. Команды выполнять из корня проекта:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
dvc repro
```

`dvc repro` проверит данные, создаст стратифицированные выборки 80/20, обучит пять вариантов моделей с трёхкратной кросс-валидацией и сохранит лучший. Исходный CSV доступен в Git, а подготовленные данные и модель пересоздаются через DVC. Если нужно отдельно запустить шаги:

```bash
python -m src.eda
python -m src.data.validation UCI_Credit_Card.csv
python -m src.data.prepare
python -m src.models.train
```

Метрики тестовой выборки лежат в `reports/metrics.json`, графики — в `reports/`, модель — в `models/model.joblib`. Эксперименты видны в интерфейсе MLflow:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Модель выбирается по ROC-AUC на кросс-валидации. Тестовая выборка используется только для итоговой оценки. Сохраняются ROC-AUC, Precision, Recall, F1, Accuracy, ROC-кривая, матрица ошибок, важность признаков и сама модель. Для категорий `SEX`, `EDUCATION`, `MARRIAGE` применяется OneHotEncoder, остальные признаки проходят импутацию медианой и масштабирование.

## API и Docker

После обучения запустите API:

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Описание полей и форма запроса доступны по адресу `http://localhost:8000/docs`. Пример запроса:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"LIMIT_BAL":20000,"SEX":2,"EDUCATION":2,"MARRIAGE":1,"AGE":24,"PAY_0":2,"PAY_2":2,"PAY_3":-1,"PAY_4":-1,"PAY_5":-2,"PAY_6":-2,"BILL_AMT1":3913,"BILL_AMT2":3102,"BILL_AMT3":689,"BILL_AMT4":0,"BILL_AMT5":0,"BILL_AMT6":0,"PAY_AMT1":0,"PAY_AMT2":689,"PAY_AMT3":0,"PAY_AMT4":0,"PAY_AMT5":0,"PAY_AMT6":0}'
```

Ответ содержит `default_prediction` (0 или 1) и `default_probability` (от 0 до 1). Для Docker сначала выполните `dvc repro`, затем:

```bash
docker build -t credit-default-api .
docker run --rm -p 8000:8000 credit-default-api
```

Или запустите `sh build_and_run.sh`. Образ содержит зависимости только для API; обучение выполняется перед сборкой.

## Проверка новых данных

При запущенном API можно отправить новые строки из CSV. Файл должен иметь те же исходные признаки, что и учебный датасет; `ID` и целевая переменная необязательны.

```bash
python -m src.monitor --input UCI_Credit_Card.csv --limit 100
```

Скрипт рассчитывает PSI для `LIMIT_BAL`, `AGE`, `PAY_0` относительно обучающей выборки. Порог 0.2 используется как простой сигнал о заметном сдвиге. Результат записывается в `reports/drift.json`. Проверка на исходных строках показывает работу скрипта, но для реального мониторинга нужен отдельный файл с новыми наблюдениями.

## Краткий анализ данных

В исходном файле 30 000 клиентов и 25 столбцов. Пропусков и повторяющихся ID нет. Дефолт в следующем месяце — у 6 636 клиентов (22,1%). Возраст от 21 до 79 лет, кредитный лимит от 10 000 до 1 000 000. Коды 0 в `EDUCATION` и `MARRIAGE` сохранены как отдельные категории, поскольку они присутствуют в источнике. Для воспроизводимости используется `random_state=42`.

## Проверки

```bash
black --check src tests
flake8 --max-line-length=88 src tests
pytest -q
```

При каждом push GitHub Actions выполняет эти проверки и валидацию исходного файла.
