# Руководство по разработке и CI/CD

Это руководство объясняет как настроен CI/CD в проекте и как настроить его самостоятельно для своих проектов.

## Содержание

1. [Что такое CI/CD](#что-такое-cicd)
2. [Структура GitHub Actions](#структура-github-actions)
3. [CI Pipeline (ci.yml)](#ci-pipeline-ciyml)
4. [CD Pipeline (cd.yml)](#cd-pipeline-cdyml)
5. [Release Pipeline (release.yml)](#release-pipeline-releaseyml)
6. [Как настроить для своего проекта](#как-настроить-для-своего-проекта)
7. [Секреты и переменные](#секреты-и-переменные)
8. [Полезные советы](#полезные-советы)

---

## Что такое CI/CD

**CI (Continuous Integration)** — автоматическая проверка кода при каждом изменении:
- Запуск тестов
- Проверка стиля кода (линтинг)
- Проверка типов
- Сборка проекта

**CD (Continuous Deployment)** — автоматический деплой после успешных проверок:
- Сборка Docker образа
- Публикация в реестр (Docker Hub, GitHub Container Registry)
- Деплой на сервер

---

## Структура GitHub Actions

```
.github/
└── workflows/
    ├── ci.yml      # Continuous Integration
    ├── cd.yml      # Continuous Deployment  
    └── release.yml # Автоматические релизы
```

### Основные понятия

```yaml
name: CI                    # Имя workflow (отображается в GitHub)

on:                         # Триггеры (когда запускать)
  push:
    branches: [main]        # При push в main
  pull_request:             # При создании PR
    branches: [main]

jobs:                       # Задачи (выполняются параллельно по умолчанию)
  test:
    runs-on: ubuntu-latest  # На какой машине запускать
    steps:                  # Шаги внутри задачи (выполняются последовательно)
      - name: Checkout      # Название шага
        uses: actions/checkout@v4  # Использовать готовый action
      
      - name: Run tests
        run: pytest         # Выполнить команду
```

---

## CI Pipeline (ci.yml)

### Что делает

1. **Lint** — проверяет код на ошибки и стиль
2. **Test** — запускает тесты на разных версиях Python
3. **Docker Build Test** — проверяет что Docker образ собирается
4. **Security** — проверяет зависимости на уязвимости

### Как читать файл

```yaml
jobs:
  lint:                              # Имя job
    name: Lint & Code Style          # Отображаемое имя
    runs-on: ubuntu-latest           # Машина

  test:
    needs: lint                      # Зависимость: запустить после lint
    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11"]  # Матрица: запустить для каждой версии
```

### Кэширование

Кэширование ускоряет сборку — не нужно каждый раз скачивать зависимости:

```yaml
- name: Cache pip
  uses: actions/cache@v4
  with:
    path: ~/.cache/pip                                    # Что кэшировать
    key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}  # Ключ кэша
```

**Как работает:** 
- `key` создаётся из хэша файла зависимостей
- Если `requirements.txt` изменился — новый кэш
- Если не изменился — используем старый

---

## CD Pipeline (cd.yml)

### Что делает

1. Собирает Docker образ
2. Публикует в GitHub Container Registry (ghcr.io)
3. Опционально публикует в Docker Hub

### Когда запускается

```yaml
on:
  push:
    branches: [main, master]     # При push в main
    tags: ['v*.*.*']             # При создании тега версии
  workflow_dispatch:              # Ручной запуск из UI
```

### Авторизация в реестрах

**GitHub Container Registry** — бесплатно, автоматически:

```yaml
- name: Log in to GHCR
  uses: docker/login-action@v3
  with:
    registry: ghcr.io
    username: ${{ github.actor }}           # Автоматически
    password: ${{ secrets.GITHUB_TOKEN }}   # Встроенный токен
```

**Docker Hub** — нужны секреты:

```yaml
- name: Log in to Docker Hub
  uses: docker/login-action@v3
  with:
    username: ${{ secrets.DOCKERHUB_USERNAME }}
    password: ${{ secrets.DOCKERHUB_TOKEN }}
```

### Автоматические теги

```yaml
- name: Extract metadata
  uses: docker/metadata-action@v5
  with:
    tags: |
      type=raw,value=latest,enable={{is_default_branch}}  # main -> latest
      type=semver,pattern={{version}}                      # v1.2.3 -> 1.2.3
      type=sha,prefix=sha-                                 # sha-abc123
```

---

## Release Pipeline (release.yml)

### Как создать релиз

```bash
# 1. Обнови версию в pyproject.toml
# 2. Закоммить изменения
git add .
git commit -m "Release v1.2.0"

# 3. Создай тег
git tag v1.2.0

# 4. Запуши тег
git push origin v1.2.0
```

Workflow автоматически:
- Создаст GitHub Release
- Сгенерирует changelog из коммитов
- Добавит информацию о Docker образе

---

## Как настроить для своего проекта

### Шаг 1: Создай структуру

```bash
mkdir -p .github/workflows
```

### Шаг 2: Скопируй workflows

Скопируй файлы `ci.yml`, `cd.yml`, `release.yml` в `.github/workflows/`

### Шаг 3: Адаптируй под свой проект

**ci.yml:**
```yaml
# Измени версию Python
env:
  PYTHON_VERSION: "3.11"

# Измени команды установки зависимостей
- name: Install dependencies
  run: pip install -r requirements.txt

# Измени команду тестов
- name: Run tests
  run: pytest tests/
```

**cd.yml:**
```yaml
# Измени имя образа
env:
  GHCR_IMAGE: ghcr.io/твой-username/твой-проект
```

### Шаг 4: Настрой секреты (опционально)

1. Перейди в Settings → Secrets and variables → Actions
2. Добавь секреты:
   - `DOCKERHUB_USERNAME` — логин Docker Hub
   - `DOCKERHUB_TOKEN` — токен (не пароль!) Docker Hub

### Шаг 5: Проверь

```bash
git add .github/
git commit -m "Add CI/CD workflows"
git push
```

Перейди в Actions в репозитории — должен запуститься CI.

---

## Секреты и переменные

### Встроенные переменные

| Переменная | Описание |
|------------|----------|
| `${{ github.actor }}` | Кто запустил workflow |
| `${{ github.repository }}` | owner/repo |
| `${{ github.ref }}` | refs/heads/main или refs/tags/v1.0.0 |
| `${{ github.sha }}` | SHA коммита |
| `${{ secrets.GITHUB_TOKEN }}` | Автоматический токен |

### Как добавить секрет

1. Settings → Secrets and variables → Actions
2. New repository secret
3. Введи имя (например `DOCKERHUB_TOKEN`) и значение

### Как получить Docker Hub токен

1. Войди в hub.docker.com
2. Account Settings → Security → New Access Token
3. Скопируй токен (показывается один раз!)

---

## Полезные советы

### 1. Используй кэширование

```yaml
- uses: actions/cache@v4
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
```

### 2. Используй матрицы для тестирования

```yaml
strategy:
  matrix:
    python-version: ["3.9", "3.10", "3.11"]
    os: [ubuntu-latest, macos-latest]
```

### 3. Не падай на некритичных проверках

```yaml
- name: Security check
  run: pip-audit -r requirements.txt
  continue-on-error: true  # Не падать если нашлись уязвимости
```

### 4. Используй `needs` для зависимостей

```yaml
jobs:
  lint:
    # ...
  
  test:
    needs: lint  # Запустить только после lint
  
  deploy:
    needs: [lint, test]  # После lint И test
```

### 5. Условный запуск

```yaml
# Только для main ветки
if: github.ref == 'refs/heads/main'

# Только если не PR
if: github.event_name != 'pull_request'

# Только для тегов
if: startsWith(github.ref, 'refs/tags/')
```

### 6. Выводи информацию в Summary

```yaml
- name: Summary
  run: |
    echo "## Results" >> $GITHUB_STEP_SUMMARY
    echo "- Tests passed: ✅" >> $GITHUB_STEP_SUMMARY
```

---

## Дебаг

### Посмотреть логи

1. Actions → выбери workflow run → выбери job → раскрой шаги

### Локальный запуск (act)

```bash
# Установи act
brew install act  # macOS
# или
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

# Запусти локально
act -j lint
```

### Включить дебаг логи

1. Settings → Secrets → New secret
2. Имя: `ACTIONS_STEP_DEBUG`, значение: `true`

---

## Примеры для разных проектов

### Node.js

```yaml
- uses: actions/setup-node@v4
  with:
    node-version: '20'
    cache: 'npm'

- run: npm ci
- run: npm test
```

### Go

```yaml
- uses: actions/setup-go@v5
  with:
    go-version: '1.21'

- run: go test ./...
```

### Rust

```yaml
- uses: actions-rs/toolchain@v1
  with:
    toolchain: stable

- run: cargo test
```

---

## Полезные ссылки

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Awesome Actions](https://github.com/sdras/awesome-actions)
- [GitHub Actions Marketplace](https://github.com/marketplace?type=actions)
- [act - локальный запуск](https://github.com/nektos/act)
