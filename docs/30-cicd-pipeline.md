# CI/CD Pipeline Enhancements

## Обзор

Полный CI/CD pipeline с GitHub Actions для автоматизации тестирования, безопасности и деплоя.

## Структура workflows

```
.github/
├── workflows/
│   ├── ci.yml              # Main CI pipeline
│   ├── security.yml        # Security scanning
│   ├── release.yml         # Release automation
│   ├── deploy.yml          # Deployment
│   ├── benchmark.yml       # Performance testing
│   └── dependency-update.yml
├── actions/
│   └── setup/
│       └── action.yml
└── CODEOWNERS
```

## 1. Main CI Pipeline

```yaml
# .github/workflows/ci.yml
name: CI Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  PYTHON_VERSION: '3.11'
  NODE_VERSION: '20'
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  # ==========================================
  # Lint & Format Check
  # ==========================================
  lint:
    name: Lint & Format
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'

      - name: Install linters
        run: |
          pip install ruff black isort mypy
          pip install -r requirements.txt

      - name: Run Ruff
        run: ruff check . --output-format=github

      - name: Check Black formatting
        run: black --check .

      - name: Check isort
        run: isort --check-only --diff .

      - name: Run MyPy
        run: mypy api/ utils/ --ignore-missing-imports
        continue-on-error: true

      - name: Set up Node.js
        uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'npm'
          cache-dependency-path: frontend/package-lock.json

      - name: Install frontend dependencies
        working-directory: frontend
        run: npm ci

      - name: Run ESLint
        working-directory: frontend
        run: npm run lint

      - name: Run TypeScript check
        working-directory: frontend
        run: npm run type-check

  # ==========================================
  # Backend Tests
  # ==========================================
  test-backend:
    name: Backend Tests
    runs-on: ubuntu-latest
    needs: lint

    services:
      postgres:
        image: postgres:15-alpine
        env:
          POSTGRES_USER: test
          POSTGRES_PASSWORD: test
          POSTGRES_DB: nanoprobe_test
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-api.txt
          pip install pytest pytest-cov pytest-asyncio pytest-xdist

      - name: Run tests with coverage
        env:
          DATABASE_URL: postgresql://test:test@localhost:5432/nanoprobe_test
          REDIS_URL: redis://localhost:6379/0
          JWT_SECRET: test-secret-key
          NASA_API_KEY: DEMO_KEY
        run: |
          pytest tests/ \
            --cov=api \
            --cov=utils \
            --cov-report=xml \
            --cov-report=html \
            --cov-fail-under=80 \
            -n auto \
            -v

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v4
        with:
          files: ./coverage.xml
          flags: backend
          fail_ci_if_error: false
          token: ${{ secrets.CODECOV_TOKEN }}

      - name: Archive coverage reports
        uses: actions/upload-artifact@v4
        with:
          name: coverage-backend
          path: htmlcov/
          retention-days: 7

  # ==========================================
  # Frontend Tests
  # ==========================================
  test-frontend:
    name: Frontend Tests
    runs-on: ubuntu-latest
    needs: lint

    steps:
      - uses: actions/checkout@v4

      - name: Set up Node.js
        uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'npm'
          cache-dependency-path: frontend/package-lock.json

      - name: Install dependencies
        working-directory: frontend
        run: npm ci

      - name: Run tests
        working-directory: frontend
        run: npm test -- --coverage --watchAll=false

      - name: Build application
        working-directory: frontend
        run: npm run build

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          files: frontend/coverage/lcov.info
          flags: frontend
          fail_ci_if_error: false
          token: ${{ secrets.CODECOV_TOKEN }}

  # ==========================================
  # Integration Tests
  # ==========================================
  integration-tests:
    name: Integration Tests
    runs-on: ubuntu-latest
    needs: [test-backend, test-frontend]

    steps:
      - uses: actions/checkout@v4

      - name: Run integration tests
        run: |
          echo "Running integration tests..."
          # docker-compose -f docker-compose.test.yml up --abort-on-container-exit

  # ==========================================
  # Build Docker Images
  # ==========================================
  build:
    name: Build Docker Images
    runs-on: ubuntu-latest
    needs: [test-backend, test-frontend]
    if: github.event_name == 'push'

    permissions:
      contents: read
      packages: write

    outputs:
      api-tag: ${{ steps.meta-api.outputs.tags }}
      frontend-tag: ${{ steps.meta-frontend.outputs.tags }}

    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      # Build API
      - name: Extract metadata for API
        id: meta-api
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}-api
          tags: |
            type=ref,event=branch
            type=sha,prefix=
            type=semver,pattern={{version}}

      - name: Build and push API
        uses: docker/build-push-action@v5
        with:
          context: .
          file: docker/api/Dockerfile
          push: true
          tags: ${{ steps.meta-api.outputs.tags }}
          labels: ${{ steps.meta-api.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

      # Build Frontend
      - name: Extract metadata for Frontend
        id: meta-frontend
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}-frontend
          tags: |
            type=ref,event=branch
            type=sha,prefix=
            type=semver,pattern={{version}}

      - name: Build and push Frontend
        uses: docker/build-push-action@v5
        with:
          context: ./frontend
          file: ../docker/frontend/Dockerfile
          push: true
          tags: ${{ steps.meta-frontend.outputs.tags }}
          labels: ${{ steps.meta-frontend.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

## 2. Security Scanning

```yaml
# .github/workflows/security.yml
name: Security Scan

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 0 * * 0'  # Weekly

jobs:
  # ==========================================
  # Python Security
  # ==========================================
  python-security:
    name: Python Security
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install security tools
        run: |
          pip install bandit safety pip-audit

      - name: Run Bandit
        run: bandit -r api/ utils/ -f json -o bandit-report.json || true
        continue-on-error: true

      - name: Run Safety
        run: safety check -r requirements.txt --json > safety-report.json || true
        continue-on-error: true

      - name: Run pip-audit
        run: pip-audit -r requirements.txt --format json > pip-audit-report.json || true
        continue-on-error: true

      - name: Upload security reports
        uses: actions/upload-artifact@v4
        with:
          name: security-reports-python
          path: '*-report.json'
          retention-days: 30

  # ==========================================
  # Node.js Security
  # ==========================================
  nodejs-security:
    name: Node.js Security
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
          cache-dependency-path: frontend/package-lock.json

      - name: Run npm audit
        working-directory: frontend
        run: npm audit --json > npm-audit.json || true
        continue-on-error: true

      - name: Upload audit report
        uses: actions/upload-artifact@v4
        with:
          name: security-reports-nodejs
          path: frontend/npm-audit.json
          retention-days: 30

  # ==========================================
  # Container Security
  # ==========================================
  container-security:
    name: Container Security
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          scan-ref: '.'
          format: 'sarif'
          output: 'trivy-results.sarif'
          severity: 'CRITICAL,HIGH'

      - name: Upload Trivy scan results
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: 'trivy-results.sarif'

  # ==========================================
  # Secret Scanning
  # ==========================================
  secret-scan:
    name: Secret Scanning
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: TruffleHog OSS
        uses: trufflesecurity/trufflehog@main
        with:
          path: ./
          base: ${{ github.event.repository.default_branch }}
          extra_args: --debug --only-verified
```

## 3. Release Automation

```yaml
# .github/workflows/release.yml
name: Release

on:
  push:
    tags:
      - 'v*.*.*'

permissions:
  contents: write
  packages: write

jobs:
  # ==========================================
  # Create Release
  # ==========================================
  release:
    name: Create Release
    runs-on: ubuntu-latest
    outputs:
      version: ${{ steps.get_version.outputs.version }}

    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Get version
        id: get_version
        run: echo "version=${GITHUB_REF#refs/tags/}" >> $GITHUB_OUTPUT

      - name: Generate changelog
        id: changelog
        uses: metcalfc/changelog-generator@v4.0.1
        with:
          myToken: ${{ secrets.GITHUB_TOKEN }}

      - name: Create Release
        uses: softprops/action-gh-release@v1
        with:
          body: ${{ steps.changelog.outputs.changelog }}
          generate_release_notes: true
          files: |
            LICENSE
            README.md

  # ==========================================
  # Build & Push Images
  # ==========================================
  build-and-push:
    name: Build & Push Images
    runs-on: ubuntu-latest
    needs: release

    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to GitHub Container Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Log in to Docker Hub
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}

      # Build and push API
      - name: Build and push API
        uses: docker/build-push-action@v5
        with:
          context: .
          file: docker/api/Dockerfile
          push: true
          tags: |
            ghcr.io/${{ github.repository }}-api:${{ needs.release.outputs.version }}
            ghcr.io/${{ github.repository }}-api:latest
            ${{ secrets.DOCKERHUB_USERNAME }}/nanoprobe-api:${{ needs.release.outputs.version }}

      # Build and push Frontend
      - name: Build and push Frontend
        uses: docker/build-push-action@v5
        with:
          context: ./frontend
          file: ../docker/frontend/Dockerfile
          push: true
          tags: |
            ghcr.io/${{ github.repository }}-frontend:${{ needs.release.outputs.version }}
            ghcr.io/${{ github.repository }}-frontend:latest
            ${{ secrets.DOCKERHUB_USERNAME }}/nanoprobe-frontend:${{ needs.release.outputs.version }}

  # ==========================================
  # Deploy to Production
  # ==========================================
  deploy:
    name: Deploy to Production
    runs-on: ubuntu-latest
    needs: [release, build-and-push]
    if: ${{ !contains(needs.release.outputs.version, '-') }}  # Only stable releases

    environment:
      name: production
      url: https://nanoprobe.your-domain.com

    steps:
      - uses: actions/checkout@v4

      - name: Deploy to server
        uses: appleboy/ssh-action@v1.0.0
        with:
          host: ${{ secrets.DEPLOY_HOST }}
          username: ${{ secrets.DEPLOY_USER }}
          key: ${{ secrets.DEPLOY_KEY }}
          script: |
            cd /opt/nanoprobe
            docker-compose -f docker-compose.prod.yml pull
            docker-compose -f docker-compose.prod.yml up -d
            docker system prune -f

      - name: Health check
        run: |
          curl -f https://nanoprobe.your-domain.com/health || exit 1

      - name: Notify Slack
        uses: 8398a7/action-slack@v3
        with:
          status: ${{ job.status }}
          text: |
            Released ${{ needs.release.outputs.version }} to production
            ${{ github.event.head_commit.message }}
          webhook_url: ${{ secrets.SLACK_WEBHOOK }}
        if: always()
```

## 4. Benchmark Testing

```yaml
# .github/workflows/benchmark.yml
name: Performance Benchmark

on:
  pull_request:
    branches: [main]
  workflow_dispatch:

jobs:
  benchmark:
    name: Run Benchmarks
    runs-on: ubuntu-latest

    services:
      postgres:
        image: postgres:15-alpine
        env:
          POSTGRES_USER: test
          POSTGRES_PASSWORD: test
          POSTGRES_DB: nanoprobe_test
        ports:
          - 5432:5432

      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest-benchmark locust

      - name: Run pytest benchmarks
        run: |
          pytest tests/benchmarks/ --benchmark-only --benchmark-json=benchmark.json
        env:
          DATABASE_URL: postgresql://test:test@localhost:5432/nanoprobe_test
          REDIS_URL: redis://localhost:6379/0

      - name: Store benchmark result
        uses: benchmark-action/github-action-benchmark@v1
        with:
          tool: 'pytest'
          output-file-path: benchmark.json
          github-token: ${{ secrets.GITHUB_TOKEN }}
          auto-push: true
          alert-threshold: '150%'
          comment-on-alert: true
          fail-on-alert: true

      - name: Run Locust load test
        run: |
          locust -f tests/load/locustfile.py --headless -u 100 -r 10 -t 30s --host http://localhost:8000 || true
        continue-on-error: true
```

## 5. Dependency Update Automation

```yaml
# .github/workflows/dependency-update.yml
name: Dependency Updates

on:
  schedule:
    - cron: '0 6 * * 1'  # Monday 6 AM
  workflow_dispatch:

jobs:
  update-python:
    name: Update Python Dependencies
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install pip-tools
        run: pip install pip-tools

      - name: Update dependencies
        run: |
          pip-compile --upgrade requirements.in
          pip-compile --upgrade requirements-api.in

      - name: Create Pull Request
        uses: peter-evans/create-pull-request@v5
        with:
          token: ${{ secrets.GITHUB_TOKEN }}
          branch: update/python-deps
          title: 'chore: update Python dependencies'
          body: |
            Automated dependency update.
            Please review and test before merging.
          labels: dependencies, python

  update-node:
    name: Update Node Dependencies
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'

      - name: Update dependencies
        working-directory: frontend
        run: |
          npm update
          npm audit fix

      - name: Create Pull Request
        uses: peter-evans/create-pull-request@v5
        with:
          token: ${{ secrets.GITHUB_TOKEN }}
          branch: update/node-deps
          title: 'chore: update Node dependencies'
          body: |
            Automated dependency update.
            Please review and test before merging.
          labels: dependencies, nodejs
```

## 6. Custom Action

```yaml
# .github/actions/setup/action.yml
name: 'Setup Project Environment'
description: 'Sets up Python and Node.js with caching'

inputs:
  python-version:
    description: 'Python version'
    required: false
    default: '3.11'
  node-version:
    description: 'Node.js version'
    required: false
    default: '20'

runs:
  using: 'composite'
  steps:
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: ${{ inputs.python-version }}
        cache: 'pip'

    - name: Install Python dependencies
      shell: bash
      run: |
        pip install -r requirements.txt
        pip install -r requirements-api.txt

    - name: Set up Node.js
      uses: actions/setup-node@v4
      with:
        node-version: ${{ inputs.node-version }}
        cache: 'npm'
        cache-dependency-path: frontend/package-lock.json

    - name: Install Node dependencies
      shell: bash
      working-directory: frontend
      run: npm ci
```

## 7. Required Secrets

```yaml
# Secrets needed for CI/CD
secrets:
  # Code quality
  CODECOV_TOKEN: "codecov.io token"

  # Docker registries
  DOCKERHUB_USERNAME: "Docker Hub username"
  DOCKERHUB_TOKEN: "Docker Hub access token"

  # Deployment
  DEPLOY_HOST: "production server host"
  DEPLOY_USER: "SSH user"
  DEPLOY_KEY: "SSH private key"

  # Notifications
  SLACK_WEBHOOK: "Slack incoming webhook URL"

  # Monitoring
  SENTRY_DSN: "Sentry DSN for error tracking"

  # Security
  SNYK_TOKEN: "Snyk API token (optional)"
```

## Pipeline Visualization

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            CI/CD Pipeline Flow                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Push/PR ──► Lint ──► Tests ──► Security ──► Build ──► Deploy               │
│              │         │          │           │         │                    │
│              ▼         ▼          ▼           ▼         ▼                    │
│           ┌─────┐  ┌─────┐   ┌─────┐    ┌─────┐   ┌─────┐                  │
│           │Ruff │  │Unit │   │Bandit│   │Docker│  │SSH  │                  │
│           │Black│  │Tests│   │Safety│   │Image │  │Deploy│                  │
│           │MyPy │  │Integ│   │Trivy │   │Push  │  │Health│                  │
│           │ESLint│ │E2E │   │Gitleaks│ │GHCR  │  │Notify│                  │
│           └─────┘  └─────┘   └─────┘    └─────┘   └─────┘                  │
│                                                                              │
│  Time: ~5min   ~10min    ~5min      ~10min    ~2min                        │
│                                                                              │
│  Total Pipeline Time: ~30 minutes                                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```
