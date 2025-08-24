# 🤖 AION - AI Self-Improvement System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-71%25-yellow.svg)](tests/)
[![Flask](https://img.shields.io/badge/Flask-2.3+-red.svg)](https://flask.palletsprojects.com/)

**AION** is a comprehensive development platform with AI, including code analysis, web interface, REST API, monitoring, testing and automation. More than just AI — it's a complete developer ecosystem.

## 🚀 What is AION?

AION is **more than AI**. It's a complete **Developer Platform** including:

### 🎯 Core Platform:
- 🔍 **Code Analysis** — deep project analysis and metrics
- 🌟 **GitHub Integration** — learning best practices from top projects
- ⚡ **Auto-fixes** — intelligent fixes and optimizations
- 📊 **Analytics** — detailed reports and insights

### 🤖 AI Module:
- 🧠 **AI Analysis** — integration with Gemma 27B
- 💡 **Smart Suggestions** — AI-powered recommendations  
- 🎯 **Automatic Improvements** — self-learning algorithms

### 🌐 Web Platform:
- 📱 **Modern Interface** — responsive web dashboard
- 💻 **Built-in Terminal** — full-featured console
- 📈 **Data Visualization** — charts and metrics
- 🎛️ **Interactive Management** — drag & drop interface

### 🚀 API Ecosystem:
- 🔗 **RESTful API** — complete integration
- 📡 **WebSocket** — real-time updates  
- 🔌 **Plugins** — extensible architecture
- 🌍 **CORS Support** — cross-origin requests

### 📊 DevOps Tools:
- 🔍 **Monitoring** — system resources and metrics
- 🧪 **Testing** — automated QA
- 📝 **Logging** — structured logging
- 🏥 **Healthchecks** — diagnostic system

### 🛠️ Infrastructure:
- ⚙️ **Configuration** — flexible settings
- 🔐 **Security** — secure by design
- 🚀 **Performance** — optimized for speed
- 📦 **Deployment** — production ready

## 📸 Screenshots

### Web Interface
```
📊 Real-time metrics dashboard
💻 Built-in terminal for commands
🎛️ Interactive analysis management
```

### CLI Interface
```bash
$ python aion.py analyze
🚀 AION - Starting analysis
🔍 Fast GitHub analysis:
   Checking AutoGPT... ✅ 178,031 ⭐
   Checking transformers... ✅ 148,729 ⭐

📁 Local code analysis:
   📄 Python files: 10
   📏 Total lines: 5,744
   ⚠️ Issues found: 12
```

## 🛠️ Installation

### Requirements
- Python 3.8+
- pip

### Quick Start

```bash
# 1. Clone repository
git clone https://github.com/starface77/aion.git
cd aion

# 2. Install dependencies
pip install -r requirements.txt

# 3. Setup (optional)
cp env.example .env
# Edit .env file with your API keys

# 4. First run
python aion.py analyze
```

## 📋 Commands

### Basic Operations
```bash
python aion.py analyze                    # Analyze project and GitHub
python aion.py status                     # Show system status
python aion.py issues                     # Found issues
python aion.py projects                   # Studied projects
python aion.py fix                        # Auto-fixes
python aion.py clean                      # Clean logs
```

### AI Commands
```bash
python aion.py ai --action analyze        # AI code analysis
python aion.py ai --action plan           # Generate improvement plan
python aion.py ai --action improve        # Apply improvements
python aion.py ai --provider inference    # Use Gemma 27B
```

### Web Interface and API
```bash
python aion.py web                        # Web interface (port 5000)
python aion.py api                        # REST API server (port 8000)
```

### Monitoring and Testing
```bash
python aion.py monitor                    # System monitoring
python aion.py test                       # Run tests
```

## 🌐 Web Interface

Launch web interface and open http://localhost:5000

**Features:**
- 📊 Real-time metrics dashboard
- 🔧 Analysis and fixes management  
- 💻 Built-in terminal
- 📈 Results visualization
- 🤖 AI interface

## 🚀 REST API

### Start API Server
```bash
python aion.py api
# API available at http://localhost:8000
```

### Main Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/health` | API health check |
| GET | `/api/v1/status` | System status |
| POST | `/api/v1/analyze` | Start analysis |
| POST | `/api/v1/ai` | AI operations |
| GET | `/api/v1/issues` | Issues list |
| GET | `/api/v1/projects` | Studied projects |

### Example Requests

```bash
# Get status
curl http://localhost:8000/api/v1/status

# Start analysis
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"async": true}'

# AI code analysis
curl -X POST http://localhost:8000/api/v1/ai \
  -H "Content-Type: application/json" \
  -d '{"action": "analyze", "provider": "inference"}'
```

## 🤖 AI Integration

AION supports integration with modern AI models:

### Supported Providers
- **Local** — basic local analysis
- **Inference.net** — Gemma 27B for advanced analysis

### AI Setup
```bash
# Set API key in .env file
INFERENCE_API_KEY=your_api_key_here

# Use AI analysis
python aion.py ai --provider inference --action analyze
```

## 📊 Monitoring

The monitoring system tracks:

- 🖥️ **System Resources** — CPU, memory, disk
- ⚡ **AION Metrics** — analyses, AI requests, errors  
- 📝 **Events** — detailed operation log
- 🏥 **System Health** — overall status assessment

```bash
# Show monitoring status
python aion.py monitor --action status

# Interactive dashboard
python aion.py monitor --action dashboard
```

## 🧪 Testing

AION includes comprehensive testing system:

```bash
# Run all tests
python aion.py test

# Specific test category
python aion.py test --action specific --name core

# Testing history  
python aion.py test --action history
```

**Test Categories:**
- **Core** — main functionality
- **Files** — file operations
- **Config** — configuration
- **API** — AI functionality
- **Monitoring** — monitoring system
- **Integration** — integration tests

## ⚙️ Configuration

### Environment Variables (.env)
```bash
# AI provider
INFERENCE_API_KEY=your_inference_api_key

# Web interface
WEB_HOST=localhost
WEB_PORT=5000

# API server
API_HOST=localhost  
API_PORT=8000

# Monitoring
MONITORING_ENABLED=true
```

### Configuration via config.yaml
```yaml
analysis:
  max_files: 10
  max_lines_per_file: 50
  github_projects: ["AutoGPT", "transformers"]
  
ai:
  provider: "inference"
  model: "google/gemma-3-27b-instruct/bf-16"
  
monitoring:
  enabled: true
  interval: 5
```

## 🏗️ Architecture

```
AION System
├── aion.py              # 🎯 Main CLI system
├── ai_core.py           # 🤖 AI providers and logic
├── web_interface.py     # 🌐 Flask web interface
├── api_server.py        # 🚀 REST API server
├── monitoring.py        # 📊 Monitoring system  
├── testing.py           # 🧪 Testing framework
├── terminal.py          # 💻 Interactive terminal
├── templates/           # 🎨 HTML templates
├── requirements.txt     # 📦 Dependencies
└── config/             # ⚙️ Configuration files
```

## 🤝 Contributing

We welcome contributions to AION development! 

### How to Contribute:

1. **Fork** the repository
2. Create **feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit** changes (`git commit -m 'Add amazing feature'`)
4. **Push** to branch (`git push origin feature/amazing-feature`)
5. Open **Pull Request**

### Development Guidelines:
- Follow PEP 8 standard
- Add tests for new functionality
- Update documentation
- Ensure all tests pass

## 🐛 Bug Reports

Found a bug? Create an [issue](https://github.com/starface77/aion/issues) with:

- **Problem description**
- **Steps to reproduce**  
- **Expected behavior**
- **Screenshots** (if applicable)
- **Python version** and **OS**

## 📝 Changelog

### v1.0.0 (2025-08-24)
- ✅ First stable release
- ✅ Core analysis functionality
- ✅ AI integration with Gemma 27B
- ✅ Web interface and REST API
- ✅ Monitoring system
- ✅ Comprehensive testing

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Flask](https://flask.palletsprojects.com/) — web framework
- [Inference.net](https://inference.net/) — AI provider
- [psutil](https://github.com/giampaolo/psutil) — system monitoring
- All project contributors

## 📞 Contact

- **GitHub Issues** — [create issue](https://github.com/starface77/aion/issues)
- **Email** — your.email@example.com
- **Telegram** — @yourusername

---

**AION** — making code analysis and improvement simple and automated! 🚀

## 🌍 Language Versions

- 🇷🇺 [Russian README](README.md)
- 🇺🇸 [English README](README_EN.md) (current)

## 📚 Documentation

- [Contributing Guide](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)
- [API Documentation](docs/api.md)
- [Installation Guide](docs/installation.md)

## 🎯 Use Cases

### For Developers
- **Code Quality Control** — automatic issue detection
- **Learning from Best Practices** — studying popular projects
- **AI-Powered Analysis** — smart code improvement suggestions

### For Teams
- **Code Review Automation** — systematic quality checks
- **Performance Monitoring** — system health tracking
- **Integration** — REST API for CI/CD pipelines

### For Organizations
- **Code Standards Enforcement** — consistent quality across projects
- **Metrics & Analytics** — detailed development insights
- **Scalable Architecture** — easily deployable solution

## 🚦 Quick Examples

### Basic Usage
```bash
# Quick project scan
python aion.py analyze

# View results
python aion.py status
python aion.py issues --limit 5

# Apply fixes
python aion.py fix --limit 3
```

### Advanced AI Analysis
```bash
# Setup AI provider
export INFERENCE_API_KEY="your_key"

# Run AI analysis
python aion.py ai --provider inference --action analyze

# Generate improvement plan
python aion.py ai --action plan

# Apply AI suggestions
python aion.py ai --action improve
```

### Web Dashboard
```bash
# Start web interface
python aion.py web --host 0.0.0.0 --port 8080

# Access dashboard at http://localhost:8080
# Features: real-time metrics, interactive controls, embedded terminal
```

### API Integration
```bash
# Start API server
python aion.py api --port 9000

# Use in your applications
curl -X POST http://localhost:9000/api/v1/analyze
curl -X GET http://localhost:9000/api/v1/issues
```

---

**Ready to improve your code with AI? Get started now!** ⚡
