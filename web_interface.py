#!/usr/bin/env python3
"""
AION Web Interface
Веб-интерфейс для управления AION
"""

from flask import Flask, render_template, jsonify, request, Response
import json
import os
import subprocess
from datetime import datetime
import threading
import time

app = Flask(__name__)

class AIONWebInterface:
    """Веб-интерфейс для AION"""
    
    def __init__(self):
        self.app = app
        
        # Устанавливаем правильный путь к шаблонам
        current_dir = os.path.dirname(os.path.abspath(__file__))
        templates_dir = os.path.join(current_dir, 'templates')
        self.app.template_folder = templates_dir
        
        self.setup_routes()
        self.stats = self._load_stats()
        self.running_tasks = {}
        
    def setup_routes(self):
        """Настройка маршрутов"""
        
        @self.app.route('/')
        def dashboard():
            """Главная страница - дашборд"""
            return render_template('dashboard.html', stats=self.stats)
        
        @self.app.route('/api/status')
        def api_status():
            """API статуса"""
            return jsonify(self._get_current_status())
        
        @self.app.route('/api/analyze', methods=['POST'])
        def api_analyze():
            """Запуск анализа"""
            data = request.get_json() or {}
            projects = data.get('projects', None)
            
            task_id = f"analyze_{int(time.time())}"
            self._run_async_command('analyze', projects, task_id)
            
            return jsonify({
                'task_id': task_id,
                'status': 'started',
                'message': 'Анализ запущен'
            })
        
        @self.app.route('/api/ai', methods=['POST'])
        def api_ai():
            """AI анализ"""
            data = request.get_json() or {}
            action = data.get('action', 'analyze')
            provider = data.get('provider', 'local')
            
            task_id = f"ai_{action}_{int(time.time())}"
            self._run_async_ai_command(action, provider, task_id)
            
            return jsonify({
                'task_id': task_id,
                'status': 'started',
                'message': f'AI {action} запущен'
            })
        
        @self.app.route('/api/fix', methods=['POST'])
        def api_fix():
            """Исправление проблем"""
            data = request.get_json() or {}
            issue_type = data.get('type', None)
            limit = data.get('limit', 5)
            
            task_id = f"fix_{int(time.time())}"
            self._run_async_fix_command(issue_type, limit, task_id)
            
            return jsonify({
                'task_id': task_id,
                'status': 'started',
                'message': 'Исправления запущены'
            })
        
        @self.app.route('/api/task/<task_id>')
        def api_task_status(task_id):
            """Статус задачи"""
            task = self.running_tasks.get(task_id, {})
            return jsonify(task)
        
        @self.app.route('/api/issues')
        def api_issues():
            """Получить список проблем"""
            issues = self._get_issues()
            return jsonify(issues)
        
        @self.app.route('/api/projects')
        def api_projects():
            """Получить изученные проекты"""
            projects = self._get_projects()
            return jsonify(projects)
        
        @self.app.route('/api/logs')
        def api_logs():
            """Получить логи"""
            logs = self._get_logs()
            return jsonify(logs)
        
        @self.app.route('/terminal')
        def terminal_page():
            """Страница терминала"""
            return render_template('terminal.html')
        
        @self.app.route('/api/terminal', methods=['POST'])
        def api_terminal():
            """Выполнение команд терминала"""
            data = request.get_json() or {}
            command = data.get('command', '')
            
            if not command:
                return jsonify({'error': 'Команда не указана'})
            
            try:
                result = subprocess.run(
                    ['python', 'aion.py'] + command.split(),
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    timeout=30
                )
                
                return jsonify({
                    'output': result.stdout,
                    'error': result.stderr,
                    'success': result.returncode == 0
                })
                
            except Exception as e:
                return jsonify({
                    'output': '',
                    'error': str(e),
                    'success': False
                })
    
    def _load_stats(self):
        """Загрузка статистики"""
        try:
            if os.path.exists('aion_log.json'):
                with open('aion_log.json', 'r', encoding='utf-8') as f:
                    return json.load(f)
        except:
            pass
        return {'runs': [], 'total_projects': 0, 'total_files': 0}
    
    def _get_current_status(self):
        """Получение текущего статуса"""
        stats = self._load_stats()
        
        if not stats['runs']:
            return {
                'status': 'no_data',
                'message': 'Анализ еще не проводился'
            }
        
        last_run = stats['runs'][-1]
        
        return {
            'status': 'ready',
            'last_analysis': last_run['timestamp'],
            'total_runs': len(stats['runs']),
            'github_projects': stats['total_projects'],
            'python_files': stats['total_files'],
            'issues_count': len(last_run.get('code_analysis', {}).get('issues', []))
        }
    
    def _get_issues(self):
        """Получение списка проблем"""
        stats = self._load_stats()
        if not stats['runs']:
            return []
        
        last_run = stats['runs'][-1]
        return last_run.get('code_analysis', {}).get('issues', [])
    
    def _get_projects(self):
        """Получение изученных проектов"""
        stats = self._load_stats()
        if not stats['runs']:
            return []
        
        last_run = stats['runs'][-1]
        return last_run.get('github_projects', [])
    
    def _get_logs(self):
        """Получение логов"""
        logs = []
        
        # Читаем основной лог
        if os.path.exists('aion_log.json'):
            try:
                with open('aion_log.json', 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for run in data.get('runs', [])[-10:]:  # Последние 10
                        logs.append({
                            'timestamp': run['timestamp'],
                            'type': 'analysis',
                            'message': f"Анализ завершен: {len(run.get('github_projects', []))} проектов"
                        })
            except:
                pass
        
        # Читаем AI лог
        if os.path.exists('ai_analysis_log.json'):
            try:
                with open('ai_analysis_log.json', 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'latest_analysis' in data:
                        analysis = data['latest_analysis']
                        logs.append({
                            'timestamp': analysis['timestamp'],
                            'type': 'ai_analysis',
                            'message': f"AI анализ: {analysis['total_files_analyzed']} файлов"
                        })
            except:
                pass
        
        return sorted(logs, key=lambda x: x['timestamp'], reverse=True)
    
    def _run_async_command(self, command, projects, task_id):
        """Асинхронное выполнение команды"""
        def run_command():
            try:
                self.running_tasks[task_id] = {
                    'status': 'running',
                    'started_at': datetime.now().isoformat(),
                    'progress': 0
                }
                
                cmd = ['python', 'aion.py', command]
                if projects:
                    cmd.extend(['--projects'] + projects)
                
                result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
                
                self.running_tasks[task_id] = {
                    'status': 'completed' if result.returncode == 0 else 'failed',
                    'started_at': self.running_tasks[task_id]['started_at'],
                    'completed_at': datetime.now().isoformat(),
                    'output': result.stdout,
                    'error': result.stderr,
                    'progress': 100
                }
                
            except Exception as e:
                self.running_tasks[task_id] = {
                    'status': 'failed',
                    'started_at': self.running_tasks[task_id]['started_at'],
                    'completed_at': datetime.now().isoformat(),
                    'error': str(e),
                    'progress': 0
                }
        
        thread = threading.Thread(target=run_command)
        thread.daemon = True
        thread.start()
    
    def _run_async_ai_command(self, action, provider, task_id):
        """Асинхронное выполнение AI команды"""
        def run_ai_command():
            try:
                self.running_tasks[task_id] = {
                    'status': 'running',
                    'started_at': datetime.now().isoformat(),
                    'progress': 0
                }
                
                cmd = ['python', 'aion.py', 'ai', '--action', action, '--provider', provider]
                result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
                
                self.running_tasks[task_id] = {
                    'status': 'completed' if result.returncode == 0 else 'failed',
                    'started_at': self.running_tasks[task_id]['started_at'],
                    'completed_at': datetime.now().isoformat(),
                    'output': result.stdout,
                    'error': result.stderr,
                    'progress': 100
                }
                
            except Exception as e:
                self.running_tasks[task_id] = {
                    'status': 'failed',
                    'started_at': self.running_tasks[task_id]['started_at'],
                    'completed_at': datetime.now().isoformat(),
                    'error': str(e),
                    'progress': 0
                }
        
        thread = threading.Thread(target=run_ai_command)
        thread.daemon = True
        thread.start()
    
    def _run_async_fix_command(self, issue_type, limit, task_id):
        """Асинхронное исправление"""
        def run_fix_command():
            try:
                self.running_tasks[task_id] = {
                    'status': 'running',
                    'started_at': datetime.now().isoformat(),
                    'progress': 0
                }
                
                cmd = ['python', 'aion.py', 'fix', '--limit', str(limit)]
                if issue_type:
                    cmd.extend(['--type', issue_type])
                
                result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
                
                self.running_tasks[task_id] = {
                    'status': 'completed' if result.returncode == 0 else 'failed',
                    'started_at': self.running_tasks[task_id]['started_at'],
                    'completed_at': datetime.now().isoformat(),
                    'output': result.stdout,
                    'error': result.stderr,
                    'progress': 100
                }
                
            except Exception as e:
                self.running_tasks[task_id] = {
                    'status': 'failed',
                    'started_at': self.running_tasks[task_id]['started_at'],
                    'completed_at': datetime.now().isoformat(),
                    'error': str(e),
                    'progress': 0
                }
        
        thread = threading.Thread(target=run_fix_command)
        thread.daemon = True
        thread.start()
    
    def run(self, host='localhost', port=5000, debug=False):
        """Запуск веб-сервера"""
        print(f"🌐 AION Web Interface")
        print(f"📡 Server: http://{host}:{port}")
        print("🎛️ Dashboard: Главная страница")
        print("💻 Terminal: /terminal")
        print("📊 API: /api/*")
        
        self.app.run(host=host, port=port, debug=debug)

def create_templates():
    """Создание HTML шаблонов"""
    
    # Создаем папку templates в правильном месте
    current_dir = os.path.dirname(os.path.abspath(__file__))
    templates_dir = os.path.join(current_dir, 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    
    # Dashboard template
    dashboard_html = '''<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AION Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #1a1a1a; color: #fff; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { text-align: center; margin-bottom: 30px; }
        .header h1 { font-size: 2.5em; color: #00ff88; margin-bottom: 10px; }
        .header p { color: #888; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .card { background: #2a2a2a; border-radius: 10px; padding: 20px; border: 1px solid #333; }
        .card h3 { color: #00ff88; margin-bottom: 15px; }
        .stat { display: flex; justify-content: space-between; margin-bottom: 10px; }
        .stat-value { color: #00ff88; font-weight: bold; }
        .button { background: #00ff88; color: #000; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin: 5px; font-weight: bold; }
        .button:hover { background: #00dd77; }
        .button.secondary { background: #333; color: #fff; }
        .button.secondary:hover { background: #555; }
        .actions { text-align: center; margin: 20px 0; }
        .log { background: #111; padding: 15px; border-radius: 5px; max-height: 300px; overflow-y: auto; font-family: monospace; }
        .log-entry { margin-bottom: 10px; }
        .log-time { color: #666; }
        #output { margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>⚡ AION Dashboard</h1>
            <p>AI Self-Improvement System Control Panel</p>
        </div>
        
        <div class="grid">
            <div class="card">
                <h3>📊 Статус системы</h3>
                <div class="stat">
                    <span>Статус:</span>
                    <span class="stat-value" id="status">Загрузка...</span>
                </div>
                <div class="stat">
                    <span>Анализов выполнено:</span>
                    <span class="stat-value" id="total-runs">0</span>
                </div>
                <div class="stat">
                    <span>GitHub проектов:</span>
                    <span class="stat-value" id="github-projects">0</span>
                </div>
                <div class="stat">
                    <span>Python файлов:</span>
                    <span class="stat-value" id="python-files">0</span>
                </div>
                <div class="stat">
                    <span>Проблем найдено:</span>
                    <span class="stat-value" id="issues-count">0</span>
                </div>
            </div>
            
            <div class="card">
                <h3>🔧 Управление</h3>
                <button class="button" onclick="runAnalysis()">📈 Запустить анализ</button>
                <button class="button" onclick="runAI()">🤖 AI анализ</button>
                <button class="button" onclick="fixIssues()">⚡ Исправить проблемы</button>
                <button class="button secondary" onclick="showProjects()">🌟 Проекты</button>
                <button class="button secondary" onclick="showIssues()">⚠️ Проблемы</button>
                <button class="button secondary" onclick="openTerminal()">💻 Терминал</button>
            </div>
            
            <div class="card">
                <h3>📝 Последние действия</h3>
                <div class="log" id="logs">
                    Загрузка логов...
                </div>
            </div>
        </div>
        
        <div id="output"></div>
    </div>

    <script>
        // Обновление статуса
        async function updateStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                
                document.getElementById('status').textContent = data.status === 'ready' ? 'Готов' : 'Нет данных';
                document.getElementById('total-runs').textContent = data.total_runs || 0;
                document.getElementById('github-projects').textContent = data.github_projects || 0;
                document.getElementById('python-files').textContent = data.python_files || 0;
                document.getElementById('issues-count').textContent = data.issues_count || 0;
            } catch (error) {
                console.error('Ошибка загрузки статуса:', error);
            }
        }
        
        // Обновление логов
        async function updateLogs() {
            try {
                const response = await fetch('/api/logs');
                const logs = await response.json();
                
                const logsElement = document.getElementById('logs');
                logsElement.innerHTML = logs.map(log => 
                    `<div class="log-entry">
                        <span class="log-time">[${new Date(log.timestamp).toLocaleString()}]</span>
                        ${log.message}
                    </div>`
                ).join('');
            } catch (error) {
                console.error('Ошибка загрузки логов:', error);
            }
        }
        
        // Запуск анализа
        async function runAnalysis() {
            try {
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({})
                });
                const data = await response.json();
                
                showOutput(`Анализ запущен (${data.task_id}). Обновление через несколько секунд...`);
                setTimeout(() => { updateStatus(); updateLogs(); }, 5000);
            } catch (error) {
                showOutput('Ошибка запуска анализа: ' + error.message);
            }
        }
        
        // AI анализ
        async function runAI() {
            try {
                const response = await fetch('/api/ai', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ action: 'analyze', provider: 'local' })
                });
                const data = await response.json();
                
                showOutput(`AI анализ запущен (${data.task_id}). Обновление через несколько секунд...`);
                setTimeout(() => { updateStatus(); updateLogs(); }, 8000);
            } catch (error) {
                showOutput('Ошибка AI анализа: ' + error.message);
            }
        }
        
        // Исправление проблем
        async function fixIssues() {
            try {
                const response = await fetch('/api/fix', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ limit: 5 })
                });
                const data = await response.json();
                
                showOutput(`Исправления запущены (${data.task_id}). Обновление через несколько секунд...`);
                setTimeout(() => { updateStatus(); updateLogs(); }, 10000);
            } catch (error) {
                showOutput('Ошибка исправлений: ' + error.message);
            }
        }
        
        // Показать проекты
        async function showProjects() {
            try {
                const response = await fetch('/api/projects');
                const projects = await response.json();
                
                const output = projects.map(p => 
                    `🌟 ${p.name}: ${p.stars} ⭐ - ${p.description}`
                ).join('\\n') || 'Нет данных о проектах';
                
                showOutput(output);
            } catch (error) {
                showOutput('Ошибка загрузки проектов: ' + error.message);
            }
        }
        
        // Показать проблемы
        async function showIssues() {
            try {
                const response = await fetch('/api/issues');
                const issues = await response.json();
                
                const output = issues.slice(0, 10).map(issue => 
                    `⚠️ ${issue.file}:${issue.line} - ${issue.description || issue.type}`
                ).join('\\n') || 'Нет проблем найдено';
                
                showOutput(output);
            } catch (error) {
                showOutput('Ошибка загрузки проблем: ' + error.message);
            }
        }
        
        // Открыть терминал
        function openTerminal() {
            window.open('/terminal', '_blank');
        }
        
        // Показать вывод
        function showOutput(text) {
            const output = document.getElementById('output');
            output.innerHTML = `<div class="card"><h3>📤 Результат</h3><div class="log">${text.replace(/\\n/g, '<br>')}</div></div>`;
        }
        
        // Инициализация
        updateStatus();
        updateLogs();
        
        // Автообновление каждые 30 секунд
        setInterval(() => {
            updateStatus();
            updateLogs();
        }, 30000);
    </script>
</body>
</html>'''
    
    with open(f'{templates_dir}/dashboard.html', 'w', encoding='utf-8') as f:
        f.write(dashboard_html)
    
    # Terminal template
    terminal_html = '''<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AION Terminal</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Courier New', monospace; background: #000; color: #00ff00; }
        .terminal { padding: 20px; height: 100vh; overflow-y: auto; }
        .header { color: #00ff88; margin-bottom: 20px; }
        .input-line { display: flex; margin-bottom: 10px; }
        .prompt { color: #00ff88; margin-right: 10px; }
        #command-input { background: transparent; border: none; color: #00ff00; font-family: inherit; font-size: inherit; outline: none; flex: 1; }
        .output { margin-bottom: 15px; white-space: pre-wrap; }
        .error { color: #ff4444; }
        .success { color: #44ff44; }
    </style>
</head>
<body>
    <div class="terminal">
        <div class="header">
            <h2>⚡ AION Terminal Interface</h2>
            <p>Введите команды AION (без 'python aion.py')</p>
            <p>Примеры: status, analyze, ai --action analyze, fix</p>
            <hr>
        </div>
        
        <div id="output"></div>
        
        <div class="input-line">
            <span class="prompt">AION></span>
            <input type="text" id="command-input" autofocus>
        </div>
    </div>

    <script>
        const output = document.getElementById('output');
        const input = document.getElementById('command-input');
        
        // Обработка команд
        input.addEventListener('keypress', async function(e) {
            if (e.key === 'Enter') {
                const command = input.value.trim();
                if (!command) return;
                
                // Показываем команду
                addOutput(`AION> ${command}`, 'command');
                input.value = '';
                
                try {
                    const response = await fetch('/api/terminal', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ command })
                    });
                    
                    const data = await response.json();
                    
                    if (data.output) {
                        addOutput(data.output, 'success');
                    }
                    
                    if (data.error) {
                        addOutput(data.error, 'error');
                    }
                    
                } catch (error) {
                    addOutput('Ошибка выполнения команды: ' + error.message, 'error');
                }
            }
        });
        
        function addOutput(text, type = '') {
            const div = document.createElement('div');
            div.className = `output ${type}`;
            div.textContent = text;
            output.appendChild(div);
            
            // Прокрутка вниз
            div.scrollIntoView();
        }
        
        // Начальное сообщение
        addOutput('Terminal ready. Type commands...', 'success');
    </script>
</body>
</html>'''
    
    with open(f'{templates_dir}/terminal.html', 'w', encoding='utf-8') as f:
        f.write(terminal_html)

def main():
    """Главная функция"""
    
    # Создаем шаблоны
    create_templates()
    
    # Запускаем веб-интерфейс
    web_interface = AIONWebInterface()
    web_interface.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == "__main__":
    main()
