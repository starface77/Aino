#!/usr/bin/env python3
"""
AION REST API Server
RESTful API для интеграции с AION
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import subprocess
import threading
import time
from datetime import datetime
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Разрешаем CORS для фронтенда

class AIONAPIServer:
    """REST API сервер для AION"""
    
    def __init__(self):
        self.app = app
        self.setup_routes()
        self.active_tasks = {}
        
    def setup_routes(self):
        """Настройка API маршрутов"""
        
        @self.app.route('/api/v1/health', methods=['GET'])
        def health_check():
            """Проверка здоровья API"""
            return jsonify({
                'status': 'healthy',
                'service': 'AION API',
                'version': '1.0.0',
                'timestamp': datetime.now().isoformat()
            })
        
        @self.app.route('/api/v1/status', methods=['GET'])
        def get_status():
            """Получить статус AION"""
            try:
                result = subprocess.run(
                    ['python', 'aion.py', 'status'],
                    capture_output=True,
                    text=True,
                    encoding='utf-8'
                )
                
                if result.returncode == 0:
                    # Парсим вывод статуса
                    status_data = self._parse_status_output(result.stdout)
                    return jsonify({
                        'success': True,
                        'data': status_data,
                        'raw_output': result.stdout
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': result.stderr
                    }), 500
                    
            except Exception as e:
                logger.error(f"Status error: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/v1/analyze', methods=['POST'])
        def analyze_project():
            """Запустить анализ проекта"""
            data = request.get_json() or {}
            projects = data.get('projects', [])
            async_mode = data.get('async', True)
            
            if async_mode:
                task_id = f"analyze_{int(time.time())}"
                self._run_async_task('analyze', {'projects': projects}, task_id)
                
                return jsonify({
                    'success': True,
                    'task_id': task_id,
                    'message': 'Анализ запущен в фоновом режиме'
                })
            else:
                # Синхронный режим
                try:
                    cmd = ['python', 'aion.py', 'analyze']
                    if projects:
                        cmd.extend(['--projects'] + projects)
                    
                    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
                    
                    return jsonify({
                        'success': result.returncode == 0,
                        'output': result.stdout,
                        'error': result.stderr if result.returncode != 0 else None
                    })
                    
                except Exception as e:
                    return jsonify({
                        'success': False,
                        'error': str(e)
                    }), 500
        
        @self.app.route('/api/v1/ai', methods=['POST'])
        def ai_analysis():
            """AI анализ"""
            data = request.get_json() or {}
            action = data.get('action', 'analyze')
            provider = data.get('provider', 'local')
            api_key = data.get('api_key')
            async_mode = data.get('async', True)
            
            if async_mode:
                task_id = f"ai_{action}_{int(time.time())}"
                self._run_async_task('ai', {
                    'action': action,
                    'provider': provider,
                    'api_key': api_key
                }, task_id)
                
                return jsonify({
                    'success': True,
                    'task_id': task_id,
                    'message': f'AI {action} запущен в фоновом режиме'
                })
            else:
                try:
                    cmd = ['python', 'aion.py', 'ai', '--action', action, '--provider', provider]
                    if api_key:
                        cmd.extend(['--api-key', api_key])
                    
                    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
                    
                    return jsonify({
                        'success': result.returncode == 0,
                        'output': result.stdout,
                        'error': result.stderr if result.returncode != 0 else None
                    })
                    
                except Exception as e:
                    return jsonify({
                        'success': False,
                        'error': str(e)
                    }), 500
        
        @self.app.route('/api/v1/fix', methods=['POST'])
        def fix_issues():
            """Исправить проблемы"""
            data = request.get_json() or {}
            issue_type = data.get('type')
            limit = data.get('limit', 5)
            async_mode = data.get('async', True)
            
            if async_mode:
                task_id = f"fix_{int(time.time())}"
                self._run_async_task('fix', {
                    'type': issue_type,
                    'limit': limit
                }, task_id)
                
                return jsonify({
                    'success': True,
                    'task_id': task_id,
                    'message': 'Исправления запущены в фоновом режиме'
                })
            else:
                try:
                    cmd = ['python', 'aion.py', 'fix', '--limit', str(limit)]
                    if issue_type:
                        cmd.extend(['--type', issue_type])
                    
                    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
                    
                    return jsonify({
                        'success': result.returncode == 0,
                        'output': result.stdout,
                        'error': result.stderr if result.returncode != 0 else None
                    })
                    
                except Exception as e:
                    return jsonify({
                        'success': False,
                        'error': str(e)
                    }), 500
        
        @self.app.route('/api/v1/tasks/<task_id>', methods=['GET'])
        def get_task_status(task_id):
            """Получить статус задачи"""
            task = self.active_tasks.get(task_id)
            
            if not task:
                return jsonify({
                    'success': False,
                    'error': 'Задача не найдена'
                }), 404
            
            return jsonify({
                'success': True,
                'data': task
            })
        
        @self.app.route('/api/v1/tasks', methods=['GET'])
        def get_all_tasks():
            """Получить все активные задачи"""
            return jsonify({
                'success': True,
                'data': {
                    'active_tasks': len([t for t in self.active_tasks.values() if t['status'] == 'running']),
                    'completed_tasks': len([t for t in self.active_tasks.values() if t['status'] == 'completed']),
                    'failed_tasks': len([t for t in self.active_tasks.values() if t['status'] == 'failed']),
                    'tasks': list(self.active_tasks.keys())
                }
            })
        
        @self.app.route('/api/v1/issues', methods=['GET'])
        def get_issues():
            """Получить список проблем"""
            try:
                result = subprocess.run(
                    ['python', 'aion.py', 'issues', '--limit', '50'],
                    capture_output=True,
                    text=True,
                    encoding='utf-8'
                )
                
                if result.returncode == 0:
                    issues = self._parse_issues_output(result.stdout)
                    return jsonify({
                        'success': True,
                        'data': {
                            'issues': issues,
                            'count': len(issues)
                        }
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': result.stderr
                    }), 500
                    
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/v1/projects', methods=['GET'])
        def get_projects():
            """Получить изученные проекты"""
            try:
                result = subprocess.run(
                    ['python', 'aion.py', 'projects'],
                    capture_output=True,
                    text=True,
                    encoding='utf-8'
                )
                
                if result.returncode == 0:
                    projects = self._parse_projects_output(result.stdout)
                    return jsonify({
                        'success': True,
                        'data': {
                            'projects': projects,
                            'count': len(projects)
                        }
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': result.stderr
                    }), 500
                    
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/v1/command', methods=['POST'])
        def execute_command():
            """Выполнить произвольную команду AION"""
            data = request.get_json() or {}
            command = data.get('command', '')
            
            if not command:
                return jsonify({
                    'success': False,
                    'error': 'Команда не указана'
                }), 400
            
            try:
                # Безопасность - разрешаем только команды AION
                allowed_commands = ['status', 'analyze', 'ai', 'fix', 'issues', 'projects', 'clean']
                cmd_parts = command.split()
                
                if not cmd_parts or cmd_parts[0] not in allowed_commands:
                    return jsonify({
                        'success': False,
                        'error': 'Команда не разрешена'
                    }), 403
                
                result = subprocess.run(
                    ['python', 'aion.py'] + cmd_parts,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    timeout=60
                )
                
                return jsonify({
                    'success': result.returncode == 0,
                    'output': result.stdout,
                    'error': result.stderr if result.returncode != 0 else None,
                    'exit_code': result.returncode
                })
                
            except subprocess.TimeoutExpired:
                return jsonify({
                    'success': False,
                    'error': 'Команда превысила лимит времени выполнения'
                }), 408
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/v1/logs', methods=['GET'])
        def get_logs():
            """Получить логи системы"""
            try:
                logs = []
                
                # Основной лог
                if os.path.exists('aion_log.json'):
                    with open('aion_log.json', 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        for run in data.get('runs', [])[-20:]:
                            logs.append({
                                'timestamp': run['timestamp'],
                                'type': 'analysis',
                                'message': f"Анализ: {len(run.get('github_projects', []))} проектов",
                                'details': run
                            })
                
                # AI лог
                if os.path.exists('ai_analysis_log.json'):
                    with open('ai_analysis_log.json', 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if 'latest_analysis' in data:
                            analysis = data['latest_analysis']
                            logs.append({
                                'timestamp': analysis['timestamp'],
                                'type': 'ai_analysis',
                                'message': f"AI: {analysis['total_files_analyzed']} файлов",
                                'details': analysis
                            })
                
                # Сортируем по времени
                logs.sort(key=lambda x: x['timestamp'], reverse=True)
                
                return jsonify({
                    'success': True,
                    'data': {
                        'logs': logs,
                        'count': len(logs)
                    }
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.errorhandler(404)
        def not_found(error):
            return jsonify({
                'success': False,
                'error': 'Endpoint not found'
            }), 404
        
        @self.app.errorhandler(500)
        def internal_error(error):
            return jsonify({
                'success': False,
                'error': 'Internal server error'
            }), 500
    
    def _run_async_task(self, command, params, task_id):
        """Запуск задачи в фоновом режиме"""
        
        def run_task():
            try:
                self.active_tasks[task_id] = {
                    'status': 'running',
                    'command': command,
                    'params': params,
                    'started_at': datetime.now().isoformat(),
                    'progress': 0
                }
                
                # Формируем команду
                cmd = ['python', 'aion.py', command]
                
                if command == 'analyze' and params.get('projects'):
                    cmd.extend(['--projects'] + params['projects'])
                elif command == 'ai':
                    cmd.extend(['--action', params['action'], '--provider', params['provider']])
                    if params.get('api_key'):
                        cmd.extend(['--api-key', params['api_key']])
                elif command == 'fix':
                    cmd.extend(['--limit', str(params['limit'])])
                    if params.get('type'):
                        cmd.extend(['--type', params['type']])
                
                # Выполняем команду
                result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
                
                self.active_tasks[task_id].update({
                    'status': 'completed' if result.returncode == 0 else 'failed',
                    'completed_at': datetime.now().isoformat(),
                    'output': result.stdout,
                    'error': result.stderr if result.returncode != 0 else None,
                    'exit_code': result.returncode,
                    'progress': 100
                })
                
                logger.info(f"Task {task_id} completed with exit code {result.returncode}")
                
            except Exception as e:
                self.active_tasks[task_id].update({
                    'status': 'failed',
                    'completed_at': datetime.now().isoformat(),
                    'error': str(e),
                    'progress': 0
                })
                logger.error(f"Task {task_id} failed: {e}")
        
        thread = threading.Thread(target=run_task)
        thread.daemon = True
        thread.start()
    
    def _parse_status_output(self, output):
        """Парсинг вывода команды status"""
        # Простой парсинг текстового вывода
        lines = output.strip().split('\n')
        status_data = {}
        
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().replace('📅', '').replace('🔄', '').replace('📚', '').replace('📁', '').replace('🌟', '').replace('⚠️', '').strip()
                value = value.strip()
                
                if key == 'Последний анализ':
                    status_data['last_analysis'] = value
                elif key == 'Всего запусков':
                    status_data['total_runs'] = int(value) if value.isdigit() else 0
                elif key == 'GitHub проектов изучено':
                    status_data['github_projects'] = int(value) if value.isdigit() else 0
                elif key == 'Python файлов в проекте':
                    status_data['python_files'] = int(value) if value.isdigit() else 0
                elif key == 'Проблем в коде':
                    status_data['issues_count'] = int(value) if value.isdigit() else 0
        
        return status_data
    
    def _parse_issues_output(self, output):
        """Парсинг вывода команды issues"""
        issues = []
        lines = output.strip().split('\n')
        
        for line in lines:
            if '📏' in line or '📝' in line:
                # Парсим строку проблемы
                if ' - ' in line:
                    location, description = line.split(' - ', 1)
                    location = location.strip().replace('📏', '').replace('📝', '').strip()
                    
                    if ':' in location:
                        file_path, line_num = location.rsplit(':', 1)
                        issues.append({
                            'file': file_path,
                            'line': int(line_num) if line_num.isdigit() else 0,
                            'description': description,
                            'type': 'long_line' if '📏' in line else 'todo'
                        })
        
        return issues
    
    def _parse_projects_output(self, output):
        """Парсинг вывода команды projects"""
        projects = []
        lines = output.strip().split('\n')
        current_project = {}
        
        for line in lines:
            line = line.strip()
            if line.startswith('⭐'):
                if current_project:
                    projects.append(current_project)
                
                project_name = line.replace('⭐', '').strip()
                current_project = {'name': project_name}
                
            elif line.startswith('🌟 Звезд:'):
                stars_text = line.replace('🌟 Звезд:', '').strip()
                current_project['stars'] = int(stars_text.replace(',', '')) if stars_text.replace(',', '').isdigit() else 0
                
            elif line.startswith('🔗 URL:'):
                current_project['url'] = line.replace('🔗 URL:', '').strip()
                
            elif line.startswith('📝'):
                current_project['description'] = line.replace('📝', '').strip()
        
        if current_project:
            projects.append(current_project)
        
        return projects
    
    def run(self, host='localhost', port=8000, debug=False):
        """Запуск API сервера"""
        print(f"🚀 AION REST API Server")
        print(f"📡 API Endpoint: http://{host}:{port}")
        print(f"🔗 Health Check: http://{host}:{port}/api/v1/health")
        print(f"📊 API Documentation:")
        print(f"   GET  /api/v1/status      - Статус системы")
        print(f"   POST /api/v1/analyze     - Запуск анализа")
        print(f"   POST /api/v1/ai          - AI анализ")
        print(f"   POST /api/v1/fix         - Исправления")
        print(f"   GET  /api/v1/issues      - Список проблем")
        print(f"   GET  /api/v1/projects    - Изученные проекты")
        print(f"   GET  /api/v1/logs        - Системные логи")
        print(f"   POST /api/v1/command     - Выполнить команду")
        
        self.app.run(host=host, port=port, debug=debug)

def main():
    """Главная функция"""
    api_server = AIONAPIServer()
    api_server.run(host='0.0.0.0', port=8000, debug=False)

if __name__ == "__main__":
    main()
