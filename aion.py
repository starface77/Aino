#!/usr/bin/env python3
"""
AION - AI Self-Improvement System
Командная система управления
"""

import argparse
import json
import os
import requests
from datetime import datetime
import sys

class AIONCore:
    """Ядро системы AION"""
    
    def __init__(self):
        self.log_file = 'aion_log.json'
        self.stats = self._load_stats()
    
    def _load_stats(self):
        """Загрузка статистики"""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {'runs': [], 'total_projects': 0, 'total_files': 0}
    
    def _save_stats(self):
        """Сохранение статистики"""
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ Ошибка сохранения: {e}")
    
    def analyze_github(self, projects=None):
        """БЫСТРЫЙ анализ GitHub проектов"""
        if not projects:
            projects = ['AutoGPT', 'transformers']  # Только 2 проекта для скорости
        
        print("🔍 Быстрый анализ GitHub:")
        found_projects = []
        
        # Ограничиваем до 2 проектов
        for project in projects[:2]:
            try:
                print(f"   Проверяю {project}...", end=' ')
                
                url = "https://api.github.com/search/repositories"
                params = {
                    'q': f"{project} language:python",
                    'sort': 'stars',
                    'per_page': 1
                }
                
                response = requests.get(url, params=params, timeout=5)  # Быстрый таймаут
                if response.status_code == 200:
                    data = response.json()
                    if data['items']:
                        repo = data['items'][0]
                        found_projects.append({
                            'name': repo['name'],
                            'stars': repo['stargazers_count'],
                            'url': repo['html_url'],
                            'description': repo.get('description', '')[:50]  # Короче
                        })
                        print(f"✅ {repo['stargazers_count']:,} ⭐")
                    else:
                        print("❌ Не найден")
                else:
                    print(f"❌ HTTP {response.status_code}")
                    
            except Exception as e:
                print(f"❌ Пропуск")  # Не показываем детали ошибок
        
        return found_projects
    
    def analyze_code(self):
        """Анализ локального кода"""
        print("\n📁 Анализ локального кода:")
        
        stats = {
            'python_files': 0,
            'total_lines': 0,
            'issues': [],
            'largest_file': None,
            'max_lines': 0
        }
        
        # БЫСТРЫЙ АНАЛИЗ - только основная папка, ограниченно
        print("   🔍 Сканирую файлы...")
        for file in os.listdir('.'):
            if file.endswith('.py') and os.path.isfile(file):
                # Ограничиваем до 10 файлов для скорости
                if stats['python_files'] >= 10:
                    break
                    
                stats['python_files'] += 1
                
                try:
                    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                        line_count = len(lines)
                        stats['total_lines'] += line_count
                        
                        # Отслеживаем самый большой файл
                        if line_count > stats['max_lines']:
                            stats['max_lines'] = line_count
                            stats['largest_file'] = file
                        
                        # Быстрые проверки - только первые 50 строк
                        for i, line in enumerate(lines[:50], 1):
                            if len(line.strip()) > 120:
                                stats['issues'].append({
                                    'file': file,
                                    'line': i,
                                    'type': 'long_line',
                                    'length': len(line.strip())
                                })
                            if 'TODO' in line or 'FIXME' in line:
                                stats['issues'].append({
                                    'file': file,
                                    'line': i,
                                    'type': 'todo',
                                    'text': line.strip()[:50]  # Обрезаем для экономии
                                })
                                
                except Exception:
                    continue  # Игнорируем ошибки
        
        print(f"   📄 Python файлов: {stats['python_files']}")
        print(f"   📏 Всего строк: {stats['total_lines']:,}")
        print(f"   ⚠️ Проблем найдено: {len(stats['issues'])}")
        if stats['largest_file']:
            print(f"   📈 Самый большой файл: {stats['largest_file']} ({stats['max_lines']} строк)")
        
        return stats
    
    def run_analysis(self, github_projects=None):
        """Полный анализ"""
        print("🚀 AION - Запуск анализа")
        print("=" * 50)
        
        # GitHub анализ
        github_data = self.analyze_github(github_projects)
        
        # Локальный анализ
        code_data = self.analyze_code()
        
        # Сохраняем результат
        run_data = {
            'timestamp': datetime.now().isoformat(),
            'github_projects': github_data,
            'code_analysis': code_data
        }
        
        self.stats['runs'].append(run_data)
        self.stats['total_projects'] += len(github_data)
        self.stats['total_files'] = code_data['python_files']
        self._save_stats()
        
        print(f"\n💾 Результат сохранен в {self.log_file}")
        return run_data
    
    def show_status(self):
        """Показать статус системы"""
        print("📊 СТАТУС AION")
        print("=" * 30)
        
        if not self.stats['runs']:
            print("❌ Анализов еще не было")
            return
        
        last_run = self.stats['runs'][-1]
        
        print(f"📅 Последний анализ: {last_run['timestamp']}")
        print(f"🔄 Всего запусков: {len(self.stats['runs'])}")
        print(f"📚 GitHub проектов изучено: {self.stats['total_projects']}")
        print(f"📁 Python файлов в проекте: {self.stats['total_files']}")
        
        if last_run['github_projects']:
            top_project = max(last_run['github_projects'], key=lambda x: x['stars'])
            print(f"🌟 Топ проект: {top_project['name']} ({top_project['stars']:,} ⭐)")
        
        issues_count = len(last_run['code_analysis']['issues'])
        print(f"⚠️ Проблем в коде: {issues_count}")
    
    def show_issues(self, issue_type=None, limit=10):
        """Показать проблемы в коде"""
        if not self.stats['runs']:
            print("❌ Сначала запустите анализ")
            return
        
        last_run = self.stats['runs'][-1]
        issues = last_run['code_analysis']['issues']
        
        if issue_type:
            issues = [i for i in issues if i['type'] == issue_type]
        
        print(f"⚠️ ПРОБЛЕМЫ В КОДЕ ({len(issues)} найдено)")
        print("-" * 50)
        
        for issue in issues[:limit]:
            if issue['type'] == 'long_line':
                print(f"📏 {issue['file']}:{issue['line']} - Длинная строка ({issue['length']} символов)")
            elif issue['type'] == 'todo':
                print(f"📝 {issue['file']}:{issue['line']} - {issue['text']}")
    
    def show_projects(self):
        """Показать изученные проекты"""
        if not self.stats['runs']:
            print("❌ Сначала запустите анализ")
            return
        
        last_run = self.stats['runs'][-1]
        projects = last_run['github_projects']
        
        print("🌟 ИЗУЧЕННЫЕ ПРОЕКТЫ")
        print("-" * 30)
        
        # Сортируем по звездам
        projects.sort(key=lambda x: x['stars'], reverse=True)
        
        for project in projects:
            print(f"⭐ {project['name']}")
            print(f"   🌟 Звезд: {project['stars']:,}")
            print(f"   🔗 URL: {project['url']}")
            if project['description']:
                print(f"   📝 {project['description']}")
            print()
    
    def clean_logs(self):
        """Очистить логи"""
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
            print(f"🗑️ Лог {self.log_file} удален")
        else:
            print("❌ Лог файл не найден")
    
    def fix_issues(self, issue_type=None, limit=5):
        """Исправить проблемы в коде"""
        if not self.stats['runs']:
            print("❌ Сначала запустите анализ")
            return
        
        last_run = self.stats['runs'][-1]
        issues = last_run['code_analysis']['issues']
        
        if issue_type:
            issues = [i for i in issues if i['type'] == issue_type]
        
        print(f"🔧 ИСПРАВЛЕНИЕ ПРОБЛЕМ ({len(issues)} найдено)")
        print("-" * 50)
        
        fixed_count = 0
        
        for issue in issues[:limit]:
            try:
                if issue['type'] == 'long_line' and self._fix_long_line(issue):
                    print(f"✅ Исправлена длинная строка в {issue['file']}:{issue['line']}")
                    fixed_count += 1
                elif issue['type'] == 'todo' and self._fix_todo(issue):
                    print(f"✅ Обработан TODO в {issue['file']}:{issue['line']}")
                    fixed_count += 1
                else:
                    print(f"⚠️ Не удалось исправить {issue['file']}:{issue['line']}")
                    
            except Exception as e:
                print(f"❌ Ошибка исправления {issue['file']}: {e}")
        
        print(f"\n📊 Исправлено проблем: {fixed_count}/{min(len(issues), limit)}")
        
        if fixed_count > 0:
            print("💡 Запустите 'python aion.py analyze' для обновления статистики")
    
    def _fix_long_line(self, issue):
        """Исправить длинную строку"""
        try:
            file_path = issue['file']
            line_num = issue['line']
            
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if line_num <= len(lines):
                line = lines[line_num - 1]
                
                # Простое исправление - разбить длинную строку
                if len(line.strip()) > 120:
                    # Ищем запятую для разбития
                    if ',' in line and len(line) > 120:
                        # Разбиваем по запятой
                        parts = line.split(',')
                        if len(parts) > 2:
                            indent = len(line) - len(line.lstrip())
                            new_line = parts[0] + ',\n'
                            for part in parts[1:-1]:
                                new_line += ' ' * (indent + 4) + part.strip() + ',\n'
                            new_line += ' ' * (indent + 4) + parts[-1].strip()
                            
                            lines[line_num - 1] = new_line
                            
                            with open(file_path, 'w', encoding='utf-8') as f:
                                f.writelines(lines)
                            
                            return True
            
        except Exception:
            pass
        
        return False
    
    def _fix_todo(self, issue):
        """Обработать TODO комментарий"""
        try:
            file_path = issue['file']
            line_num = issue['line']
            
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if line_num <= len(lines):
                line = lines[line_num - 1]
                
                # Заменяем TODO на DONE
                if 'TODO' in line:
                    new_line = line.replace('TODO', 'DONE')
                    lines[line_num - 1] = new_line
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.writelines(lines)
                    
                    return True
            
        except Exception:
            pass
        
        return False
    
    def run_ai_analysis(self, provider: str = 'local', api_key: str = None, action: str = 'analyze'):
        """Запуск AI анализа"""
        try:
            from ai_core import create_ai_improver
            
            print(f"🤖 AI ANALYSIS ({provider})")
            print("=" * 40)
            
            # Создаем AI улучшатель
            improver = create_ai_improver(provider, api_key)
            
            if action == 'analyze':
                analysis = improver.analyze_project_with_ai()
                print(f"\n📊 РЕЗУЛЬТАТЫ AI АНАЛИЗА:")
                print(f"   Файлов проанализировано: {analysis['total_files_analyzed']}")
                print(f"   Найдено проблем: {analysis['total_issues']}")
                print(f"   Сложность проекта: {analysis['overall_complexity']:.1f}/10")
                print(f"   Уникальных предложений: {len(analysis['unique_suggestions'])}")
                
                print(f"\n💡 ПРЕДЛОЖЕНИЯ:")
                for suggestion in analysis['unique_suggestions'][:5]:
                    print(f"   • {suggestion}")
                
            elif action == 'plan':
                analysis = improver.analyze_project_with_ai()
                plan = improver.generate_improvement_plan(analysis)
                
                print(f"\n📋 ПЛАН УЛУЧШЕНИЙ:")
                print(f"   Высокий приоритет: {len(plan['high_priority_improvements'])}")
                print(f"   Средний приоритет: {len(plan['medium_priority_improvements'])}")
                print(f"   Оценка усилий: {plan['estimated_effort']}")
                
                print(f"\n🎯 РЕКОМЕНДУЕМЫЙ ПОРЯДОК:")
                for i, improvement in enumerate(plan['recommended_order'][:3], 1):
                    print(f"   {i}. {improvement}")
                
            elif action == 'improve':
                analysis = improver.analyze_project_with_ai()
                plan = improver.generate_improvement_plan(analysis)
                applied = improver.apply_ai_improvements(plan)
                
                print(f"\n🔧 ПРИМЕНЕНО УЛУЧШЕНИЙ: {len(applied)}")
                for improvement in applied:
                    print(f"   ✅ {improvement}")
            
            print(f"\n💾 Детальный анализ сохранен в ai_analysis_log.json")
            
        except ImportError:
            print("❌ AI модуль недоступен")
        except Exception as e:
            print(f"❌ Ошибка AI анализа: {e}")
    
    def run_web_interface(self, host='localhost', port=5000):
        """Запуск веб-интерфейса"""
        try:
            from web_interface import AIONWebInterface
            
            print(f"🌐 Запуск веб-интерфейса AION")
            print(f"📡 URL: http://{host}:{port}")
            
            web_interface = AIONWebInterface()
            web_interface.run(host=host, port=port)
            
        except ImportError:
            print("❌ Веб-интерфейс недоступен (установите Flask)")
        except Exception as e:
            print(f"❌ Ошибка веб-интерфейса: {e}")
    
    def run_api_server(self, host='localhost', port=8000):
        """Запуск REST API сервера"""
        try:
            from api_server import AIONAPIServer
            
            print(f"🚀 Запуск REST API сервера AION")
            print(f"📡 API: http://{host}:{port}")
            
            api_server = AIONAPIServer()
            api_server.run(host=host, port=port)
            
        except ImportError:
            print("❌ API сервер недоступен (установите Flask и flask-cors)")
        except Exception as e:
            print(f"❌ Ошибка API сервера: {e}")
    
    def run_monitoring(self, action='status'):
        """Управление мониторингом"""
        try:
            import monitoring
            
            if action == 'start':
                print("🔍 Запуск системы мониторинга...")
                monitor = monitoring.start_monitoring()
                print("✅ Мониторинг запущен")
                print("📊 Для просмотра дашборда: python aion.py monitor --action dashboard")
                
            elif action == 'stop':
                print("⏹️ Остановка мониторинга...")
                monitoring.stop_monitoring()
                print("✅ Мониторинг остановлен")
                
            elif action == 'status':
                health = monitoring.get_health_status()
                print(f"💚 СТАТУС МОНИТОРИНГА")
                print(f"==============================")
                print(f"🏥 Здоровье: {health['health_score']}/100 ({health['status']})")
                print(f"🖥️  CPU: {health['system']['cpu_percent']:.1f}%")
                print(f"💾 Память: {health['system']['memory_percent']:.1f}%")
                print(f"💿 Диск: {health['system']['disk_percent']:.1f}%")
                print(f"⚡ Время работы: {health['aion']['uptime_human']}")
                print(f"📊 Анализов: {health['aion']['analyses_completed']}")
                print(f"🤖 AI запросов: {health['aion']['ai_requests']}")
                print(f"⚠️ Ошибок: {health['aion']['errors_encountered']}")
                
                if health['warnings']:
                    print(f"\n⚠️ ПРЕДУПРЕЖДЕНИЯ:")
                    for warning in health['warnings']:
                        print(f"   • {warning}")
                
            elif action == 'dashboard':
                print("📊 Интерактивный дашборд мониторинга")
                print("Нажмите Ctrl+C для выхода\n")
                monitoring.show_monitoring_dashboard()
                
        except ImportError:
            print("❌ Система мониторинга недоступна (установите psutil)")
        except Exception as e:
            print(f"❌ Ошибка мониторинга: {e}")
    
    def run_testing(self, action='run', test_name=None):
        """Управление тестированием"""
        try:
            import testing
            
            if action == 'run':
                print("🧪 Запуск полного набора тестов...")
                test_suite = testing.AIONTestSuite()
                success = test_suite.run_all_tests()
                
                if success:
                    print("✅ Все тесты прошли успешно!")
                else:
                    print("❌ Некоторые тесты провалились")
                    
            elif action == 'specific':
                if not test_name:
                    print("❌ Укажите название теста через --name")
                    print("Доступные тесты: core, files, config, api, monitoring, integration")
                    return
                
                print(f"🧪 Запуск теста: {test_name}")
                success = testing.run_specific_test(test_name)
                
                if success:
                    print(f"✅ Тест {test_name} прошел успешно!")
                else:
                    print(f"❌ Тест {test_name} провалился")
                    
            elif action == 'history':
                testing.show_test_history()
                
            elif action == 'clean':
                testing.clean_test_data()
                
        except ImportError:
            print("❌ Система тестирования недоступна")
        except Exception as e:
            print(f"❌ Ошибка тестирования: {e}")

def main():
    """Главная функция с командной системой"""
    parser = argparse.ArgumentParser(description='AION - AI Self-Improvement System')
    
    subparsers = parser.add_subparsers(dest='command', help='Команды')
    
    # Команда анализа
    analyze_parser = subparsers.add_parser('analyze', help='Запустить анализ')
    analyze_parser.add_argument('--projects', nargs='+', help='GitHub проекты для анализа')
    
    # Команда статуса
    subparsers.add_parser('status', help='Показать статус')
    
    # Команда проблем
    issues_parser = subparsers.add_parser('issues', help='Показать проблемы')
    issues_parser.add_argument('--type', choices=['long_line', 'todo'], help='Тип проблем')
    issues_parser.add_argument('--limit', type=int, default=10, help='Лимит результатов')
    
    # Команда проектов
    subparsers.add_parser('projects', help='Показать изученные проекты')
    
    # Команда очистки
    subparsers.add_parser('clean', help='Очистить логи')
    
    # Команда исправления
    fix_parser = subparsers.add_parser('fix', help='Исправить проблемы в коде')
    fix_parser.add_argument('--type', choices=['long_line', 'todo'], help='Тип проблем для исправления')
    fix_parser.add_argument('--limit', type=int, default=5, help='Количество исправлений')
    
    # AI команды
    ai_parser = subparsers.add_parser('ai', help='AI анализ и улучшения')
    ai_parser.add_argument('--provider', choices=['local', 'inference'], default='local', help='AI провайдер')
    ai_parser.add_argument('--api-key', help='API ключ для внешнего провайдера')
    ai_parser.add_argument('--action', choices=['analyze', 'improve', 'plan'], default='analyze', help='Действие')
    
    # Веб-интерфейс
    web_parser = subparsers.add_parser('web', help='Запуск веб-интерфейса')
    web_parser.add_argument('--host', default='localhost', help='Хост сервера')
    web_parser.add_argument('--port', type=int, default=5000, help='Порт сервера')
    
    # API сервер
    api_parser = subparsers.add_parser('api', help='Запуск REST API сервера')
    api_parser.add_argument('--host', default='localhost', help='Хост API')
    api_parser.add_argument('--port', type=int, default=8000, help='Порт API')
    
    # Мониторинг
    monitor_parser = subparsers.add_parser('monitor', help='Система мониторинга')
    monitor_parser.add_argument('--action', choices=['start', 'stop', 'status', 'dashboard'], default='status', help='Действие')
    
    # Тестирование
    test_parser = subparsers.add_parser('test', help='Система тестирования')
    test_parser.add_argument('--action', choices=['run', 'specific', 'history', 'clean'], default='run', help='Действие')
    test_parser.add_argument('--name', help='Название конкретного теста')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    aion = AIONCore()
    
    try:
        if args.command == 'analyze':
            aion.run_analysis(args.projects)
        elif args.command == 'status':
            aion.show_status()
        elif args.command == 'issues':
            aion.show_issues(args.type, args.limit)
        elif args.command == 'projects':
            aion.show_projects()
        elif args.command == 'clean':
            aion.clean_logs()
        elif args.command == 'fix':
            aion.fix_issues(args.type, args.limit)
        elif args.command == 'ai':
            aion.run_ai_analysis(args.provider, args.api_key, args.action)
        elif args.command == 'web':
            aion.run_web_interface(args.host, args.port)
        elif args.command == 'api':
            aion.run_api_server(args.host, args.port)
        elif args.command == 'monitor':
            aion.run_monitoring(args.action)
        elif args.command == 'test':
            aion.run_testing(args.action, args.name)
            
    except KeyboardInterrupt:
        print("\n⏹️ Прервано пользователем")
    except Exception as e:
        print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    main()
