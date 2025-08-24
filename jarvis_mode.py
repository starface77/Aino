#!/usr/bin/env python3
"""
AION Interactive Terminal
Интерактивная командная система
"""

import os
import sys
import time
import subprocess
from datetime import datetime
import json

class AIONTerminal:
    """Интерактивная командная система AION"""
    
    def __init__(self):
        self.name = "AION"
        self.running = True
        self.context = {}
        self.commands_history = []
        
    def start(self):
        """Запуск интерактивного режима"""
        self._print_welcome()
        
        while self.running:
            try:
                user_input = input(f"\n{self.name}> ").strip()
                
                if not user_input:
                    continue
                
                self.commands_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'input': user_input
                })
                
                self._process_command(user_input)
                
            except KeyboardInterrupt:
                self._print_goodbye()
                break
            except EOFError:
                self._print_goodbye()
                break
    
    def _print_welcome(self):
        """Приветствие"""
        print("⚡ AION Terminal v1.0")
        print("=" * 40)
        print("Интерактивная система анализа кода")
        print("Введите команду для выполнения")
        print()
        print("Команды:")
        print("analyze project - анализ проекта")
        print("show status - текущий статус") 
        print("fix issues - исправить проблемы")
        print("list projects - GitHub проекты")
        print("help - справка")
        print("exit - выход")
        print("=" * 40)
    
    def _print_goodbye(self):
        """Прощание"""
        print(f"\n👋 До свидания! Было приятно работать.")
        print(f"📊 Выполнено команд: {len(self.commands_history)}")
        self.running = False
    
    def _process_command(self, user_input):
        """Обработка команды пользователя"""
        
        # Нормализуем ввод
        cmd = user_input.lower().strip()
        
        print(f"[EXEC] {user_input}")
        time.sleep(0.3)
        
        # Выход
        if any(word in cmd for word in ['выход', 'exit', 'quit', 'пока', 'стоп']):
            self._print_goodbye()
            return
        
        # Помощь
        elif any(word in cmd for word in ['помощь', 'help', 'команды', 'что умеешь']):
            self._show_help()
        
        # Анализ проекта
        elif any(word in cmd for word in ['анализ', 'проанализируй', 'analyze', 'проверь проект']):
            self._analyze_project(user_input)
        
        # Статус
        elif any(word in cmd for word in ['статус', 'status', 'состояние', 'как дела']):
            self._show_status()
        
        # Проблемы
        elif any(word in cmd for word in ['проблемы', 'issues', 'ошибки', 'баги']):
            self._show_issues(user_input)
        
        # Исправления
        elif any(word in cmd for word in ['исправь', 'fix', 'почини', 'реши']):
            self._fix_issues(user_input)
        
        # Проекты GitHub
        elif any(word in cmd for word in ['проекты', 'projects', 'github', 'топ']):
            self._show_projects()
        
        # Очистка
        elif any(word in cmd for word in ['очисти', 'clean', 'удали логи']):
            self._clean_logs()
        
        # Запуск анализа конкретных проектов
        elif any(word in cmd for word in ['autogpt', 'pytorch', 'transformers', 'langchain']):
            self._analyze_specific_projects(user_input)
        
        # Неизвестная команда
        else:
            self._unknown_command(user_input)
    
    def _show_help(self):
        """Показать справку"""
        print("💡 Вот что я умею:")
        print()
        print("📊 АНАЛИЗ:")
        print("  'проанализируй проект' - полный анализ кода и GitHub")
        print("  'проверь autogpt' - анализ конкретного проекта")
        print()
        print("📈 СТАТУС:")
        print("  'покажи статус' - текущее состояние")
        print("  'как дела' - краткая сводка")
        print()
        print("🔧 ИСПРАВЛЕНИЯ:")
        print("  'покажи проблемы' - список найденных проблем")
        print("  'исправь проблемы' - автоматическое исправление")
        print("  'исправь только длинные строки' - конкретный тип")
        print()
        print("🌟 ПРОЕКТЫ:")
        print("  'покажи проекты' - изученные GitHub проекты")
        print()
        print("🗑️ УТИЛИТЫ:")
        print("  'очисти логи' - удалить все логи")
        print()
        print("❓ ОБЩЕЕ:")
        print("  'помощь' - эта справка")
        print("  'выход' - завершить работу")
    
    def _analyze_project(self, user_input):
        """Анализ проекта"""
        print("🚀 Запускаю полный анализ проекта...")
        
        # Определяем какие проекты анализировать
        projects = None
        if 'autogpt' in user_input.lower():
            projects = ['AutoGPT']
        elif 'pytorch' in user_input.lower():
            projects = ['pytorch']
        elif 'transformers' in user_input.lower():
            projects = ['transformers']
        elif 'langchain' in user_input.lower():
            projects = ['langchain']
        
        # Запускаем анализ
        self._run_aion_command('analyze', projects)
    
    def _analyze_specific_projects(self, user_input):
        """Анализ конкретных проектов"""
        projects = []
        
        if 'autogpt' in user_input.lower():
            projects.append('AutoGPT')
        if 'pytorch' in user_input.lower():
            projects.append('pytorch')
        if 'transformers' in user_input.lower():
            projects.append('transformers')
        if 'langchain' in user_input.lower():
            projects.append('langchain')
        
        if projects:
            print(f"🎯 Анализирую проекты: {', '.join(projects)}")
            self._run_aion_command('analyze', projects)
        else:
            print("❌ Не понял какие проекты анализировать")
    
    def _show_status(self):
        """Показать статус"""
        print("📊 Проверяю текущий статус...")
        self._run_aion_command('status')
    
    def _show_issues(self, user_input):
        """Показать проблемы"""
        print("⚠️ Ищу проблемы в коде...")
        
        # Определяем тип проблем
        issue_type = None
        limit = 10
        
        if 'длинн' in user_input.lower() or 'long' in user_input.lower():
            issue_type = 'long_line'
            print("🔍 Фокусируюсь на длинных строках")
        elif 'todo' in user_input.lower() or 'туду' in user_input.lower():
            issue_type = 'todo'
            print("🔍 Фокусируюсь на TODO комментариях")
        
        # Ищем лимит
        words = user_input.split()
        for word in words:
            if word.isdigit():
                limit = int(word)
                print(f"🔢 Показываю {limit} проблем")
                break
        
        self._run_aion_command('issues', None, issue_type, limit)
    
    def _fix_issues(self, user_input):
        """Исправить проблемы"""
        print("🔧 Исправляю найденные проблемы...")
        
        # Определяем что исправлять
        issue_type = None
        limit = 5
        
        if 'длинн' in user_input.lower() or 'long' in user_input.lower():
            issue_type = 'long_line'
            print("🎯 Исправляю длинные строки")
        elif 'todo' in user_input.lower() or 'туду' in user_input.lower():
            issue_type = 'todo'
            print("🎯 Обрабатываю TODO комментарии")
        else:
            print("🎯 Исправляю все типы проблем")
        
        # Ищем количество
        words = user_input.split()
        for word in words:
            if word.isdigit():
                limit = int(word)
                print(f"🔢 Исправляю {limit} проблем")
                break
        
        self._run_aion_command('fix', None, issue_type, limit)
    
    def _show_projects(self):
        """Показать проекты"""
        print("🌟 Показываю изученные проекты...")
        self._run_aion_command('projects')
    
    def _clean_logs(self):
        """Очистить логи"""
        print("🗑️ Очищаю логи...")
        self._run_aion_command('clean')
    
    def _unknown_command(self, user_input):
        """Неизвестная команда"""
        print(f"🤷 Не понял команду: '{user_input}'")
        print("💡 Попробуйте:")
        print("  • 'помощь' - посмотреть все команды")
        print("  • 'проанализируй проект' - запустить анализ")
        print("  • 'покажи статус' - текущее состояние")
    
    def _run_aion_command(self, command, projects=None, issue_type=None, limit=None):
        """Запуск команды AION"""
        try:
            # Формируем команду
            cmd = ['python', 'aion.py', command]
            
            if command == 'analyze' and projects:
                cmd.extend(['--projects'] + projects)
            
            if command in ['issues', 'fix']:
                if issue_type:
                    cmd.extend(['--type', issue_type])
                if limit:
                    cmd.extend(['--limit', str(limit)])
            
            print(f"⚙️ Выполняю: {' '.join(cmd)}")
            print("-" * 40)
            
            # Запускаем команду
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            
            if result.returncode == 0:
                print(result.stdout)
                print("✅ Команда выполнена успешно")
            else:
                print("❌ Ошибка выполнения:")
                print(result.stderr)
                
        except Exception as e:
            print(f"❌ Ошибка запуска команды: {e}")

def main():
    """Главная функция"""
    terminal = AIONTerminal()
    terminal.start()

if __name__ == "__main__":
    main()
