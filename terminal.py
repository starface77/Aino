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
        print(f"\nСеанс завершен")
        print(f"Выполнено команд: {len(self.commands_history)}")
        self.running = False
    
    def _process_command(self, user_input):
        """Обработка команды пользователя"""
        
        cmd = user_input.lower().strip()
        
        print(f"[EXEC] {user_input}")
        time.sleep(0.3)
        
        # Выход
        if any(word in cmd for word in ['exit', 'quit', 'выход']):
            self._print_goodbye()
            return
        
        # Помощь
        elif any(word in cmd for word in ['help', 'commands']):
            self._show_help()
        
        # Анализ проекта
        elif 'analyze' in cmd or 'анализ' in cmd:
            self._analyze_project(user_input)
        
        # Статус
        elif 'status' in cmd or 'статус' in cmd:
            self._show_status()
        
        # Проблемы
        elif 'issues' in cmd or 'проблемы' in cmd:
            self._show_issues(user_input)
        
        # Исправления
        elif 'fix' in cmd or 'исправь' in cmd:
            self._fix_issues(user_input)
        
        # Проекты GitHub
        elif 'projects' in cmd or 'list' in cmd:
            self._show_projects()
        
        # Очистка
        elif 'clean' in cmd or 'очисти' in cmd:
            self._clean_logs()
        
        else:
            self._unknown_command(user_input)
    
    def _show_help(self):
        """Показать справку"""
        print("AVAILABLE COMMANDS:")
        print()
        print("ANALYSIS:")
        print("  analyze project     - запустить полный анализ")
        print("  analyze autogpt     - анализ конкретного проекта")
        print()
        print("STATUS:")
        print("  show status         - текущее состояние")
        print("  list projects       - изученные проекты")
        print()
        print("FIXES:")
        print("  show issues         - список проблем")
        print("  fix issues          - исправить проблемы")
        print("  fix long lines      - исправить длинные строки")
        print()
        print("UTILS:")
        print("  clean logs          - очистить логи")
        print("  help                - эта справка")
        print("  exit                - выход")
    
    def _analyze_project(self, user_input):
        """Анализ проекта"""
        print("[ANALYSIS] Starting project analysis...")
        
        projects = None
        if 'autogpt' in user_input.lower():
            projects = ['AutoGPT']
        elif 'pytorch' in user_input.lower():
            projects = ['pytorch']
        elif 'transformers' in user_input.lower():
            projects = ['transformers']
        elif 'langchain' in user_input.lower():
            projects = ['langchain']
        
        self._run_aion_command('analyze', projects)
    
    def _show_status(self):
        """Показать статус"""
        print("[STATUS] Checking system status...")
        self._run_aion_command('status')
    
    def _show_issues(self, user_input):
        """Показать проблемы"""
        print("[ISSUES] Scanning for code issues...")
        
        issue_type = None
        limit = 10
        
        if 'long' in user_input.lower():
            issue_type = 'long_line'
        elif 'todo' in user_input.lower():
            issue_type = 'todo'
        
        words = user_input.split()
        for word in words:
            if word.isdigit():
                limit = int(word)
                break
        
        self._run_aion_command('issues', None, issue_type, limit)
    
    def _fix_issues(self, user_input):
        """Исправить проблемы"""
        print("[FIX] Applying automatic fixes...")
        
        issue_type = None
        limit = 5
        
        if 'long' in user_input.lower():
            issue_type = 'long_line'
        elif 'todo' in user_input.lower():
            issue_type = 'todo'
        
        words = user_input.split()
        for word in words:
            if word.isdigit():
                limit = int(word)
                break
        
        self._run_aion_command('fix', None, issue_type, limit)
    
    def _show_projects(self):
        """Показать проекты"""
        print("[PROJECTS] Loading GitHub projects...")
        self._run_aion_command('projects')
    
    def _clean_logs(self):
        """Очистить логи"""
        print("[CLEAN] Removing log files...")
        self._run_aion_command('clean')
    
    def _unknown_command(self, user_input):
        """Неизвестная команда"""
        print(f"[ERROR] Unknown command: {user_input}")
        print("Type 'help' for available commands")
    
    def _run_aion_command(self, command, projects=None, issue_type=None, limit=None):
        """Запуск команды AION"""
        try:
            cmd = ['python', 'aion.py', command]
            
            if command == 'analyze' and projects:
                cmd.extend(['--projects'] + projects)
            
            if command in ['issues', 'fix']:
                if issue_type:
                    cmd.extend(['--type', issue_type])
                if limit:
                    cmd.extend(['--limit', str(limit)])
            
            print(f"[COMMAND] {' '.join(cmd)}")
            print("-" * 30)
            
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            
            if result.returncode == 0:
                print(result.stdout)
                print("[SUCCESS] Command completed")
            else:
                print("[ERROR] Command failed:")
                print(result.stderr)
                
        except Exception as e:
            print(f"[ERROR] Failed to execute: {e}")

def main():
    """Главная функция"""
    terminal = AIONTerminal()
    terminal.start()

if __name__ == "__main__":
    main()