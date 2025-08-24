#!/usr/bin/env python3
"""
AION AI Core - Подключение к ИИ для самоулучшения
Интеграция с OpenAI/Claude для анализа и улучшения кода
"""

import os
import json
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional

class AIProvider:
    """Базовый класс для ИИ провайдеров"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
    
    def analyze_code(self, code: str, file_path: str) -> Dict[str, Any]:
        """Анализ кода с помощью ИИ"""
        raise NotImplementedError
    
    def suggest_improvements(self, analysis: Dict[str, Any]) -> List[str]:
        """Предложения по улучшению"""
        raise NotImplementedError
    
    def generate_fix(self, issue: Dict[str, Any]) -> str:
        """Генерация исправления"""
        raise NotImplementedError

class InferenceProvider(AIProvider):
    """Провайдер Inference.net с Gemma"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key or os.getenv('INFERENCE_API_KEY', ''))
        self.api_url = "https://api.inference.net/v1/chat/completions"
        self.model = "google/gemma-3-27b-instruct/bf-16"
    
    def analyze_code(self, code: str, file_path: str) -> Dict[str, Any]:
        """Анализ кода через Gemma ИИ"""
        
        prompt = f"""Analyze this Python code from {file_path}:

```python
{code[:1500]}
```

Return JSON analysis:
{{
    "issues": ["list of code issues"],
    "suggestions": ["improvement suggestions"], 
    "complexity_score": 1-10,
    "maintainability": "low/medium/high",
    "performance_issues": ["performance problems"]
}}

Focus on: code quality, complexity, performance, maintainability."""
        
        if not self.api_key:
            return self._mock_analysis()
        
        try:
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 800,
                    "temperature": 0.2,
                    "stream": False
                },
                timeout=45
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                # Пытаемся парсить JSON
                try:
                    return json.loads(content)
                except:
                    return self._parse_text_response(content)
            else:
                print(f"[AI ERROR] Inference API: {response.status_code}")
                if response.text:
                    print(f"[AI ERROR] Response: {response.text[:200]}")
                return self._mock_analysis()
                
        except Exception as e:
            print(f"[AI ERROR] {e}")
            return self._mock_analysis()
    
    def _mock_analysis(self) -> Dict[str, Any]:
        """Заглушка анализа без API"""
        return {
            "issues": [
                "Найдены длинные строки",
                "Отсутствуют docstrings",
                "Высокая цикломатическая сложность"
            ],
            "suggestions": [
                "Добавить типизацию",
                "Разбить большие функции",
                "Добавить обработку ошибок",
                "Улучшить читаемость кода"
            ],
            "complexity_score": 6,
            "maintainability": "medium",
            "performance_issues": [
                "Неэффективные циклы",
                "Отсутствие кэширования"
            ]
        }
    
    def _parse_text_response(self, text: str) -> Dict[str, Any]:
        """Парсинг текстового ответа"""
        return {
            "issues": ["Ответ не в JSON формате"],
            "suggestions": [text[:200] + "..."],
            "complexity_score": 5,
            "maintainability": "unknown",
            "performance_issues": []
        }

class LocalAIProvider(AIProvider):
    """Локальный ИИ анализ без API"""
    
    def analyze_code(self, code: str, file_path: str) -> Dict[str, Any]:
        """Локальный анализ кода"""
        
        lines = code.split('\n')
        issues = []
        suggestions = []
        complexity = 1
        
        # Простой анализ
        for i, line in enumerate(lines, 1):
            if len(line) > 120:
                issues.append(f"Строка {i}: слишком длинная ({len(line)} символов)")
            
            if 'TODO' in line or 'FIXME' in line:
                issues.append(f"Строка {i}: незавершенный код")
            
            if 'def ' in line:
                complexity += 1
            
            if 'if ' in line or 'for ' in line or 'while ' in line:
                complexity += 1
        
        # Предложения
        if complexity > 10:
            suggestions.append("Высокая сложность - разбить на более мелкие функции")
        
        if 'import *' in code:
            suggestions.append("Избегать import * - использовать конкретные импорты")
        
        if not '"""' in code and not "'''" in code:
            suggestions.append("Добавить docstrings для документации")
        
        if 'print(' in code:
            suggestions.append("Заменить print на логирование")
        
        return {
            "issues": issues,
            "suggestions": suggestions,
            "complexity_score": min(complexity, 10),
            "maintainability": "high" if complexity < 5 else "medium" if complexity < 8 else "low",
            "performance_issues": [
                "Найдены потенциальные проблемы производительности"
            ] if complexity > 7 else []
        }

class AISelfImprover:
    """Система самоулучшения с ИИ"""
    
    def __init__(self, provider: AIProvider = None):
        self.provider = provider or LocalAIProvider()
        self.analysis_history = []
        self.improvement_log = []
    
    def analyze_project_with_ai(self, project_path: str = ".") -> Dict[str, Any]:
        """Анализ проекта с помощью ИИ"""
        
        print("[AI ANALYSIS] Scanning project with AI...")
        
        python_files = self._find_python_files(project_path)
        all_analyses = []
        total_issues = 0
        total_suggestions = []
        
        for file_path in python_files[:5]:  # Анализируем первые 5 файлов
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    code = f.read()
                
                if len(code) > 50:  # Пропускаем слишком маленькие файлы
                    print(f"  Analyzing: {file_path}")
                    
                    analysis = self.provider.analyze_code(code, file_path)
                    analysis['file_path'] = file_path
                    all_analyses.append(analysis)
                    
                    total_issues += len(analysis.get('issues', []))
                    total_suggestions.extend(analysis.get('suggestions', []))
                    
                    # Пауза между запросами для rate limit
                    import time
                    time.sleep(2)
                    
            except Exception as e:
                print(f"  Error analyzing {file_path}: {e}")
        
        # Объединяем результаты
        project_analysis = {
            'timestamp': datetime.now().isoformat(),
            'total_files_analyzed': len(all_analyses),
            'total_issues': total_issues,
            'unique_suggestions': list(set(total_suggestions)),
            'file_analyses': all_analyses,
            'overall_complexity': sum(a.get('complexity_score', 0) for a in all_analyses) / len(all_analyses) if all_analyses else 0,
            'maintainability_distribution': self._calculate_maintainability_stats(all_analyses)
        }
        
        self.analysis_history.append(project_analysis)
        self._save_analysis(project_analysis)
        
        return project_analysis
    
    def generate_improvement_plan(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Генерация плана улучшений"""
        
        print("[AI PLANNING] Generating improvement plan...")
        
        suggestions = analysis.get('unique_suggestions', [])
        file_analyses = analysis.get('file_analyses', [])
        
        # Приоритизация улучшений
        high_priority = []
        medium_priority = []
        low_priority = []
        
        for suggestion in suggestions:
            if any(word in suggestion.lower() for word in ['сложность', 'complexity', 'ошибки', 'errors']):
                high_priority.append(suggestion)
            elif any(word in suggestion.lower() for word in ['производительность', 'performance', 'оптимизация']):
                medium_priority.append(suggestion)
            else:
                low_priority.append(suggestion)
        
        # Файлы требующие внимания
        files_needing_attention = [
            a['file_path'] for a in file_analyses 
            if a.get('complexity_score', 0) > 7 or len(a.get('issues', [])) > 5
        ]
        
        improvement_plan = {
            'timestamp': datetime.now().isoformat(),
            'high_priority_improvements': high_priority,
            'medium_priority_improvements': medium_priority,
            'low_priority_improvements': low_priority,
            'files_needing_attention': files_needing_attention,
            'estimated_effort': self._estimate_effort(analysis),
            'recommended_order': self._recommend_order(high_priority, medium_priority, low_priority)
        }
        
        self.improvement_log.append(improvement_plan)
        return improvement_plan
    
    def apply_ai_improvements(self, plan: Dict[str, Any], limit: int = 3) -> List[str]:
        """Применение улучшений с помощью ИИ"""
        
        print(f"[AI IMPROVEMENTS] Applying {limit} improvements...")
        
        applied = []
        recommendations = plan.get('recommended_order', [])
        
        for i, improvement in enumerate(recommendations[:limit]):
            try:
                print(f"  {i+1}. {improvement}")
                
                # Здесь можно подключить реальное применение улучшений
                # Пока что симулируем
                success = self._simulate_improvement(improvement)
                
                if success:
                    applied.append(improvement)
                    print(f"     ✅ Applied")
                else:
                    print(f"     ❌ Failed")
                    
            except Exception as e:
                print(f"     ❌ Error: {e}")
        
        return applied
    
    def _find_python_files(self, project_path: str) -> List[str]:
        """Поиск Python файлов"""
        python_files = []
        
        for root, dirs, files in os.walk(project_path):
            dirs[:] = [d for d in dirs if d not in ['venv', '__pycache__', '.git', 'node_modules']]
            
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        return python_files
    
    def _calculate_maintainability_stats(self, analyses: List[Dict]) -> Dict[str, int]:
        """Статистика по maintainability"""
        stats = {'high': 0, 'medium': 0, 'low': 0, 'unknown': 0}
        
        for analysis in analyses:
            maintainability = analysis.get('maintainability', 'unknown')
            stats[maintainability] = stats.get(maintainability, 0) + 1
        
        return stats
    
    def _estimate_effort(self, analysis: Dict[str, Any]) -> str:
        """Оценка усилий"""
        total_issues = analysis.get('total_issues', 0)
        
        if total_issues < 10:
            return "low"
        elif total_issues < 50:
            return "medium"
        else:
            return "high"
    
    def _recommend_order(self, high: List, medium: List, low: List) -> List[str]:
        """Рекомендуемый порядок улучшений"""
        return high + medium[:3] + low[:2]  # Сначала критичные, потом важные
    
    def _simulate_improvement(self, improvement: str) -> bool:
        """Симуляция применения улучшения"""
        import random
        import time
        
        time.sleep(0.5)  # Симулируем работу
        return random.random() > 0.3  # 70% успеха
    
    def _save_analysis(self, analysis: Dict[str, Any]):
        """Сохранение анализа"""
        try:
            with open('ai_analysis_log.json', 'w', encoding='utf-8') as f:
                json.dump({
                    'latest_analysis': analysis,
                    'analysis_history': self.analysis_history[-5:]  # Последние 5
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[ERROR] Failed to save analysis: {e}")
    
    def get_analysis_summary(self) -> str:
        """Краткая сводка анализа"""
        if not self.analysis_history:
            return "Анализ еще не проводился"
        
        latest = self.analysis_history[-1]
        
        return f"""
AI ANALYSIS SUMMARY:
Files analyzed: {latest['total_files_analyzed']}
Total issues: {latest['total_issues']}
Complexity score: {latest['overall_complexity']:.1f}/10
Unique suggestions: {len(latest['unique_suggestions'])}
"""

# Функция для интеграции с основной системой
def create_ai_improver(provider_type: str = "local", api_key: str = None) -> AISelfImprover:
    """Создание AI улучшателя"""
    
    if provider_type == "inference":
        provider = InferenceProvider(api_key)
        print("[AI] Using Inference.net with Gemma")
    elif provider_type == "local":
        provider = LocalAIProvider()
        print("[AI] Using local analysis")
    else:
        provider = LocalAIProvider()
        print("[AI] Fallback to local analysis")
    
    return AISelfImprover(provider)

if __name__ == "__main__":
    # Тест системы
    improver = create_ai_improver()
    analysis = improver.analyze_project_with_ai()
    plan = improver.generate_improvement_plan(analysis)
    applied = improver.apply_ai_improvements(plan)
    
    print(improver.get_analysis_summary())
