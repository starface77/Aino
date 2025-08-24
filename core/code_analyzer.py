#!/usr/bin/env python3
"""
Code Analyzer - Анализ и улучшение кода
"""

import logging
import json
import re
from datetime import datetime
from typing import Dict, List, Optional
import asyncio

logger = logging.getLogger(__name__)

class CodeAnalyzer:
    """Анализатор и улучшатель кода"""
    
    def __init__(self, ai_system):
        self.ai_system = ai_system
        self.code_improvements = []
        self.last_improvement = None
        
    async def analyze_code_quality(self, code: str, language: str = "python") -> Dict:
        """Анализ качества кода"""
        try:
            analysis_prompt = f"""
            Проанализируй качество кода на {language}:
            
            {code}
            
            Предоставь анализ в формате JSON:
            {{
                "quality_score": 0-100,
                "issues": ["список проблем"],
                "improvements": ["список улучшений"],
                "complexity": "low/medium/high",
                "maintainability": "low/medium/high"
            }}
            """
            
            response = await self.ai_system.call_gemma_api(analysis_prompt)
            
            # Пытаемся извлечь JSON из ответа
            try:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                else:
                    return {
                        "quality_score": 70,
                        "issues": ["Не удалось проанализировать"],
                        "improvements": ["Требуется ручной анализ"],
                        "complexity": "medium",
                        "maintainability": "medium"
                    }
            except:
                return {
                    "quality_score": 70,
                    "issues": ["Ошибка парсинга"],
                    "improvements": ["Требуется ручной анализ"],
                    "complexity": "medium",
                    "maintainability": "medium"
                }
        except Exception as e:
            logger.error(f"Ошибка анализа кода: {e}")
            return {
                "quality_score": 50,
                "issues": ["Ошибка анализа"],
                "improvements": ["Проверьте код вручную"],
                "complexity": "unknown",
                "maintainability": "unknown"
            }
    
    async def suggest_code_improvements(self, code: str, language: str = "python") -> str:
        """Предложение улучшений кода"""
        try:
            improvement_prompt = f"""
            Предложи улучшения для кода на {language}:
            
            {code}
            
            Предоставь:
            1. Конкретные улучшения с примерами
            2. Оптимизации производительности
            3. Улучшения читаемости
            4. Современные практики
            5. Исправление потенциальных ошибок
            """
            
            return await self.ai_system.call_gemma_api(improvement_prompt)
        except Exception as e:
            logger.error(f"Ошибка предложения улучшений: {e}")
            return "Не удалось предложить улучшения"
    
    async def auto_improve_code(self, file_path: str = None):
        """Автоматическое улучшение кода"""
        try:
            logger.info("🔧 Запуск автоматического улучшения кода...")
            
            # Анализируем текущий код
            if file_path is None:
                file_path = __file__
            
            with open(file_path, 'r', encoding='utf-8') as f:
                current_code = f.read()
            
            # Анализируем качество
            quality = await self.analyze_code_quality(current_code, "python")
            
            # Если качество ниже 80, предлагаем улучшения
            if quality.get("quality_score", 0) < 80:
                improvements = await self.suggest_code_improvements(current_code, "python")
                
                # Сохраняем улучшения
                improvement_data = {
                    "timestamp": datetime.now().isoformat(),
                    "quality_score": quality.get("quality_score", 0),
                    "improvements": improvements,
                    "file": file_path
                }
                
                self.code_improvements.append(improvement_data)
                self.last_improvement = datetime.now()
                
                logger.info(f"🔧 Предложены улучшения для {file_path}")
                logger.info(f"📊 Качество кода: {quality.get('quality_score', 0)}%")
            
        except Exception as e:
            logger.error(f"Ошибка автоматического улучшения: {e}")
    
    def get_improvements_history(self) -> List[Dict]:
        """Получение истории улучшений"""
        return self.code_improvements
    
    def get_metrics(self) -> Dict:
        """Получение метрик анализатора"""
        return {
            'improvements_count': len(self.code_improvements),
            'last_improvement': self.last_improvement.isoformat() if self.last_improvement else None,
            'analyzer_status': 'active'
        }
