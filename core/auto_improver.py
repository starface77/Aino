#!/usr/bin/env python3
"""
Auto Improver - Автоматическое самоулучшение системы
"""

import logging
import time
import threading
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List
import json

logger = logging.getLogger(__name__)

class AutoImprover:
    """Система автоматического самоулучшения"""
    
    def __init__(self, ai_system, code_analyzer):
        self.ai_system = ai_system
        self.code_analyzer = code_analyzer
        self.auto_improvement_enabled = True
        self.improvement_interval = 3600  # 1 час
        self.improvement_thread = None
        self.improvement_history = []
        
    def start_auto_improvement(self):
        """Запуск фонового процесса самоулучшения"""
        if self.auto_improvement_enabled and not self.improvement_thread:
            self.improvement_thread = threading.Thread(target=self._improvement_loop, daemon=True)
            self.improvement_thread.start()
            logger.info("🔄 Запущен процесс автоматического улучшения кода")
    
    def stop_auto_improvement(self):
        """Остановка процесса самоулучшения"""
        self.auto_improvement_enabled = False
        if self.improvement_thread:
            self.improvement_thread.join(timeout=5)
        logger.info("🛑 Процесс автоматического улучшения остановлен")
    
    def _improvement_loop(self):
        """Цикл автоматического улучшения"""
        while self.auto_improvement_enabled:
            try:
                asyncio.run(self._run_improvement_cycle())
                time.sleep(self.improvement_interval)
            except Exception as e:
                logger.error(f"Ошибка в цикле улучшения: {e}")
                time.sleep(60)  # Пауза 1 минута при ошибке
    
    async def _run_improvement_cycle(self):
        """Выполнение цикла улучшения"""
        try:
            logger.info("🔧 Выполнение цикла автоматического улучшения...")
            
            # Улучшаем основные файлы системы
            files_to_improve = [
                "ai/tendo-aino/run.py",
                "ai/tendo-aino/core/ai_system.py",
                "ai/tendo-aino/core/github_integration.py",
                "ai/tendo-aino/core/code_analyzer.py"
            ]
            
            for file_path in files_to_improve:
                try:
                    await self.code_analyzer.auto_improve_code(file_path)
                except Exception as e:
                    logger.error(f"Ошибка улучшения {file_path}: {e}")
            
            # Анализируем производительность системы
            await self._analyze_system_performance()
            
            # Сохраняем историю улучшений
            improvement_record = {
                "timestamp": datetime.now().isoformat(),
                "files_improved": len(files_to_improve),
                "system_metrics": self.ai_system.get_metrics()
            }
            self.improvement_history.append(improvement_record)
            
            logger.info("✅ Цикл автоматического улучшения завершен")
            
        except Exception as e:
            logger.error(f"Ошибка цикла улучшения: {e}")
    
    async def _analyze_system_performance(self):
        """Анализ производительности системы"""
        try:
            metrics = self.ai_system.get_metrics()
            
            # Если производительность низкая, предлагаем оптимизации
            if metrics.get('overall_performance', 1.0) < 0.8:
                optimization_prompt = f"""
                Проанализируй метрики системы и предложи оптимизации:
                
                {json.dumps(metrics, indent=2)}
                
                Предложи конкретные улучшения для повышения производительности.
                """
                
                optimization_suggestions = await self.ai_system.call_gemma_api(optimization_prompt)
                logger.info(f"🔧 Предложения по оптимизации: {optimization_suggestions[:200]}...")
                
        except Exception as e:
            logger.error(f"Ошибка анализа производительности: {e}")
    
    def get_improvement_history(self) -> List[Dict]:
        """Получение истории улучшений"""
        return self.improvement_history
    
    def get_metrics(self) -> Dict:
        """Получение метрик самоулучшения"""
        return {
            'auto_improvement_enabled': self.auto_improvement_enabled,
            'improvement_interval': self.improvement_interval,
            'improvement_history_count': len(self.improvement_history),
            'last_improvement_cycle': self.improvement_history[-1]['timestamp'] if self.improvement_history else None,
            'improver_status': 'active' if self.auto_improvement_enabled else 'inactive'
        }
