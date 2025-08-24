#!/usr/bin/env python3
"""
GitHub Integration - Интеграция с GitHub API
"""

import logging
import os
import base64
from typing import Dict, List, Optional
import aiohttp

logger = logging.getLogger(__name__)

class GitHubIntegration:
    """Интеграция с GitHub API"""
    
    def __init__(self):
        self.github_token = os.getenv("GITHUB_TOKEN", "")
        self.github_api_url = "https://api.github.com"
        self.repositories = []
        
    async def get_repositories(self, username: str = None) -> List[Dict]:
        """Получение репозиториев с GitHub"""
        try:
            headers = {}
            if self.github_token:
                headers["Authorization"] = f"token {self.github_token}"
            
            url = f"{self.github_api_url}/user/repos"
            if username:
                url = f"{self.github_api_url}/users/{username}/repos"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        repos = await response.json()
                        self.repositories = repos[:10]  # Топ 10 репозиториев
                        return repos
                    else:
                        logger.error(f"GitHub API error: {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Ошибка получения репозиториев: {e}")
            return []
    
    async def get_repository_content(self, owner: str, repo: str, path: str = "") -> Dict:
        """Получение содержимого файлов репозитория"""
        try:
            headers = {}
            if self.github_token:
                headers["Authorization"] = f"token {self.github_token}"
            
            url = f"{self.github_api_url}/repos/{owner}/{repo}/contents/{path}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        content = await response.json()
                        return content
                    else:
                        logger.error(f"Ошибка получения контента: {response.status}")
                        return {}
        except Exception as e:
            logger.error(f"Ошибка получения контента: {e}")
            return {}
    
    async def get_file_content(self, owner: str, repo: str, path: str) -> str:
        """Получение содержимого конкретного файла"""
        try:
            content = await self.get_repository_content(owner, repo, path)
            if content and 'content' in content:
                # Декодируем base64 контент
                decoded_content = base64.b64decode(content['content']).decode('utf-8')
                return decoded_content
            return ""
        except Exception as e:
            logger.error(f"Ошибка получения файла: {e}")
            return ""
    
    async def search_repositories(self, query: str, language: str = None) -> List[Dict]:
        """Поиск репозиториев"""
        try:
            headers = {}
            if self.github_token:
                headers["Authorization"] = f"token {self.github_token}"
            
            url = f"{self.github_api_url}/search/repositories"
            params = {"q": query}
            if language:
                params["q"] += f" language:{language}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get('items', [])
                    else:
                        logger.error(f"Ошибка поиска: {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Ошибка поиска репозиториев: {e}")
            return []
    
    def get_metrics(self) -> Dict:
        """Получение метрик GitHub интеграции"""
        return {
            'github_integration': bool(self.github_token),
            'repositories_count': len(self.repositories),
            'api_status': 'active' if self.github_token else 'inactive'
        }
