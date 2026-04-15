import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    """全局配置"""

    def __init__(self):
        # 调试模式
        self.DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'

        # ===== DeepSeek LLM 配置 =====
        self.DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
        self.DEEPSEEK_API_BASE_URL = os.getenv('DEEPSEEK_API_BASE_URL', 'https://api.deepseek.com/v1')
        self.DEEPSEEK_MODEL = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')

        # ===== Neo4j 图数据库配置 =====
        self.NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.NEO4J_URL = os.getenv('NEO4J_URL')  # 兼容旧配置
        self.NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
        self.NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD',default='cznb6666')
        self.NEO4J_DATABASE = os.getenv('NEO4J_DATABASE', 'bishe')

        # ===== PostgreSQL 关系数据库配置 =====
        self.PG_HOST = os.getenv('PG_HOST', 'localhost')
        self.PG_PORT = int(os.getenv('PG_PORT', '5432'))
        self.PG_DATABASE = os.getenv('PG_DATABASE', 'bishe')
        self.PG_USER = os.getenv('PG_USER', 'cznb6666')
        self.PG_PASSWORD = os.getenv('PG_PASSWORD', 'cznb6666')

        # ===== 嵌入模型配置 =====
        # 本地模型: sentence-transformers
        self.EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'shibing624/text2vec-base-chinese')
        self.EMBEDDING_DIM = int(os.getenv('EMBEDDING_DIM', '768'))  # 向量维度

        # ===== 文件路径配置 =====
        self.SHPFILES_DIR = os.getenv('SHPFILES_DIR', './shpfiles')

        # ===== 高德地图API配置 =====
        self.AMAP_API_KEY = os.getenv('AMAP_API_KEY', 'c3526f90459691d155221dae78c84b7c')
        self.AMAP_REGION_ADCODE = os.getenv('AMAP_REGION_ADCODE', '420111')  # 洪山区默认

    def get_embedding_config(self) -> dict:
        """获取嵌入模型配置"""
        return {
            "model": self.EMBEDDING_MODEL,
            "dim": self.EMBEDDING_DIM
        }

    def get_neo4j_config(self) -> dict:
        """获取Neo4j连接配置"""
        uri = self.NEO4J_URI or self.NEO4J_URL
        return {
            "uri": uri,
            "user": self.NEO4J_USER,
            "password": self.NEO4J_PASSWORD
        }

    def get_postgres_config(self) -> dict:
        """获取PostgreSQL连接配置"""
        return {
            "host": self.PG_HOST,
            "port": self.PG_PORT,
            "database": self.PG_DATABASE,
            "user": self.PG_USER,
            "password": self.PG_PASSWORD
        }

    def get_llm_config(self) -> dict:
        """获取LLM配置"""
        return {
            "api_key": self.DEEPSEEK_API_KEY,
            "base_url": self.DEEPSEEK_API_BASE_URL,
            "model": self.DEEPSEEK_MODEL
        }

    def get_amap_config(self) -> dict:
        """获取高德地图API配置"""
        return {
            "api_key": self.AMAP_API_KEY,
            "region": self.AMAP_REGION_ADCODE
        }


# 全局配置实例
settings = Settings()
