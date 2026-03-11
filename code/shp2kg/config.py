import os

from dotenv import load_dotenv

load_dotenv()

# 2. 从环境变量中读取配置，并设置默认值和类型


class Settings:
    def __init__(self):
        self.DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'  # 转换为布尔值
        # self.SECRET_KEY = os.getenv('SECRET_KEY')  # 如果没有会返回 None
        # self.DATABASE_URL = os.getenv('DATABASE_URL')
        # self.API_KEY = os.getenv('API_KEY')
        self.NEO4J_URL = os.getenv('NEO4J_URL')
        self.NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
        self.NEO4J_DATABASE = os.getenv('NEO4J_DATABASE')
        self.DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
        self.DEEPSEEK_API_BASE_URL = os.getenv('DEEPSEEK_API_BASE_URL', 'https://api.deepseek.com/v1')  # 默认值
        self.DEEPSEEK_MODEL = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')  # 默认值
        self.SHPFILES_DIR = os.getenv('SHPFILES_DIR', './shpfiles')  # 默认值

# 3. 创建一个全局配置实例，方便其他地方导入使用
settings = Settings()
