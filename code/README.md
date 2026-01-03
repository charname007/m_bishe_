# 小红书爬虫项目

## ⚠️ 重要提示

**本项目仅供学习研究使用，请遵守以下原则：**

1. **遵守法律法规**：遵守《网络安全法》《个人信息保护法》等相关法律法规
2. **遵守网站条款**：遵守小红书的使用条款和robots.txt协议
3. **合理使用**：不要进行大规模爬取，避免对服务器造成压力
4. **尊重隐私**：不要爬取用户隐私信息
5. **建议使用官方API**：优先考虑使用小红书官方提供的API接口

## 项目结构

```
.
├── requirements.txt      # 依赖包
├── config.py            # 配置文件
├── xiaohongshu_spider.py  # 主爬虫程序
├── utils.py             # 工具函数
└── README.md            # 说明文档
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

```bash
python xiaohongshu_spider.py
```

## 注意事项

- 需要配置Cookie和User-Agent
- 建议使用代理IP避免被封
- 设置合理的请求间隔（建议2-5秒）
- 仅用于个人学习研究，不得用于商业用途

