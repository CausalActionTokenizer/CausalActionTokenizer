from setuptools import setup, find_packages

setup(
    name='catok',  # 替换成你的包名，例如 'my_awesome_package' 或 'vector_quantize_pytorch_local'
    version='0.1.0',   # 你的包的初始版本号
    author='Anonymous',
    author_email='anonymous@anonymous.com',
    description='Causal Action Tokenizer', # 包的简短描述
    # long_description=open('README.md', encoding='utf-8').read() if_exists('README.md'), # 从 README.md 读取详细描述 (可选)
    # long_description_content_type='text/markdown', # README 文件格式 (如果是 markdown)
    # url='https://github.com/你的用户名/你的项目仓库名', # 项目的URL (例如 GitHub 仓库链接)
    
    packages=find_packages(exclude=['tests*', 'docs*']), # 自动查找项目中的包，排除测试和文档目录
                                                      # 如果你的包不在根目录（例如在 src/你的包名），可以使用 find_packages(where='src')
                                                      # 或者手动指定: packages=['你的包名', '你的包名.子包名']

    # install_requires=[  # 列出你的包所依赖的其他包
    #     'numpy>=1.20',
    #     'torch',
    #     # '其他依赖包==版本号',
    # ],
    
    classifiers=[  # 包的分类信息，有助于在 PyPI 上搜索和分类
        'Development Status :: 3 - Alpha', # 开发阶段：3 - Alpha, 4 - Beta, 5 - Production/Stable
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License', # 你的项目许可证，例如 MIT, Apache 2.0
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Operating System :: OS Independent', # 通常是跨平台的
    ],
    
    python_requires='>=3.7', # 指定你的包兼容的 Python 版本
    
    # 如果你的包包含非 Python 文件 (例如数据文件、模板等)，可能需要 include_package_data=True 和 MANIFEST.in 文件
    # include_package_data=True,
    
    # 如果你的包需要提供命令行工具
    # entry_points={
    #     'console_scripts': [
    #         '你的命令名=你的包名.模块名:函数名',
    #     ],
    # },
)

# 一个辅助函数，用于安全地读取 README.md
import os
def if_exists(file_path):
    return os.path.exists(file_path) and os.path.isfile(file_path)

# 为了在 setup() 中使用，需要确保 if_exists 定义在上面或者直接在 long_description 中处理
# 更简洁的方式是：
# try:
#     with open('README.md', encoding='utf-8') as f:
#         long_description = f.read()
# except FileNotFoundError:
#     long_description = '这里写一个关于你的包的简短描述'
#
# setup(
#    ...
#    long_description=long_description,
#    ...
# )