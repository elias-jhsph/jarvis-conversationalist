__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
__version__ = '0.3.2'
# Path: src/jarvis_conversationalist/conversationalist.py
