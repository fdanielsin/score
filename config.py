import os

basedir = os.path.abspath(os.path.dirname(__file__))
class Config(object):
    DATABASE = 'sqlite.db'
    SECRET_KEY = 'secret'
    #DATABASE_PATH = os.path.join(basedir, DATABASE)
    SQLALCHEMY_DATABASE_URI = 'sqlite:///sqlite.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO = True
    DEBUG = True