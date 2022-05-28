from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent
LOG_ROOT = os.path.join(BASE_DIR, 'logs')
ERROR_LOGGING = True

HOST = os.environ.get('HOST')

SQLALCHEMY_DATABASE_URL = 'postgresql+psycopg2://' + os.getenv('DB_USER') + ':' + os.getenv(
    'DB_PASSWORD') + '@' + os.getenv('DB_HOST') + ':' + os.getenv('DB_PORT') + '/' + os.getenv('DB_NAME')
