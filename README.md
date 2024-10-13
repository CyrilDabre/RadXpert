RadXpert Project
================
Table of Contents
-----------------
Project Overview
----------------
RadXpert is a [briefly describe the project].
Setup Instructions
-------------------
Step 1: Clone Repository
Bash
git clone https://github.com/CyrilDabre/RadXpert.git
cd RadXpert
Step 2: Install Dependencies
Bash
pip install -r requirements.txt
Step 3: Configure Django Settings
settings.py Configuration
Ensure DATABASES setting is correct (e.g., SQLite).
Set up static and media file handling:
Python
STATIC_URL = '/static/'
STATICFILES_DIRS = [os.path.join(BASE_DIR, 'static')]
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
Apply Migrations
Bash
python manage.py makemigrations
python manage.py migrate
Step 4: Create Superuser
Bash
python manage.py createsuperuser
Follow prompts to create username, email, and password.
Step 5: Run Server
Bash
python manage.py runserver
Open browser and navigate to .
Troubleshooting
---------------
Check GitHub credentials for permission issues.
Verify Django settings and migrations.
