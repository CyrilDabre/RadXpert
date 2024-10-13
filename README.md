1. Clone the GitHub Repository
git clone https://github.com/CyrilDabre/RadXpert.git
cd RadXpert
________________________________________
2. Install Dependencies
1.	Install Dependencies:
pip install -r requirements.txt
This will install all the necessary packages for Django, image processing, machine learning models (like OpenCV, PyTorch, etc.), and other tools used in the project.
________________________________________
3. Configure Django Settings
1.	Set Up settings.py:
o	Make sure the settings.py is properly configured for your local environment. Check the DATABASES setting if you are using SQLite or another database.
o	Ensure static and media file handling is set up:
STATIC_URL = '/static/'
STATICFILES_DIRS = [os.path.join(BASE_DIR, 'static')]
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
2.	Apply Migrations:
python manage.py makemigrations
python manage.py migrate
________________________________________
4. Create Superuser for Admin Access
You need a superuser to access the Django admin panel.
python manage.py createsuperuser
Follow the prompts to create a username, email, and password.
________________________________________
5. Running the Server
Once everything is set up, run the Django development server:
python manage.py runserver
Open your browser and go to http://127.0.0.1:8000/ to see the application running.
