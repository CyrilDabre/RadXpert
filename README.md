RadXpert
================
Table of Contents
-----------------
Project Overview
----------------
The medical field faces significant challenges due to the overwhelming volume of imag-
ing data and a shortage of radiologists, leading to increased workloads and diagnostic
errors. Traditional radiology reporting systems often fail to produce detailed, compre-
hensive reports, resulting in incomplete diagnoses and delays in treatment. To address
these issues, RadXpert leverages advanced AI technologies such as BiomedCLIP, Med-
SAM, R2GenGPT, and Google Gemini API to automate the generation of radiology re-
ports. BiomedCLIP ensures accurate image-to-text alignment, while MedSAM provides
interactive segmentation to highlight critical regions in medical images. R2GenGPT gen-
erates coherent and contextually relevant reports, and the Google Gemini API enhances the
clarity and detail of the final output.
RadXpert offers a user-friendly dashboard that enables radiologists to upload images,
receive automated, detailed reports, and focus on more complex cases, improving overall
efficiency. By automating routine tasks, RadXpert significantly reduces the workload on
radiologists, allowing them to concentrate on critical decision-making. The system’s abil-
ity to generate precise, actionable reports leads to faster diagnoses and earlier interventions,
ultimately enhancing patient outcomes.
Developed using Agile methodologies, RadXpert has been tested on real-world datasets
like IU-Xray and MIMIC-CXR, demonstrating its potential to streamline radiology work-
flows. Future developments aim to expand RadXpert’s capabilities to support additional
imaging modalities and ensure seamless integration with existing healthcare systems. By
addressing the current gaps in radiology reporting and improving diagnostic accuracy,
RadXpert represents a transformative solution in modern healthcare, with the potential to
revolutionize radiology practices globally.

Setup Instructions
-------------------
Step 1: Clone Repository

  git clone https://github.com/CyrilDabre/RadXpert.git
  
  cd RadXpert

Step 2: Install Dependencies

  pip install -r requirements.txt

Step 3: Configure Django Settings

  settings.py Configuration
  
  Ensure DATABASES setting is correct (e.g., SQLite).

Set up static and media file handling:


  STATIC_URL = '/static/' 
  
  STATICFILES_DIRS = [os.path.join(BASE_DIR, 'static')] 
  
  MEDIA_URL = '/media/' 
  
  MEDIA_ROOT = os.path.join(BASE_DIR, 'media') 

  Apply Migrations
  
  python manage.py makemigrations
  
  python manage.py migrate

Step 4: Create Superuser

  python manage.py createsuperuser

  Follow prompts to create username, email, and password.

Step 5: Run Server


  python manage.py runserver

  Open browser and navigate to http://127.0.0.1:8000/

Troubleshooting
---------------
Check GitHub credentials for permission issues.

Verify Django settings and migrations.
