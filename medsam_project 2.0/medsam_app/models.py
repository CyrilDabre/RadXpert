# models.py

import os
import uuid
from django.db import models

def image_upload_path(instance, filename):
    return os.path.join('uploads/'+str(instance.folder_name), filename)

class UploadedImage(models.Model):
    folder_name = models.CharField(max_length=255, default=uuid.uuid4, editable=False)
    image = models.ImageField(upload_to=image_upload_path)
    upload_time = models.DateTimeField(auto_now_add=True)
