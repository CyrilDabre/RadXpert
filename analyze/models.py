

import os
import uuid
from django.db import models
from django.contrib.auth.models import User

def image_upload_path(instance, filename):
    return os.path.join('uploads/'+str(instance.folder_name), filename)

class UploadedImage(models.Model):
    folder_name = models.CharField(max_length=255, default=uuid.uuid4, editable=False)
    image = models.ImageField(upload_to=image_upload_path)
    upload_time = models.DateTimeField(auto_now_add=True)


class Profile(models.Model):
    USER_TYPES = (
        ("doctor", "Doctor"),
        ("patient", "Patient"),
    )

    user = models.OneToOneField(User, on_delete=models.CASCADE)
    usertype = models.CharField(max_length=10, choices=USER_TYPES)

    def _str_(self):
        return f"{self.user.username} - {self.usertype}"