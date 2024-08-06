# Generated by Django 5.0.6 on 2024-07-10 13:21

import django.utils.timezone
import medsam_app.models
import uuid
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('medsam_app', '0003_remove_uploadedimage_upload_time_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='uploadedimage',
            name='upload_time',
            field=models.DateTimeField(auto_now_add=True, default=django.utils.timezone.now),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='uploadedimage',
            name='folder_name',
            field=models.CharField(default=uuid.uuid4, editable=False, max_length=255),
        ),
        migrations.AlterField(
            model_name='uploadedimage',
            name='image',
            field=models.ImageField(upload_to=medsam_app.models.image_upload_path),
        ),
    ]
