# Generated by Django 3.2.8 on 2021-10-19 11:44

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('imgfuse', '0006_upload_use'),
    ]

    operations = [
        migrations.AddField(
            model_name='upload',
            name='Expected_img',
            field=models.ImageField(blank=True, null=True, upload_to='images'),
        ),
    ]