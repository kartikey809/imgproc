# Generated by Django 3.2.8 on 2021-10-19 09:28

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('imgfuse', '0002_alter_upload_fused_img'),
    ]

    operations = [
        migrations.AlterField(
            model_name='upload',
            name='Fused_img',
            field=models.ImageField(blank=True, default='media\\images\\default.JPG', upload_to='images'),
        ),
    ]