# Generated by Django 3.2.9 on 2021-11-24 14:27

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('imgfuse', '0004_upload_fs8'),
    ]

    operations = [
        migrations.AlterField(
            model_name='upload',
            name='fs1',
            field=models.ImageField(blank=True, default='default.jpg', null=True, upload_to='images'),
        ),
        migrations.AlterField(
            model_name='upload',
            name='fs2',
            field=models.ImageField(blank=True, default='default.jpg', null=True, upload_to='images'),
        ),
        migrations.AlterField(
            model_name='upload',
            name='fs3',
            field=models.ImageField(blank=True, default='default.jpg', null=True, upload_to='images'),
        ),
        migrations.AlterField(
            model_name='upload',
            name='fs4',
            field=models.ImageField(blank=True, default='default.jpg', null=True, upload_to='images'),
        ),
        migrations.AlterField(
            model_name='upload',
            name='fs5',
            field=models.ImageField(blank=True, default='default.jpg', null=True, upload_to='images'),
        ),
        migrations.AlterField(
            model_name='upload',
            name='fs6',
            field=models.ImageField(blank=True, default='default.jpg', null=True, upload_to='images'),
        ),
        migrations.AlterField(
            model_name='upload',
            name='fs7',
            field=models.ImageField(blank=True, default='default.jpg', null=True, upload_to='images'),
        ),
        migrations.AlterField(
            model_name='upload',
            name='fs8',
            field=models.ImageField(blank=True, default='default.jpg', null=True, upload_to='images'),
        ),
        migrations.AlterField(
            model_name='upload',
            name='fs9',
            field=models.ImageField(blank=True, default='default.jpg', null=True, upload_to='images'),
        ),
    ]