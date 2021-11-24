# Generated by Django 3.2.9 on 2021-11-24 04:18

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('imgfuse', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='upload',
            name='FUSION_METHOD',
            field=models.CharField(blank=True, choices=[('meanmean', 'MEANMEAN'), ('meanmax', 'MEANMAX'), ('meanmin', 'MEANMIN'), ('maxmean', 'MAXMEAN'), ('maxmax', 'MAXMAX'), ('maxmin', 'MAXMIN'), ('minmean', 'MINMEAN'), ('minmax', 'MINMAX'), ('minmin', 'MINMIN')], max_length=50),
        ),
    ]