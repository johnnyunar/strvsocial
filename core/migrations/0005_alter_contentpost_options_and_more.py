# Generated by Django 5.1.6 on 2025-03-05 20:45

import core.utils
import core.validators
import functools
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0004_alter_contentpost_user'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='contentpost',
            options={'get_latest_by': 'updated_at', 'ordering': ['-created_at'], 'verbose_name': 'Content', 'verbose_name_plural': 'Content'},
        ),
        migrations.AlterField(
            model_name='contentpost',
            name='media_file',
            field=models.FileField(blank=True, null=True, upload_to=functools.partial(core.utils.generate_random_filename, *(), **{'subdir': 'content'}), validators=[core.validators.validate_media_file], verbose_name='Media File'),
        ),
    ]
