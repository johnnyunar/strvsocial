# Generated by Django 5.1.6 on 2025-03-05 20:45

import users.manager
from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('users', '0003_socialuser_avatar'),
    ]

    operations = [
        migrations.AlterModelManagers(
            name='socialuser',
            managers=[
                ('objects', users.manager.SocialUserManager()),
            ],
        ),
    ]
