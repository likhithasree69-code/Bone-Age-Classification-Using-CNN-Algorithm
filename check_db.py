import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'bone_age_project.settings')
django.setup()

from classifier.models import BoneAgeRecord
from django.contrib.auth.models import User

print(f"Total BoneAgeRecords: {BoneAgeRecord.objects.count()}")
for record in BoneAgeRecord.objects.all():
    user_name = record.user.username if record.user else "Anonymous"
    print(f"ID: {record.id}, Patient: {record.patient_name}, User: {user_name}, Date: {record.uploaded_at}")

print("\nUsers List:")
for user in User.objects.all():
    print(f"User: {user.username}, Email: {user.email}")
