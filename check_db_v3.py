import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'bone_age_project.settings')
django.setup()

from classifier.models import BoneAgeRecord
from django.contrib.auth.models import User

anon_records = BoneAgeRecord.objects.filter(user__isnull=True)
print(f"Total Anonymous records (user is NULL): {anon_records.count()}")
for record in anon_records:
    print(f"ID: {record.id}, Patient: {record.patient_name}, Date: {record.uploaded_at}")

all_users = User.objects.all()
print(f"\nTotal Users: {all_users.count()}")
for user in all_users:
    count = BoneAgeRecord.objects.filter(user=user).count()
    print(f"User: {user.username}, Records: {count}")
