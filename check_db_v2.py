import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'bone_age_project.settings')
django.setup()

from classifier.models import BoneAgeRecord
from django.contrib.auth.models import User
from django.db.models import Count

print("Records count per user:")
user_counts = BoneAgeRecord.objects.values('user__username').annotate(total=Count('id'))
for entry in user_counts:
    print(f"User: {entry['user__username'] or 'Anonymous'}, Count: {entry['total']}")

print("\nRecent records details:")
for record in BoneAgeRecord.objects.all().order_by('-id')[:20]:
    print(f"ID: {record.id}, Patient: {record.patient_name}, User: {record.user.username if record.user else 'Anon'}, Date: {record.uploaded_at}")
