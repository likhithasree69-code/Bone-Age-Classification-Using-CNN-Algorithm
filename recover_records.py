import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'bone_age_project.settings')
django.setup()

from classifier.models import BoneAgeRecord
from django.contrib.auth.models import User

# 1. Target user ni identify cheddham
target_username = 'boneage@gmail.com'
try:
    target_user = User.objects.get(username=target_username)
    print(f"Target User found: {target_user.username}")
    
    # 2. Mottham records ni ee user ki update cheddam
    records_to_update = BoneAgeRecord.objects.all()
    count = records_to_update.count()
    
    for record in records_to_update:
        record.user = target_user
        record.save()
        
    print(f"Successfully recovered {count} records to {target_username}!")

except User.DoesNotExist:
    print(f"Error: User {target_username} not found in database.")
except Exception as e:
    print(f"Error occurred: {str(e)}")
