from django.contrib import admin
from .models import BoneAgeRecord, UserProfile


@admin.register(BoneAgeRecord)
class BoneAgeRecordAdmin(admin.ModelAdmin):
    list_display = ('patient_name', 'patient_gender', 'predicted_age_years', 'user', 'uploaded_at')
    list_filter = ('patient_gender', 'uploaded_at')
    search_fields = ('patient_name', 'user__username')
    readonly_fields = ('predicted_age_months', 'predicted_age_years', 'uploaded_at')


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'phone_number', 'location', 'role')
    list_filter = ('role',)
    search_fields = ('user__username', 'phone_number', 'location')
