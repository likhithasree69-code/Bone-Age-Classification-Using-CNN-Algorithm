from django.db import models
from django.contrib.auth.models import User


class UserProfile(models.Model):
    ROLE_CHOICES = [
        ('user', 'User'),
        ('admin', 'Admin'),
    ]
    
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    phone_number = models.CharField(max_length=20, blank=True)
    location = models.CharField(max_length=150, blank=True)
    role = models.CharField(max_length=10, choices=ROLE_CHOICES, default='user')

    def __str__(self):
        return f"{self.user.username}'s Profile ({self.role})"


class BoneAgeRecord(models.Model):
    """Model to store X-ray uploads and prediction results."""

    GENDER_CHOICES = [
        ('male', 'Male'),
        ('female', 'Female'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    patient_name = models.CharField(max_length=100, blank=True)
    patient_gender = models.CharField(max_length=10, choices=GENDER_CHOICES, default='male')
    xray_image = models.ImageField(upload_to='uploads/')
    predicted_age_months = models.FloatField(null=True, blank=True)
    predicted_age_years = models.FloatField(null=True, blank=True)
    bone_stage = models.CharField(max_length=50, null=True, blank=True)
    bone_abnormality = models.CharField(max_length=100, null=True, blank=True)
    affected_area_size = models.CharField(max_length=50, null=True, blank=True)
    annotated_image = models.ImageField(upload_to='annotated/', null=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-uploaded_at']

    def __str__(self):
        return f"{self.patient_name} - {self.predicted_age_years} years ({self.uploaded_at.strftime('%Y-%m-%d')})"
