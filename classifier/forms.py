from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import BoneAgeRecord


class RegisterForm(forms.Form):
    """User registration form collecting only requested details."""
    first_name = forms.CharField(
        max_length=50,
        required=True,
        widget=forms.TextInput(attrs={'class': 'form-input', 'placeholder': 'First name'})
    )
    last_name = forms.CharField(
        max_length=50,
        required=True,
        widget=forms.TextInput(attrs={'class': 'form-input', 'placeholder': 'Last name'})
    )
    phone_number = forms.CharField(
        max_length=20,
        required=True,
        widget=forms.TextInput(attrs={'class': 'form-input', 'placeholder': 'Phone Number'})
    )
    location = forms.CharField(
        max_length=150,
        required=True,
        widget=forms.TextInput(attrs={'class': 'form-input', 'placeholder': 'Location (e.g., City)'})
    )
    username = forms.CharField(
        max_length=50,
        required=True,
        widget=forms.TextInput(attrs={'class': 'form-input', 'placeholder': 'Create a username'})
    )
    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(attrs={'class': 'form-input', 'placeholder': 'Email address', 'autocomplete': 'email'})
    )
    password = forms.CharField(
        required=True,
        widget=forms.PasswordInput(attrs={'class': 'form-input', 'placeholder': 'Create a password'})
    )

    def clean_username(self):
        username = self.cleaned_data.get('username')
        if User.objects.filter(username=username).exists():
            raise forms.ValidationError("This username is already taken.")
        return username

    def clean_email(self):
        email = self.cleaned_data.get('email')
        if User.objects.filter(email=email).exists():
            raise forms.ValidationError("A user with that email already exists.")
        return email

    def save(self):
        username = self.cleaned_data['username']
        email = self.cleaned_data['email']
        # Use the explicit username provided by the user
        user = User.objects.create_user(
            username=username,
            email=email,
            password=self.cleaned_data['password'],
            first_name=self.cleaned_data['first_name'],
            last_name=self.cleaned_data['last_name']
        )
        
        # We need to import UserProfile here avoiding circular imports if necessary, 
        # but models are imported at the top. Let's make sure UserProfile is imported.
        from .models import UserProfile
        UserProfile.objects.create(
            user=user,
            phone_number=self.cleaned_data['phone_number'],
            location=self.cleaned_data['location'],
            role='user' # Default to user role
        )
        return user


class XRayUploadForm(forms.ModelForm):
    """Form for uploading hand X-ray images for bone age prediction."""

    class Meta:
        model = BoneAgeRecord
        fields = ['patient_name', 'patient_gender', 'xray_image']
        widgets = {
            'patient_name': forms.TextInput(attrs={
                'class': 'form-input',
                'placeholder': 'Enter patient name',
            }),
            'patient_gender': forms.Select(attrs={
                'class': 'form-input',
            }),
            'xray_image': forms.FileInput(attrs={
                'class': 'form-input file-input',
                'accept': 'image/*',
            }),
        }
