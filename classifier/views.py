from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import json
import os
import uuid
from .forms import RegisterForm, XRayUploadForm
from .models import BoneAgeRecord, UserProfile
from .ml_model.predict import predict_bone_age
from functools import wraps

def admin_required(view_func):
    """Custom decorator to check if user has admin role."""
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return redirect('login')
        try:
            if request.user.userprofile.role != 'admin':
                messages.error(request, 'Access Denied: Admin privileges required.')
                return redirect('dashboard')
        except UserProfile.DoesNotExist:
            return redirect('dashboard')
        return view_func(request, *args, **kwargs)
    return _wrapped_view



def home(request):
    """Landing page - Always shows home page. Users can navigate to login/dashboard."""
    return render(request, 'home.html')


def login_view(request):
    """Handle regular user login - Excludes Admins."""
    if request.user.is_authenticated:
        try:
            profile = UserProfile.objects.get(user=request.user)
            if profile.role == 'admin':
                return redirect('admin_dashboard')
            else:
                return redirect('dashboard')
        except UserProfile.DoesNotExist:
            return redirect('dashboard')

    if request.method == 'POST':
        username = request.POST.get('username', '')
        password = request.POST.get('password', '')
        
        user = authenticate(request, username=username, password=password)
        if user is not None:
            # Prevent Admins from using the regular user login page
            try:
                profile = UserProfile.objects.get(user=user)
                if profile.role == 'admin':
                    messages.error(request, 'This is a User Login. Administrators must use the Admin Access portal.')
                    return render(request, 'login.html')
            except UserProfile.DoesNotExist:
                pass

            login(request, user)
            messages.success(request, f'Welcome back, {user.username}!')
            return redirect('dashboard')
        else:
            messages.error(request, 'Invalid User username or password.')

    return render(request, 'login.html', {'title': 'User Login'})


def admin_login_view(request):
    """Handle administrator login - Excludes regular Users."""
    if request.user.is_authenticated:
        try:
            profile = UserProfile.objects.get(user=request.user)
            if profile.role == 'admin':
                return redirect('admin_dashboard')
            else:
                return redirect('dashboard')
        except UserProfile.DoesNotExist:
            return redirect('dashboard')

    if request.method == 'POST':
        username = request.POST.get('username', '')
        password = request.POST.get('password', '')
        
        user = authenticate(request, username=username, password=password)
        if user is not None:
            # Check if user is actually an admin
            try:
                profile = UserProfile.objects.get(user=user)
                if profile.role != 'admin':
                    messages.error(request, 'Access Denied: Regular users cannot use the Admin portal.')
                    return render(request, 'login.html', {'title': 'Admin Login'})
            except UserProfile.DoesNotExist:
                messages.error(request, 'Access Denied: Admin profile not found.')
                return render(request, 'login.html', {'title': 'Admin Login'})

            login(request, user)
            messages.success(request, 'Admin session started.')
            return redirect('admin_dashboard')
        else:
            messages.error(request, 'Invalid Admin credentials.')

    return render(request, 'login.html', {'title': 'Admin Login'})



def register_view(request):
    """Handle user registration and ensure a fresh session."""
    if request.user.is_authenticated:
        logout(request)

    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            # login(request, user) # We won't auto-login so they can see the login page
            messages.success(request, f'Account created successfully! Please log in to continue.')
            return redirect('login')
        else:
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f'{error}')
    else:
        form = RegisterForm()

    return render(request, 'register.html', {'form': form})


def logout_view(request):
    """Handle user logout."""
    logout(request)
    messages.info(request, 'You have been logged out successfully.')
    return redirect('login')


@login_required
def dashboard_view(request):
    """Regular User Dashboard showing their predictions and stats."""
    try:
        profile = UserProfile.objects.get(user=request.user)
        if profile.role == 'admin':
            return redirect('admin_dashboard')
    except UserProfile.DoesNotExist:
        pass

    total_predictions = BoneAgeRecord.objects.count()
    my_predictions = BoneAgeRecord.objects.filter(user=request.user).count()
    recent_records = BoneAgeRecord.objects.filter(user=request.user)[:10]

    # Global Case Ratio for perspective
    male_count = BoneAgeRecord.objects.filter(patient_gender='male').count()
    female_count = BoneAgeRecord.objects.filter(patient_gender='female').count()

    context = {
        'total_predictions': total_predictions,
        'recent_records': recent_records,
        'my_predictions': my_predictions,
        'male_count': male_count,
        'female_count': female_count,
        'is_admin': False,
    }
    return render(request, 'dashboard.html', context)


@admin_required
def admin_dashboard_view(request):
    """Admin Dashboard - Restricted to users with admin role."""
    total_predictions = BoneAgeRecord.objects.count()
    total_users = User.objects.count()
    recent_records = BoneAgeRecord.objects.all()[:10]
    my_predictions = BoneAgeRecord.objects.filter(user=request.user).count()

    # Global Case distribution
    male_count = BoneAgeRecord.objects.filter(patient_gender='male').count()
    female_count = BoneAgeRecord.objects.filter(patient_gender='female').count()

    all_users = UserProfile.objects.all().select_related('user')
    
    context = {
        'total_predictions': total_predictions,
        'total_users': total_users,
        'recent_records': recent_records,
        'my_predictions': my_predictions,
        'male_count': male_count,
        'female_count': female_count,
        'is_admin': True,
        'registered_users': all_users,
    }
    return render(request, 'admin_dashboard.html', context)


@login_required
def prediction_view(request):
    """Handle X-ray upload and bone age prediction."""
    if request.method == 'POST':
        form = XRayUploadForm(request.POST, request.FILES)
        if form.is_valid():
            record = form.save(commit=False)
            record.user = request.user
            record.save()

            # Run prediction
            image_path = record.xray_image.path
            gender = record.patient_gender

            try:
                prediction_result = predict_bone_age(image_path, gender)
                predicted_months = prediction_result['months']
                predicted_years = round(predicted_months / 12, 1)

                record.predicted_age_months = round(predicted_months, 1)
                record.predicted_age_years = predicted_years
                record.bone_abnormality = prediction_result['abnormality']
                record.affected_area_size = prediction_result['affected_area']
                if prediction_result.get('annotated_image'):
                    record.annotated_image.name = prediction_result['annotated_image']

                # Calculate maturity level (bone stage)
                if predicted_months < 84:
                    record.bone_stage = "Low (Early Stage)"
                elif predicted_months < 168:
                    record.bone_stage = "Medium (Mid Stage)"
                else:
                    record.bone_stage = "High (Mature Stage)"

                record.save()
                messages.success(request, 'Bone age prediction completed and saved to your history!')
                return redirect('result', record_id=record.id)
            except Exception as e:
                messages.error(request, f'Prediction failed: {str(e)}')
                if record.id:
                    record.delete()
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = XRayUploadForm()

    return render(request, 'prediction.html', {'form': form})


@login_required
def result_view(request, record_id):
    """Display prediction results, restricted to the record owner."""
    record = get_object_or_404(BoneAgeRecord, id=record_id, user=request.user)
    years = int(record.predicted_age_months // 12) if record.predicted_age_months else 0
    months = int(record.predicted_age_months % 12) if record.predicted_age_months else 0

    # Determine Bone Maturity Level
    maturity_level = "Unknown"
    maturity_color = "var(--text-primary)"
    
    if record.predicted_age_months:
        if record.predicted_age_months < 84:  # Under 7 years
            maturity_level = "Low (Early Stage)"
            maturity_color = "var(--danger-light)"
        elif record.predicted_age_months < 168:  # 7 to 14 years
            maturity_level = "Medium (Mid Stage)"
            maturity_color = "var(--warning)"
        else:  # 14+ years
            maturity_level = "High (Mature Stage)"
            maturity_color = "var(--success-light)"

    context = {
        'record': record,
        'age_years': years,
        'age_months': months,
        'formatted_age': f"{years} years, {months} months",
        'maturity_level': maturity_level,
        'maturity_color': maturity_color,
    }
    return render(request, 'result.html', context)


@login_required
def history_view(request):
    """Display prediction history - all records for admin, own records for regular user."""
    try:
        profile = UserProfile.objects.get(user=request.user)
        is_admin = profile.role == 'admin'
    except UserProfile.DoesNotExist:
        is_admin = False

    if is_admin:
        # Admin sees all users' records
        records = BoneAgeRecord.objects.all()
    else:
        # Regular user sees only their own records
        records = BoneAgeRecord.objects.filter(user=request.user)

    return render(request, 'history.html', {'records': records, 'is_admin': is_admin})


@login_required
def delete_record(request, record_id):
    """Delete a prediction record."""
    record = get_object_or_404(BoneAgeRecord, id=record_id, user=request.user)
    record.delete()
    messages.success(request, 'Record deleted successfully.')
    return redirect('history')


@csrf_exempt
def mobile_login_api(request):
    """Mobile API: Login and return user info as JSON."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=405)
    try:
        data = json.loads(request.body)
        username = data.get('username', '')
        password = data.get('password', '')
    except Exception:
        username = request.POST.get('username', '')
        password = request.POST.get('password', '')

    user = authenticate(request, username=username, password=password)
    if user is not None:
        login(request, user)
        try:
            profile = UserProfile.objects.get(user=user)
            role = profile.role
        except UserProfile.DoesNotExist:
            role = 'user'
        return JsonResponse({
            'success': True,
            'username': user.username,
            'first_name': user.first_name,
            'role': role,
        })
    return JsonResponse({'success': False, 'error': 'Invalid username or password'}, status=401)


@csrf_exempt
def mobile_predict_api(request):
    """Mobile API: Accept image upload and return bone age prediction as JSON."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=405)

    # Basic auth check via POST fields
    username = request.POST.get('username', '')
    password = request.POST.get('password', '')
    user = authenticate(request, username=username, password=password)
    if user is None:
        return JsonResponse({'success': False, 'error': 'Invalid credentials'}, status=401)

    if 'xray_image' not in request.FILES:
        return JsonResponse({'success': False, 'error': 'No image file provided'}, status=400)

    image_file = request.FILES['xray_image']
    patient_name = request.POST.get('patient_name', 'Unknown Patient')
    gender = request.POST.get('gender', 'male')
    if gender not in ['male', 'female']:
        gender = 'male'

    # Save the uploaded file
    ext = os.path.splitext(image_file.name)[1] or '.jpg'
    unique_name = f'xrays/mobile_{uuid.uuid4().hex}{ext}'
    saved_path = default_storage.save(unique_name, ContentFile(image_file.read()))
    full_path = default_storage.path(saved_path)

    try:
        prediction_result = predict_bone_age(full_path, gender)
        predicted_months = prediction_result['months']
        predicted_years = round(predicted_months / 12, 1)

        if predicted_months < 84:
            bone_stage = 'Low (Early Stage)'
        elif predicted_months < 168:
            bone_stage = 'Medium (Mid Stage)'
        else:
            bone_stage = 'High (Mature Stage)'

        # Save record to DB
        record = BoneAgeRecord(
            user=user,
            patient_name=patient_name,
            patient_gender=gender,
            predicted_age_months=round(predicted_months, 1),
            predicted_age_years=predicted_years,
            bone_abnormality=prediction_result.get('abnormality', 'Normal'),
            affected_area_size=prediction_result.get('affected_area', '0%'),
            bone_stage=bone_stage,
        )
        record.xray_image.name = saved_path
        record.save()

        return JsonResponse({
            'success': True,
            'patient_name': patient_name,
            'gender': gender,
            'predicted_months': round(predicted_months, 1),
            'predicted_years': predicted_years,
            'bone_stage': bone_stage,
            'abnormality': prediction_result.get('abnormality', 'Normal'),
            'affected_area': prediction_result.get('affected_area', '0%'),
            'record_id': record.id,
        })
    except Exception as e:
        # Cleanup on failure
        if default_storage.exists(saved_path):
            default_storage.delete(saved_path)
        return JsonResponse({'success': False, 'error': str(e)}, status=500)
#   T r i g g e r   b u i l d  
 