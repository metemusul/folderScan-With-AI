import os
import zipfile
import tempfile
import subprocess
import json
import pandas as pd
import joblib
from django.shortcuts import render, redirect
from .forms import UploadFileForm
from django.contrib.auth.decorators import login_required
from Appgoogleauth.models import FirestoreModel
from django.http import Http404

def handle_uploaded_file(f):
    with tempfile.TemporaryDirectory() as tmpdirname:
        zip_path = os.path.join(tmpdirname, 'uploaded.zip')
        with open(zip_path, 'wb+') as destination:
            for chunk in f.chunks():
                destination.write(chunk)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdirname)
        
        # Run Bandit and save the report as JSON
        report_path = os.path.join(tmpdirname, 'bandit_report.json')
        subprocess.run(['bandit', '-r', tmpdirname, '-f', 'json', '-o', report_path], capture_output=True, text=True)
        
        # Read the Bandit report
        with open(report_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Extract the "results" section
        results = data.get("results", [])
        
        # Extract issue_text and line_number
        issues = [{"issue_text": result.get("issue_text"), "line_number": result.get("line_number")} for result in results]
        
        # Process the metrics into a DataFrame
        metrics_data = data.get("metrics", {})
        df_list = []
        for file_path, values in metrics_data.items():
            row = {"file_path": file_path}
            row.update(values)
            df_list.append(row)
        
        # Create a DataFrame
        df = pd.DataFrame(df_list)
        
        # Create a summary row
        project_summary = df.drop(columns=["file_path"]).sum().to_frame().T
        project_summary.insert(0, "folder_name", "uploaded_project")  # Add a project name
        
        return project_summary, issues

@login_required
def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                report_df, issues = handle_uploaded_file(request.FILES['file'])
                
                # Model yükleme ve tahmin işlemleri
                model_path = os.path.join(os.path.dirname(__file__), 'models', 'random_forest_combined.pkl')
                loaded_models = joblib.load(model_path)

                quality_model = loaded_models["quality_model"]
                security_model = loaded_models["security_model"]

                X_test = report_df.drop(columns=["folder_name"], errors="ignore")
                predicted_quality = quality_model.predict(X_test)
                predicted_security = security_model.predict(X_test)

                report_df["predicted_software_quality"] = predicted_quality
                report_df["predicted_software_security"] = predicted_security

                # Sonuçları Firestore'a kaydet
                report_html = report_df.to_html(index=False)
                issues_html = "<br>".join([f"Line {issue['line_number']}: {issue['issue_text']}" for issue in issues])
                
                # Firestore'a kaydet
                upload_data = FirestoreModel.create_upload(
                    user=request.user.username,
                    zip_name=request.FILES['file'].name,
                    bandit_results=report_html,
                    ai_results=issues_html
                )
                
                return redirect('view_files')
            except Exception as e:
                print(f"Error during file upload: {str(e)}")
                form.add_error(None, f"Error processing file: {str(e)}")
    else:
        form = UploadFileForm()
    return render(request, 'security/upload.html', {'form': form})

@login_required
def view_report(request, file_id):
    file_data = FirestoreModel.get_file_report(file_id, request.user.username)
    
    if not file_data:
        raise Http404("File not found or access denied")
    
    context = {
        'file_name': file_data['zip_name'],
        'report': file_data.get('bandit_results', {}),
        'issues': file_data.get('ai_results', {}),
        'bandit_results': file_data.get('bandit_results', {}),
        'ai_results': file_data.get('ai_results', {})
    }
    
    return render(request, 'security/report.html', context) 