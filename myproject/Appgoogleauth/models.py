from django.db import models
from firebase_admin import firestore
from django.conf import settings
from datetime import datetime

# Create your models here.

class UploadedZip(models.Model):
    user = models.CharField(max_length=255)
    zip_name = models.CharField(max_length=255)
    upload_time = models.DateTimeField(auto_now_add=True)
    bandit_results = models.JSONField(null=True, blank=True)
    ai_results = models.JSONField(null=True, blank=True)

    def __str__(self):
        return f"{self.user} - {self.zip_name}"

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        # Save to Firestore
        db = firestore.client()
        doc_ref = db.collection('uploads').document(f"{self.user}_{self.id}")
        doc_ref.set({
            'user': self.user,
            'zip_name': self.zip_name,
            'upload_time': self.upload_time,
            'bandit_results': self.bandit_results,
            'ai_results': self.ai_results
        })

    @staticmethod
    def get_user_uploads(username):
        db = firestore.client()
        uploads = db.collection('uploads').where('user', '==', username).stream()
        return [doc.to_dict() for doc in uploads]

class FirestoreModel:
    @staticmethod
    def create_upload(user, zip_name, bandit_results=None, ai_results=None):
        db = firestore.client()
        doc_ref = db.collection('uploads').document()
        doc_data = {
            'user': user,
            'zip_name': zip_name,
            'upload_time': datetime.now(),
            'bandit_results': bandit_results or {},
            'ai_results': ai_results or {},
            'doc_id': doc_ref.id  # Document ID'yi ekleyelim
        }
        doc_ref.set(doc_data)
        return doc_data

    @staticmethod
    def get_user_uploads(username):
        db = firestore.client()
        uploads = db.collection('uploads').where('user', '==', username).stream()
        return [
            {
                **doc.to_dict(),
                'doc_id': doc.id  # Her belgeye doc_id ekleyelim
            } 
            for doc in uploads
        ]

    @staticmethod
    def get_file_report(file_id, username):
        db = firestore.client()
        doc_ref = db.collection('uploads').document(file_id)
        doc = doc_ref.get()
        
        if doc.exists:
            data = doc.to_dict()
            # Kullanıcı kontrolü
            if data['user'] == username:
                return {**data, 'doc_id': doc.id}  # doc_id'yi ekleyelim
        return None

    @staticmethod
    def delete_upload(doc_id, username):
        db = firestore.client()
        doc_ref = db.collection('uploads').document(doc_id)
        doc = doc_ref.get()
        
        if doc.exists and doc.to_dict()['user'] == username:
            doc_ref.delete()
            return True
        return False
