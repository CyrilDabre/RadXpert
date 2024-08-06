import os
import json
import uuid
from datetime import datetime
from django.conf import settings
from django.shortcuts import render, redirect, get_object_or_404, get_list_or_404
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .forms import ImageUploadForm
from .models import UploadedImage
from .medsam import get_medsam_model, medsam_inference
from PIL import Image
import numpy as np
import torch
import clip
import spacy
import asyncio
from asgiref.sync import sync_to_async
import torch.nn as nn

# Initialize the MedSAM model once at the start
medsam_model = get_medsam_model()

# Initialize CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Initialize spaCy model
nlp = spacy.load("en_core_web_sm")

class TextFeatureProjector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TextFeatureProjector, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

# Initialize the TextFeatureProjector with input_dim=512 and output_dim=1024
text_projector = TextFeatureProjector(512, 1024).to(device)

def upload_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            now = datetime.now()
            formatted_date = now.strftime('%d%m%Y%H%M%S')
            folder_name = str(uuid.uuid4()) + str(formatted_date)
            files = request.FILES.getlist('image')
            for i, file in enumerate(files):
                image_instance = UploadedImage(image=file, folder_name=folder_name)
                image_instance.save()
            return redirect('segmentation_page', folder_name=folder_name)
    elif 'folder_name' in request.GET:
        folder_name = request.GET['folder_name']
        return redirect('segmentation_page', folder_name=folder_name)
    else:
        form = ImageUploadForm()
    return render(request, 'upload.html', {'form': form})

report_text = "Impression: Multifocal pneumonia     Findings:            Lung Fields:Bilateral - There are opacities in both the right and left lungs. Multifocal - These opacities are present in multiple areas of the lungs, rather than a single concentrated area. Appearance - The opacities may appear hazy, poorly defined, or with ill-defined borders. In some cases, there may be consolidation (complete filling of air sacs with fluid) evident. Distribution - The report doesn't specify the location of the opacities, but they could be in any part of the lungs, upper lobes, lower lobes, or both.Pleura: No evidence of pleural effusion (fluid collection between the lung and chest wall) is seen. Pneumothorax: No pneumothorax (air collection between the lung and chest wall) is identified. Mediastinum: The region in the center of the chest containing the heart, esophagus, and trachea appears normal in size and contour. Hilar Regions: The areas where the major blood vessels and airways enter the lungs (hilum) appear normal in size and contour. Additional Notes: This report suggests pneumonia, an infection that causes inflammation of the lungs. The specific cause of the pneumonia cannot be determined from a chest X-ray alone. Further investigations like sputum culture, blood tests, or a CT scan may be needed to identify the causative organism. The absence of pleural effusion and pneumothorax is a positive finding, indicating less severe complications. Recommendations: Correlation with clinical presentation and symptoms is crucial for diagnosis and further management. Depending on the severity of symptoms, additional tests might be recommended by your doctor. A follow-up chest X-ray may be advised to monitor the response to treatment"

def segmentation_page(request, folder_name):
    folder_path = os.path.join(settings.MEDIA_ROOT, 'uploads', folder_name)
    images = [os.path.join('uploads', folder_name, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    images = sorted(images)
    images = [os.path.join(settings.MEDIA_URL, image).replace('\\', '/') for image in images]
    print("Image paths:", images)  # Debug: print the image paths to check
    # Assuming there's a single report text for all images
    return render(request, 'segmentation.html', {'images': images, 'folder_name': folder_name, 'report_text': report_text})

def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((1024, 1024))
    img_np = np.array(img)
    img_np = (img_np - img_np.min()) / np.clip(img_np.max() - img_np.min(), a_min=1e-8, a_max=None)
    return img_np

def crop_to_bbox(image_path, bbox):
    img = Image.open(image_path)
    left, top, right, bottom = bbox
    cropped_img = img.crop((left, top, right, bottom))
    return preprocess(cropped_img).unsqueeze(0).to(device)

def process_with_clip(image_path, bbox, report_text, color):
    # Load and preprocess the whole image
    whole_image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    # Crop image to bounding box
    cropped_image = crop_to_bbox(image_path, bbox)

    # Encode the whole image
    with torch.no_grad():
        whole_image_features = clip_model.encode_image(whole_image)
    whole_image_features /= whole_image_features.norm(dim=-1, keepdim=True)

    # Encode the cropped image
    with torch.no_grad():
        cropped_image_features = clip_model.encode_image(cropped_image)
    cropped_image_features /= cropped_image_features.norm(dim=-1, keepdim=True)

    # Combine whole image and cropped image features
    combined_image_features = torch.cat((whole_image_features, cropped_image_features), dim=-1)

    # Split report text into manageable chunks
    max_context_length = 77
    sentences = [sent.text for sent in nlp(report_text).sents]
    text_chunks = []
    current_chunk = []

    for sentence in sentences:
        if len(current_chunk) + len(clip.tokenize(sentence)[0]) <= max_context_length:
            current_chunk.append(sentence)
        else:
            text_chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]

    if current_chunk:
        text_chunks.append(" ".join(current_chunk))

    # Process each chunk and calculate similarity scores
    highlighted_text = []
    for chunk in text_chunks:
        text = clip.tokenize(chunk).to(device)
        with torch.no_grad():
            text_features = clip_model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_features = text_projector(text_features)

        # Ensure text_features and combined_image_features have the same dimensions
        text_features = text_features.view(1, -1)  # shape: (1, 1024)
        combined_image_features = combined_image_features.view(1, -1)  # shape: (1, 1024)

        # Calculate similarity scores
        similarity_scores = torch.matmul(combined_image_features, text_features.T).detach().cpu().numpy().tolist()
        
        # Print similarity scores for debugging
        print(f"chunk : {chunk} | Similarity Scores: {similarity_scores}")

        for sentence in chunk.split(". "):
            highlighted_text.append({
                'sentence': sentence,
                'score': similarity_scores[0][0] if len(similarity_scores[0]) > 0 else 0,
                'color': color
            })

    return {
        'embedding': combined_image_features.detach().cpu().numpy().tolist(),
        'highlighted_text': highlighted_text
    }

@csrf_exempt
async def get_segmentation(request, folder_name):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            bbox_coords = data.get('bbox')
            image_index = data.get('imageIndex')
            image_path = data.get('image_path')
            
            # Debug information
            print("Received bbox_coords:", bbox_coords)
            print("Received image_index:", image_index)
            print("Received image_path:", image_path)

            if bbox_coords is None or image_index is None or image_path is None:
                return JsonResponse({'error': 'Bounding box coordinates, image index, or image path not provided'}, status=400)

            # Remove the '/media/' prefix to get the filesystem path
            image_path = image_path.replace(settings.MEDIA_URL, "")                                                        
            full_image_path = os.path.join(settings.MEDIA_ROOT, image_path).replace('\\', '/')
            
            if not os.path.exists(full_image_path):
                return JsonResponse({'error': f'Image path {full_image_path} does not exist'}, status=400)

            bbox_coords = [float(coord) for coord in bbox_coords]
            print("Validated bbox_coords:", bbox_coords)

            image_tensor = torch.tensor(load_and_preprocess_image(full_image_path)).float().permute(2, 0, 1).unsqueeze(0)
            segmented_mask = await asyncio.to_thread(medsam_inference, medsam_model, image_tensor, bbox_coords)

            # Process the whole image and cropped area with CLIP
            color = data.get('color')
            clip_results = await asyncio.to_thread(process_with_clip, full_image_path, bbox_coords, report_text, color)

            return JsonResponse({'mask': segmented_mask.tolist(), 'clip_results': clip_results})
        except Exception as e:
            print("Error during segmentation:", str(e))
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Invalid request method'}, status=400)
