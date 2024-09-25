import os
import json
from django.conf import settings
from django.shortcuts import render, HttpResponse, redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.templatetags.static import static

import uuid
from datetime import datetime
from django.conf import settings
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .forms import ImageUploadForm
from .models import UploadedImage
from .medsam import get_medsam_model, medsam_inference
from PIL import Image
import numpy as np
import torch
import open_clip
import spacy
import asyncio
from asgiref.sync import sync_to_async
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
import os
import requests
import json




@login_required(login_url='login')
def dash(request):
    # Get the search query from the GET parameters
    search_query = request.GET.get('q', '')
    # Load the JSON data (ensure the path is correct)
    json_file_path = os.path.join(settings.BASE_DIR, 'history', 'results.json')
    with open(json_file_path) as f:
        R2GENGPT_results = json.load(f)
    # Extract report IDs from the JSON data
    # After loading JSON data
    all_report_ids = list(R2GENGPT_results.keys())


    # Filter report IDs based on the search query
    if search_query:
        filtered_report_ids = [rid for rid in all_report_ids if search_query.lower() in rid.lower()]
    else:
        filtered_report_ids = all_report_ids
   
    context = {
        'report_ids': filtered_report_ids,
        'search_query': search_query,
    }
    return render(request, 'dash.html', context)
    

def regoPage(request):
    if request.method=='POST':
        uname = request.POST.get('username')
        email = request.POST.get('email')
        pass1 = request.POST.get('password1')
        pass2 = request.POST.get('password2')
        
        if pass1 == pass2:
            if User.objects.filter(email=email).exists():
                messages.info(request, 'Email already Used')
                return redirect('register')
            elif User.objects.filter(username=uname).exists():
                messages.info(request, 'Username Already Used')
                return redirect('register')
            else:
                my_user = User.objects.create_user(uname, email, pass1)
                my_user.save()
                return redirect('login')
        else:
            messages.info(request, 'Password not the same')
            return redirect('register')
        
    return render(request, 'rego.html')



def loginPage(request):
    if request.method=='POST':
        username = request.POST.get('username')
        pass1 = request.POST.get('pass')
        user=authenticate(request, username=username, password=pass1)
        if user is not None:
            login(request, user)
            return redirect('dash')
        else:
            messages.info(request, 'Credentials Invalid')
            return redirect('login')
    else:  
        return render(request, 'login.html')



def LogOutpage(request):
    logout(request)
    return redirect('login')


@login_required(login_url='login')
def historyPage(request):
    json_file_path = os.path.join(settings.BASE_DIR, 'history', 'results.json')
    
    with open(json_file_path) as f:
        results_data = json.load(f)
    
    report_ids = list(results_data.keys())
    
    report_id = request.GET.get('report_id')
    report_text = ""
    image_paths = []
    
    if report_id and report_id in results_data:
        report_entry = results_data[report_id]
        
        if isinstance(report_entry, list) and len(report_entry) > 0:
            report_text = report_entry[0]
        
        images_dir = os.path.join(settings.BASE_DIR, 'static', 'images', report_id)
        
        if os.path.isdir(images_dir):
            for filename in os.listdir(images_dir):
                if filename.endswith('.png'):
                    image_paths.append(static(f'images/{report_id}/{filename}'))
    
    context = {
        'report_ids': report_ids,
        'report_text': report_text,
        'selected_report_id': report_id,
        'image_paths': image_paths
    }
    
    return render(request, 'hist.html', context)




##########################################################################


json_file_path = os.path.join(settings.BASE_DIR, 'history', 'results.json')
with open(json_file_path) as f:
    R2GENGPT_results = json.load(f)

# Initialize the MedSAM model once at the start
medsam_model = get_medsam_model()

# Function to extract report text from report ID
@csrf_exempt
def get_report_text(report_id):
    return R2GENGPT_results[report_id][0]
@csrf_exempt
def generate_gemeni_text(prompt, api_key):
    api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=" + api_key
    
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }]
    }
    print(prompt)
    response = requests.post(api_url, headers=headers, json=data)
    if response.status_code == 200:
        data = response.json()  # parse the JSON data from the response
        text = data['candidates'][0]['content']['parts'][0]['text']
        return text
    else:
        # Print the full error response for more detailed debugging
        print(response.text)  
        raise Exception(f"Request failed with status code: {response.status_code}")
        
@csrf_exempt
def enhance_report(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            folder_name = data.get('folder_name')
            
            if not folder_name:
                return JsonResponse({'error': 'folder name not provided'}, status=400)
            report_text =  get_report_text(folder_name) 
            
            api_key = os.environ.get("GEMINI_API_KEY") 
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set.")

            user_type = "medical"
            prompt_suffix = " Please provide a detailed report that includes: A clear summary of the findings, An explanation of the implications of each finding, A discussion of the potential causes of any observed abnormalities, A conclusion that ties together the findings and impression, Use a clear and concise format, with headings and bullet points as needed. Assume the reader is a "+user_type+" professional."
            prompt = "Expand on the following chest X-ray report: "+report_text+ prompt_suffix

            # Enhance the report text using Gemini AI
            enhanced_text = generate_gemeni_text(prompt, api_key)
            
            # Save the enhanced text in a JSON file
            save_path = os.path.join(settings.MEDIA_ROOT, 'enhanced_reports', f'{folder_name}.json')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            with open(save_path, 'w') as f:
                json.dump({'enhanced_text': enhanced_text}, f)

            return JsonResponse({'enhanced_text': enhanced_text})
        except Exception as e:
            print(f"Error enhancing report: {e}")
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Invalid request method'}, status=400)

# DO NOT HARDCODE YOUR API KEY IN THE CODE. 
# Instead, store it as an environment variable for security.

# Initialize BiomedCLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
clip_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(model_name)
clip_model.to(device)
tokenizer = open_clip.get_tokenizer(model_name)

# Initialize spaCy model
nlp = spacy.load("en_core_web_sm")


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
                image_instance.save
                
            return redirect('segmentation_page', folder_name=folder_name)
    elif 'folder_name' in request.GET:
        folder_name = request.GET['folder_name']
        return redirect('segmentation_page', folder_name=folder_name)
    else:
        form = ImageUploadForm()
    return render(request, 'upload.html', {'form': form})
@csrf_exempt
def segmentation_page(request, folder_name):
    folder_path = os.path.join(str(settings.BASE_DIR), 'static', 'images', folder_name)
    images = [os.path.join('images', folder_name, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    images = sorted(images)
    images = [os.path.join(settings.STATIC_URL, image).replace('\\', '/') for image in images]
    print("Image paths:", images)  # Debug: print the image paths to check
    report_text = get_report_text(folder_name)
    return render(request, 'segmentation.html', {'images': images, 'folder_name': folder_name, 'report_text': report_text})

@csrf_exempt
def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((1024, 1024))
    img_np = np.array(img)
    img_np = (img_np - img_np.min()) / np.clip(img_np.max() - img_np.min(), a_min=1e-8, a_max=None)
    return img_np
    
@csrf_exempt
def crop_to_bbox(image_path, bbox):
    img = Image.open(image_path)
    left, top, right, bottom = bbox
    cropped_img = img.crop((left, top, right, bottom))
    return preprocess_val(cropped_img).unsqueeze(0).to(device)
@csrf_exempt
def process_with_biomedclip(image_path, bbox, report_text, color):
    try:
        # Crop image to bounding box
        cropped_image = crop_to_bbox(image_path, bbox)

        # Encode the cropped image
        with torch.no_grad():
            cropped_image_features = clip_model.encode_image(cropped_image)
        cropped_image_features /= cropped_image_features.norm(dim=-1, keepdim=True)

        # Split report text into manageable chunks
        max_context_length = 77
        sentences = [sent.text for sent in nlp(report_text).sents]
        text_chunks = []
        current_chunk = []

        for sentence in sentences:
            if len(current_chunk) + len(tokenizer(sentence)[0]) <= max_context_length:
                current_chunk.append(sentence)
            else:
                text_chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]

        if current_chunk:
            text_chunks.append(" ".join(current_chunk))

        # Process each chunk and calculate similarity scores
        highlighted_text = []
        for chunk in text_chunks:
            text = tokenizer(chunk).to(device)
            with torch.no_grad():
                text_features = clip_model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            
            # Ensure text_features and cropped_image_features have compatible dimensions
            text_features = text_features.view(-1)  # shape: (512,)
            cropped_image_features = cropped_image_features.view(-1)  # shape: (512,)

            # Calculate similarity scores
            similarity_score = torch.dot(cropped_image_features, text_features).item()
            #print(f"chunk : {chunk} | Similarity Scores: {similarity_score}")

            for sentence in chunk.split(". "):
                highlighted_text.append({
                    'sentence': sentence,
                    'score': similarity_score,
                    'color': color
                })

        return {
            'embedding': cropped_image_features.detach().cpu().numpy().tolist(),
            'highlighted_text': highlighted_text
        }
    except Exception as e:
        print(f"Error in process_with_biomedclip: {e}")
        raise

@csrf_exempt
async def get_segmentation(request, folder_name):
    if request.method == 'POST':
        try:
            # Parse incoming JSON data
            print(f"Received request body: {request.body}")
            try:
                data = json.loads(request.body)
            except json.JSONDecodeError as e:
                print(f"JSON Decode Error: {e}")
                return JsonResponse({'error': 'Invalid JSON data'}, status=400)
                
            bbox_coords = data.get('bbox')
            image_index = data.get('image_index')
            image_path = data.get('image_path')

            # Log data to debug
            print(f"Received BBox: {bbox_coords}")
            print(f"Received Image Index: {image_index}")
            print(f"Received Image Path: {image_path}")
            print(f"Folder Name: {folder_name}")

            # Fetch report text associated with the folder
            report_text = get_report_text(folder_name)
            print("Report Text:", report_text)

            # Remove the media prefix to construct the full image path
            image_path = image_path.replace('/', '', 1) 
            # Join MEDIA_ROOT and image_path correctly
            full_image_path = os.path.join(settings.BASE_DIR, image_path).replace('\\', '/')

            print(f"Full Image Path: {full_image_path}")


            # Check if the image exists at the given path
            if not os.path.exists(full_image_path):
                return JsonResponse({'error': f'Image path {full_image_path} does not exist'}, status=400)

            # Ensure bbox coordinates are correctly formatted
            bbox_coords = [float(coord) for coord in bbox_coords]
            print(f"Formatted BBox Coords: {bbox_coords}")

            # Preprocess the image and run MedSAM inference asynchronously
            image_tensor = torch.tensor(load_and_preprocess_image(full_image_path)).float().permute(2, 0, 1).unsqueeze(0)
            segmented_mask = await asyncio.to_thread(medsam_inference, medsam_model, image_tensor, bbox_coords)

            print("Segmentation completed successfully")

            # Process the cropped image area with BiomedCLIP asynchronously
            color = data.get('color')  # Ensure the bounding box color is provided
            clip_results = await asyncio.to_thread(process_with_biomedclip, full_image_path, bbox_coords, report_text, color)

            print("BiomedCLIP processing completed successfully")

            # Return the segmented mask and clip results
            return JsonResponse({'index': image_index,'mask': segmented_mask.tolist(), 'clip_results': clip_results})

        except KeyError as ke:
            print(f"Key error: {ke}")
            return JsonResponse({'error': f'Missing key: {str(ke)}'}, status=400)

        except Exception as e:
            print(f"Error during segmentation: {e}")
            return JsonResponse({'error': f'Error during segmentation: {str(e)}'}, status=500)

    # Return error if the request method is not POST
    return JsonResponse({'error': 'Invalid request method'}, status=400)