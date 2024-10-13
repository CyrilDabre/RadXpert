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
from bs4 import BeautifulSoup




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

@login_required(login_url="login")
def patientdash(request):
    # Path to the results.json file (make sure this path is correct)
    json_file_path = os.path.join(settings.BASE_DIR, "history", "published_reports.json")

    # Check if the JSON file exists
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"JSON file not found at: {json_file_path}")

    # Read the JSON data from the file
    with open(json_file_path) as f:
        results_data = json.load(f)

    # Initialize variables
    report_text = ""
    image_paths = []
    selected_report_id = request.GET.get("referenceID")

    # If the user entered a reference ID, find the report and associated images
    if selected_report_id and selected_report_id in results_data:
        report_text = results_data[selected_report_id][0]  # Get the report text

        # Path to the folder containing images for the selected report ID
        images_dir = os.path.join(
            settings.BASE_DIR, "static", "images", selected_report_id
        )

        if os.path.isdir(images_dir):
            # Loop through files in the directory and collect image paths
            for filename in sorted(os.listdir(images_dir)):
                if filename.endswith(".png"):
                    image_paths.append(f"images/{selected_report_id}/{filename}")

    context = {
        "report_text": report_text,
        "selected_report_id": selected_report_id,
        "image_paths": image_paths,
    }

    return render(request, "patientdash.html", context)

def loginPage(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("pass")
        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            if user.profile.usertype == "doctor":
                return redirect("dash")
            else:
                return redirect("patientdash")
        else:
            messages.info(request, "Invalid Credentials")
            return redirect("login")
    else:
        return render(request, "login.html")

def LogOutpage(request):
    logout(request)
    return redirect('login')


@login_required(login_url='login')
def historyPage(request):
    json_file_path = os.path.join(settings.BASE_DIR, 'history', 'published_reports.json')
    
    # Check if the file exists
    if not os.path.exists(json_file_path):
        print(f"File not found: {json_file_path}")
        return HttpResponse("JSON file not found", status=404)

    with open(json_file_path) as f:
        results_data = json.load(f)

    report_ids = list(results_data.keys())
    
    # Debug print to check if report IDs are loaded
    print(f"Report IDs loaded: {report_ids}")

    # Automatically choose the first report if no report ID is provided or found
    report_id = request.GET.get('report_id')
    if report_id is None or report_id not in results_data:
        report_id = report_ids[0] if report_ids else None
    
    report_text = ""
    image_paths = []
    
    if report_id:
        report_text = results_data.get(report_id, "Report not found")
        print(f"Report text for {report_id}: {report_text}")
        
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






#-----------------------------------------------------------------------------------------------

# Load and preprocess image for BiomedCLIP processing
def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((1024, 1024))
    img_np = np.array(img)
    img_np = (img_np - img_np.min()) / np.clip(img_np.max() - img_np.min(), a_min=1e-8, a_max=None)
    return img_np

# Crop image to bounding box for BiomedCLIP
def crop_to_bbox(image_path, bbox):
    img = Image.open(image_path)
    left, top, right, bottom = bbox
    cropped_img = img.crop((left, top, right, bottom))
    return preprocess_val(cropped_img).unsqueeze(0).to(device)

# Process image with BiomedCLIP for report analysis
def process_with_biomedclip(image_path, bbox, report_text, color):
    try:
        cropped_image = crop_to_bbox(image_path, bbox)

        # Encode the cropped image
        with torch.no_grad():
            cropped_image_features = clip_model.encode_image(cropped_image)
        cropped_image_features /= cropped_image_features.norm(dim=-1, keepdim=True)

        # Split report text into chunks for BiomedCLIP processing
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

            text_features = text_features.view(-1)
            cropped_image_features = cropped_image_features.view(-1)

            similarity_score = torch.dot(cropped_image_features, text_features).item()

            for sentence in chunk.split(". "):
                highlighted_text.append({
                    'sentence': sentence,
                    'score': similarity_score
                })

        # Generate HTML for original and enhanced report based on new logic
        report_html = generate_highlighted_html(highlighted_text, color)

        return {
            'report_html': report_html  # Return the generated HTML
        }

    except Exception as e:
        print(f"Error in process_with_biomedclip: {e}")
        raise


# Main segmentation and report analysis view
@csrf_exempt
async def get_segmentation(request, folder_name):
    if request.method == 'POST':
        try:
            # Parse incoming request data
            data = json.loads(request.body)
            bbox_coords = data.get('bbox')
            image_index = data.get('image_index')
            image_path = data.get('image_path')

            # Fetch original and enhanced report text
            original_report_text = data.get('original_report')
            enhanced_report_text =data.get('enhanced_report')

            # Construct full image path
            image_path = image_path.replace('/', '', 1)
            full_image_path = os.path.join(settings.BASE_DIR, image_path).replace('\\', '/')

            if not os.path.exists(full_image_path):
                return JsonResponse({'error': f'Image path {full_image_path} does not exist'}, status=400)

            bbox_coords = [float(coord) for coord in bbox_coords]

            # Preprocess the image and run MedSAM inference
            image_tensor = torch.tensor(load_and_preprocess_image(full_image_path)).float().permute(2, 0, 1).unsqueeze(0)
            segmented_mask = await asyncio.to_thread(medsam_inference, medsam_model, image_tensor, bbox_coords)

            color = data.get('color')

            # Process original report with BiomedCLIP
            original_clip_results = await asyncio.to_thread(process_with_biomedclip, full_image_path, bbox_coords, original_report_text, color)

            # Process enhanced report with BiomedCLIP
            enhanced_clip_results = {}
            if enhanced_report_text:
                enhanced_clip_results = await asyncio.to_thread(process_with_biomedclip, full_image_path, bbox_coords, enhanced_report_text, color)

            return JsonResponse({
                'index': image_index,
                'mask': segmented_mask.tolist(),
                'original_clip_results': original_clip_results,
                'enhanced_clip_results': enhanced_clip_results
            })

        except Exception as e:
            print(f"Error during segmentation: {e}")
            return JsonResponse({'error': f'Error during segmentation: {str(e)}'}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=400)

# Save the report when the save button is clicked
@csrf_exempt
def save_enhanced_report(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            folder_name = data.get('folder_name')
            enhanced_report = data.get('enhanced_report')

            # Save the enhanced report to the corresponding JSON file
            enhanced_report_path = os.path.join(settings.MEDIA_ROOT, 'enhanced_reports', f'{folder_name}.json')
            os.makedirs(os.path.dirname(enhanced_report_path), exist_ok=True)

            with open(enhanced_report_path, 'w') as f:
                json.dump({'enhanced_report': enhanced_report}, f)

            return JsonResponse({'success': True})

        except Exception as e:
            print(f"Error saving enhanced report: {e}")
            return JsonResponse({'success': False, 'error': str(e)})
    else:
        return JsonResponse({'success': False, 'error': 'Invalid request method'})

@csrf_exempt
def publish_report(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            folder_name = data.get('folder_name')
            enhanced_report = data.get('enhanced_report')
            enhanced_report = remove_highlights(enhanced_report)
            enhanced_report = enhanced_report.replace('\\', "").strip()    
            # Remove original report from the results.json
            original_report_path = os.path.join(settings.BASE_DIR, 'history', 'results.json')
            with open(original_report_path, 'r') as f:
                results_data = json.load(f)

            if folder_name in results_data:
                del results_data[folder_name]

                with open(original_report_path, 'w') as f:
                    json.dump(results_data, f, indent=4)

            # Add the enhanced report to a new "published_reports.json"
            published_report_path = os.path.join(settings.BASE_DIR, 'history', 'published_reports.json')
            if not os.path.exists(published_report_path):
                with open(published_report_path, 'w') as f:
                    json.dump({}, f)  # Create empty JSON file if not exists

            with open(published_report_path, 'r+') as f:
                published_data = json.load(f)
                published_data[folder_name] = enhanced_report
                f.seek(0)
                json.dump(published_data, f, indent=4)

            return JsonResponse({'success': True})

        except Exception as e:
            print(f"Error publishing report: {e}")
            return JsonResponse({'success': False, 'error': str(e)})
    else:
        return JsonResponse({'success': False, 'error': 'Invalid request method'})


def generate_highlighted_html(highlighted_text, color):
    html_content = ""

    # Get RGB values from color
    rgb = getRgbFromColor(color)

    # Get scores and sort them
    scores = [line_data['score'] for line_data in highlighted_text]
    sorted_scores = sorted(scores, reverse=True)

    # Generate the full report with different shades of the passed color
    for i, line_data in enumerate(highlighted_text):
        sentence = line_data['sentence']
        score = line_data['score']

        # Calculate rank
        rank = sorted_scores.index(score) + 1

        # Calculate opacity based on ranking (0-1)
        opacity = 1 - (rank * 0.15)  # 0.15 difference between each shade

        # Highlighted sentence with varying opacity background
        html_content += f'<span style="background-color: rgba{rgb + (opacity,)}; padding: 2px 5px; margin: 2px 0; display: inline-block;">{sentence}.</span><br>'

    return html_content


# Helper function to get RGB from color
def getRgbFromColor(color):
    # Assume `color` is a hex value, convert to RGB
    color = color.lstrip('#')
    return tuple(int(color[i:i+2], 16) for i in (0, 2, 4))


def remove_highlights(html_content):
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Remove span elements with background color styles
    for tag in soup.find_all('span'):
        if 'style' in tag.attrs and 'background-color' in tag['style']:
            tag.unwrap()  # Remove the span element and keep its contents

    # Remove br tags (if desired)
    for br in soup.find_all('br'):
        br.unwrap()  # or br.decompose() to completely remove the tag

    # Return cleaned HTML as a string
    return str(soup)

