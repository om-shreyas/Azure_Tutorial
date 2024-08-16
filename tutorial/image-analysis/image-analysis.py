#pip install azure-cognitiveservices-vision-computervision
from dotenv import load_dotenv
import os
from array import array
from PIL import Image, ImageDraw
import sys
import time
from matplotlib import pyplot as plt
import numpy as np


# Import namespaces
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials


def main():
    global cv_client

    try:
        # Get Configuration Settings
        load_dotenv()
        cog_endpoint = os.getenv('COG_SERVICE_ENDPOINT')
        cog_key = os.getenv('COG_SERVICE_KEY')

        # Get image
        image_file = 'images/street.jpg'
        if len(sys.argv) > 1:
            image_file = sys.argv[1]

        # Authenticate Azure AI Vision client
        credential = CognitiveServicesCredentials(cog_key)
        cv_client = ComputerVisionClient(cog_endpoint, credential)

        # Analyze image
        AnalyzeImage(image_file)

        # Generate thumbnail
        GetThumbnail(image_file)

    except Exception as ex:
        print(ex)


def AnalyzeImage(image_file):
    print('Analyzing', image_file)
    features = [VisualFeatureTypes.description,
                VisualFeatureTypes.tags,
                VisualFeatureTypes.categories,
                VisualFeatureTypes.brands,
                VisualFeatureTypes.objects,
                VisualFeatureTypes.adult]

    # Get image analysis
    with open(image_file, mode='rb') as image_data:
        analysis = cv_client.analyze_image_in_stream(image_data, features)

    for caption in analysis.description.captions:
        print("Description: '{}'(confidence:{:.2f}%)".format(caption.text, caption.confidence * 100))
    if(len(analysis.tags)>0):
        print("Tags: ")
        for tag in analysis.tags:
            print(f"-{tag.name}- (confidence: {tag.confidence*100:.2f}%)")

    if (len(analysis.categories) > 0):
        print("Categories:")
        landmarks = []
        for category in analysis.categories:
            print(f"-{category.name} (confidence {category.score*100:.2f}%)")
            if category.detail:
                for landmark in category.detail.landmarks:
                    if landmark not in landmarks:
                        landmarks.append(landmark)

    if(len(landmarks)>0):
        print("Landmarks:")
        for landmark in landmarks:
            print(f"-{landmark.name} (confidence {landmark.confidence*100:.2f}%)")

    if len(analysis.brands)>0:
        print("Brands: ")
        for brand in analysis.brands:
            print(f"-{brand.name} (confidence {brand.confidence*100:.2f}%)")

    if len(analysis.objects) > 0:
        print("Objects in image:")
        
        # Prepare image for drawing
        fig = plt.figure(figsize=(8, 8))
        plt.axis('off')
        image = Image.open(image_file)
        draw = ImageDraw.Draw(image)
        color = 'cyan'
        
        for detected_object in analysis.objects:
            # Print object name
            print("{} (confidence: {:.2f}%)".format(detected_object.object_property, detected_object.confidence * 100))
            
            # Draw object bounding box
            r = detected_object.rectangle
            bounding_box = ((r.x, r.y), (r.x + r.w, r.y + r.h))
            draw.rectangle(bounding_box, outline=color, width=3)
            
            # Annotate the object
            plt.annotate(detected_object.object_property, (r.x, r.y), backgroundcolor=color)
        
        # Save annotated image
        plt.imshow(image)
        outputfile = 'objects.jpg'
        fig.savefig(outputfile)
        print('Results saved in', outputfile)

# Print adult content ratings
    ratings = {
        "Is Adult Content": analysis.adult.is_adult_content,
        "Is Racy Content": analysis.adult.is_racy_content,
        "Is Gory Content": analysis.adult.is_gory_content
    }

    print("Ratings:")
    for category, rating in ratings.items():
        print(f"{category}: {rating}")

    with open(image_file, mode="rb") as image_data:
        # Get thumbnail data
        thumbnail_stream = cv_client.generate_thumbnail_in_stream(106, 108, image_data, True)

        # Save thumbnail image
        thumbnail_file_name = "thumbnail.png"
        with open(thumbnail_file_name, "wb") as thumbnail_file:
            for chunk in thumbnail_stream:
                thumbnail_file.write(chunk)
        
        print("Thumbnail saved in", thumbnail_file_name)



def GetThumbnail(image_file):
    print('Generating thumbnail')
    # Generate a thumbnail
    # Add code for thumbnail generation here


if __name__ == "__main__":
    main()
