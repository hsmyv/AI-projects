





# Part 2: Image Captioning and Summarization using BLIP Pretrained Model

#Load the required libraries
import torch
import tensorflow as tf
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

#load the pretrained BLIP processor and model:
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")








class BlipCaptionSummaryLayer(tf.keras.layers.Layer):
    def __init__(self, processor, model, **kwargs):
        """
        Initialize the custom Keras layer with the BLIP processor and model.

        Args:
            processor: The BLIP processor for preparing inputs for the model.
            model: The BLIP model for generating captions or summaries.
        """
        super().__init__(**kwargs)
        self.processor = processor
        self.model = model

    def call(self, image_path, task):
        # Use tf.py_function to run the custom image processing and text generation
        return tf.py_function(self.process_image, [image_path, task], tf.string)

    def process_image(self, image_path, task):
        """
        Perform image loading, preprocessing, and text generation.

        Args:
            image_path: Path to the image file as a string.
            task: The type of task ("caption" or "summary").

        Returns:
            The generated caption or summary as a string.
        """
        try:
            # Decode the image path from the TensorFlow tensor to a Python string
            image_path_str = image_path.numpy().decode("utf-8")

            # Open the image using PIL and convert it to RGB format
            image = Image.open(image_path_str).convert("RGB")

            # Set the appropriate prompt based on the task
            if task.numpy().decode("utf-8") == "caption":
                prompt = "This is a picture of"  # Modify prompt for more natural output
            else:
                prompt = "This is a detailed photo showing"  # Modify for summary

            # Prepare inputs for the BLIP model
            inputs = self.processor(images=image, text=prompt, return_tensors="pt")

            # Generate text output using the BLIP model
            output = self.model.generate(**inputs)

            # Decode the output into a readable string
            result = self.processor.decode(output[0], skip_special_tokens=True)
            return result
        except Exception as e:
            # Handle errors during image processing or text generation
            print(f"Error: {e}")
            return "Error processing image"
















# Task 8: Implement a Helper Function to Use the Custom Keras Layer
# In this task, you will implement a helper function generate_text that utilizes the custom BlipCaptionSummaryLayer Keras layer to generate captions or summaries for a given image. The function will accept an image path and a task type (caption or summary), process the image using the BLIP model, and return the generated text.

# Steps:
# Create the Helper Function generate_text:
# The function will accept following parameters:

# image_path: The path to the image file (in tensor format).
# task: The type of task to perform, which can either be "caption" or "summary" (in tensor format).
# Inside the function:

# Create an instance(blip_layer) of the BlipCaptionSummaryLayer.
# Call this layer with the provided image path and task type.
# Return the generated caption or summary as the output.







def generate_text(image_path, task):
    # BLIP processor və modeli əvvəlcədən yüklənmiş olmalıdır
    blip_layer = BlipCaptionSummaryLayer(processor, model)

    # Tensor şəklinə sal
    image_path_tensor = tf.convert_to_tensor(image_path, dtype=tf.string)
    task_tensor = tf.convert_to_tensor(task, dtype=tf.string)

    # Layer-ə ötür və nəticəni qaytar
    return blip_layer(image_path_tensor, task_tensor)



# Path to an example image 
image_path = tf.constant("aircraft_damage_dataset_v1/test/dent/144_10_JPG_jpg.rf.4d008cc33e217c1606b76585469d626b.jpg")  # actual path of image

# Generate a caption for the image
caption = generate_text(image_path, tf.constant("caption"))
# Decode and print the generated caption
print("Caption:", caption.numpy().decode("utf-8"))

# Generate a summary for the image
summary = generate_text(image_path, tf.constant("summary"))
# Decode and print the generated summary
print("Summary:", summary.numpy().decode("utf-8"))























# Path to an example image 
image_path = tf.constant("aircraft_damage_dataset_v1/test/dent/144_10_JPG_jpg.rf.4d008cc33e217c1606b76585469d626b.jpg")  # actual path of image

# Generate a caption for the image
caption = generate_text(image_path, tf.constant("caption"))
# Decode and print the generated caption
print("Caption:", caption.numpy().decode("utf-8"))

# Generate a summary for the image
summary = generate_text(image_path, tf.constant("summary"))
# Decode and print the generated summary
print("Summary:", summary.numpy().decode("utf-8"))

# Task 9: Generate a caption for an image using the using BLIP pretrained model
# Use the image_path variable given below to load the image. Run the cell to before proceeding to next step.
# Use the generate_text function to generate a caption for the image.
# Use the example given in 2.2 Generating Captions and Summaries for this task
# Note: Generated captions may not always be accurate, as the model is limited by its training data and may not fully understand new or specific images.

# Şəkli göstərin (artıq göstərmisiniz)
img = plt.imread(image_url)
plt.imshow(img)
plt.axis('off')
plt.show()

# Şəkilin yolu tensor formatında (verilmişdir)
image_path = tf.constant("aircraft_damage_dataset_v1/test/dent/149_22_JPG_jpg.rf.4899cbb6f4aad9588fa3811bb886c34d.jpg")

# generate_text funksiyasını istifadə edərək caption yaradın
caption = generate_text(image_path, "caption")

print("Generated Caption:", caption)



















# Task 10: Generate a summary of an image using BLIP pretrained model
# Use the image_path variable given below to load the image. Run the cell before proceeding to next step.
# Use the generate_text function to generate a caption for the image.
# Use the example given in 2.2 Generating Captions and Summaries for this task
# Note: Generated summary may not always be accurate, as the model is limited by its training data and may not fully understand new or specific images.



# Şəkilin yolu tensor formatında (verilmişdir)
image_path = tf.constant("aircraft_damage_dataset_v1/test/dent/149_22_JPG_jpg.rf.4899cbb6f4aad9588fa3811bb886c34d.jpg")

# generate_text funksiyasını istifadə edərək summary yaradın
summary = generate_text(image_path, tf.constant("summary"))

# Summary-ni çap edin
print("Generated Summary:", summary.numpy().decode("utf-8"))
