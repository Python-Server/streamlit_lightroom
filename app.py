import streamlit as st
import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from PIL import Image
import io
from collections import defaultdict

class PresetManager:
    """Handles loading and parsing of XMP preset files with category organization"""
    def __init__(self, preset_dir="Lightroom_Presets"):
        self.preset_dir = preset_dir
        # Dictionary to store presets organized by category
        self.categorized_presets = defaultdict(dict)
        self.load_presets()
    
    def load_presets(self):
        """
        Recursively load all XMP presets from the preset directory and its subdirectories.
        Organizes presets by their subdirectory categories.
        """
        # Walk through all subdirectories
        for root, dirs, files in os.walk(self.preset_dir):
            for filename in files:
                if filename.endswith(".xmp"):
                    # Get the relative path from the preset directory
                    rel_path = os.path.relpath(root, self.preset_dir)
                    # If we're in the root preset directory, use "Uncategorized" as category
                    category = "Uncategorized" if rel_path == "." else rel_path
                    
                    preset_path = os.path.join(root, filename)
                    try:
                        tree = ET.parse(preset_path)
                        root_elem = tree.getroot()
                        
                        # Extract basic adjustments (simplified version)
                        settings = {
                            'exposure': self._get_value(root_elem, 'Exposure2012', 0),
                            'contrast': self._get_value(root_elem, 'Contrast2012', 0),
                            'highlights': self._get_value(root_elem, 'Highlights2012', 0),
                            'shadows': self._get_value(root_elem, 'Shadows2012', 0),
                            'whites': self._get_value(root_elem, 'Whites2012', 0),
                            'blacks': self._get_value(root_elem, 'Blacks2012', 0),
                            'clarity': self._get_value(root_elem, 'Clarity2012', 0),
                            'vibrance': self._get_value(root_elem, 'Vibrance', 0),
                            'saturation': self._get_value(root_elem, 'Saturation', 0),
                            'full_path': preset_path  # Store the full path for reference
                        }
                        
                        # Store preset in the appropriate category
                        self.categorized_presets[category][filename] = settings
                        
                    except ET.ParseError:
                        st.warning(f"Failed to parse preset: {filename}")
    
    def _get_value(self, root, param, default=0):
        """Extract parameter value from XMP file"""
        # Note: Actual XMP parsing would be more complex
        # This is a simplified version for demonstration
        return float(default)
    
    def get_categories(self):
        """Return sorted list of available categories"""
        return sorted(self.categorized_presets.keys())
    
    def get_presets_in_category(self, category):
        """Return sorted list of presets in a specific category"""
        return sorted(self.categorized_presets[category].keys())
    
    def get_preset_settings(self, category, preset_name):
        """Retrieve settings for a specific preset in a category"""
        return self.categorized_presets[category][preset_name]

class ImageProcessor:
    """Handles image processing and preset application"""
    @staticmethod
    def apply_preset(image, settings):
        """Apply preset settings to the image"""
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        
        # Convert to float32 for processing
        img_float = img_array.astype(np.float32) / 255.0
        
        # Apply basic adjustments (simplified version)
        # Exposure
        img_float = img_float * (2 ** settings['exposure'])
        
        # Contrast
        img_float = np.clip((img_float - 0.5) * (1 + settings['contrast']/100) + 0.5, 0, 1)
        
        # Convert back to uint8
        processed = (img_float * 255).astype(np.uint8)
        
        return Image.fromarray(processed)

def main():
    st.title("Image Preset Editor")
    
    # Initialize preset manager
    preset_manager = PresetManager()
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Load and display original image
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_column_width=True)
        
        # Create two columns for preset selection
        col1, col2 = st.columns(2)
        
        with col1:
            # Category selection
            selected_category = st.selectbox(
                "Choose a category",
                options=preset_manager.get_categories()
            )
        
        with col2:
            # Preset selection within the chosen category
            selected_preset = st.selectbox(
                "Choose a preset",
                options=preset_manager.get_presets_in_category(selected_category)
            )
        
        # Get preset settings
        settings = preset_manager.get_preset_settings(selected_category, selected_preset).copy()
        
        # Create sliders for adjusting preset values
        st.subheader("Adjust Settings")
        
        # Organize sliders into three columns for better layout
        col1, col2, col3 = st.columns(3)
        
        with col1:
            settings['exposure'] = st.slider("Exposure", -5.0, 5.0, float(settings['exposure']))
            settings['contrast'] = st.slider("Contrast", -100, 100, int(settings['contrast']))
            settings['highlights'] = st.slider("Highlights", -100, 100, int(settings['highlights']))
        
        with col2:
            settings['shadows'] = st.slider("Shadows", -100, 100, int(settings['shadows']))
            settings['whites'] = st.slider("Whites", -100, 100, int(settings['whites']))
            settings['blacks'] = st.slider("Blacks", -100, 100, int(settings['blacks']))
        
        with col3:
            settings['clarity'] = st.slider("Clarity", -100, 100, int(settings['clarity']))
            settings['vibrance'] = st.slider("Vibrance", -100, 100, int(settings['vibrance']))
            settings['saturation'] = st.slider("Saturation", -100, 100, int(settings['saturation']))
        
        # Process and display preview
        processed_image = ImageProcessor.apply_preset(image, settings)
        st.image(processed_image, caption="Processed Image", use_column_width=True)
        
        # Download button
        if st.button("Download Processed Image"):
            # Convert image to bytes
            buf = io.BytesIO()
            processed_image.save(buf, format="PNG")
            byte_im = buf.getvalue()
            
            st.download_button(
                label="Download Image",
                data=byte_im,
                file_name="processed_image.png",
                mime="image/png"
            )

if __name__ == "__main__":
    main()