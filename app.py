from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import pandas as pd
import logging
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import json
import re
from datetime import datetime
from typing import Dict, List, Optional
import io
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
import uuid
import asyncio
import time
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from google import genai
from google.genai import types
from dataclasses import dataclass
import zipfile
from langchain.schema import HumanMessage, SystemMessage
# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@dataclass
class CarouselContent:
    """Data class to hold complete carousel content."""
    hook_title: str
    hook_content: str
    hook_image_prompt: str
    slide2_title: str
    slide2_content: str
    slide2_image_prompt: str
    slide3_title: str
    slide3_content: str
    slide3_image_prompt: str
    slide4_title: str
    slide4_content: str
    slide4_image_prompt: str
    cta_title: str
    cta_content: str
    cta_image_prompt: str
    overall_theme: str
    narrative_flow: str
    topic_category: str

class CarouselGenerator:
    def __init__(self, groq_api_key: str, google_api_key: str):
        """Initialize the Carousel Generator."""
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama3-8b-8192"
        )
        # Initialize Google Generative AI
        self.gemini = genai.Client(api_key=google_api_key)
        
        # Create output directory
        os.makedirs("instagram_carousels", exist_ok=True)
    
    def generate_carousel_topics(self, preferences: Dict) -> List[Dict]:
        """Generate Instagram carousel topics based on user preferences."""
        
        # Build the dynamic prompt based on preferences
        prompt_template = PromptTemplate.from_template(
            """You are a content strategist and expert in {niche} with deep knowledge of {specific_topic}.

Your goal: {goal} through engaging Instagram carousel posts.

Target Audience Details:
- Age Group: {age_group}
- Skill Level: {skill_level}
- Interests: {specific_topic} within {niche}

Content Requirements:
- Generate {num_topics} carousel post ideas
- Tone: {tone}
- Focus Areas: {focus_areas}
- Hook: Create curiosity and grab attention for the target audience
- Each slide should flow naturally to the next
- Content should be valuable, actionable, and appropriate for the skill level
- CTAs should be: {cta_style}

Content Structure (5 slides total):
1. Hook (attention-grabbing opener)
2. Content Slide 1 (main concept/point)
3. Content Slide 2 (supporting detail/example)
4. Content Slide 3 (additional insight/tip)
5. CTA (call-to-action)

Format output as ONLY valid JSON with this exact structure:
{{
    "topics": [
        {{
            "number": 1,
            "hook": "...",
            "slide2": "...",
            "slide3": "...",
            "slide4": "...",
            "cta": "..."
        }}
        // ... continue for all {num_topics} topics
    ]
}}

Additional Guidelines:
- Make content specific to {specific_topic}
- Ensure each post delivers genuine value
- Vary the hook styles (questions, bold statements, surprising facts, etc.)
- Keep content beginner-friendly while being valuable
- No code snippets or technical screenshots mentioned
- Focus on practical, actionable advice

Respond with ONLY valid JSON. Do not include any markdown, explanations, or comments. The response must start with '{{' and end with '}}'. If you cannot produce valid JSON, respond with an empty JSON object: {{}}
"""
        )
        
        try:
            # Create LLM chain
            chain = LLMChain(llm=self.llm, prompt=prompt_template)
            
            # Generate topics
            logger.info(f"Generating {preferences['num_topics']} carousel topics for {preferences['specific_topic']}...")
            response = chain.run(**preferences)
            
            # Try to extract the first valid JSON object from the response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                json_str = json_match.group(0)
                try:
                    result = json.loads(json_str)
                except Exception as e:
                    logger.error(f"JSON decode error: {e}")
                    return []
                topics = result.get('topics', [])
                logger.info(f"Successfully generated {len(topics)} topics")
                return topics
            else:
                logger.error("No valid JSON found in response")
                return []
                
        except Exception as e:
            logger.error(f"Error generating topics: {e}")
            return []

    def get_category_specific_prompts(self, topic_category: str) -> Dict:
        """Get category-specific styling and prompts using LLM."""
        
        system_prompt = """You are a design expert. Generate Instagram carousel styling configuration.

Return ONLY valid JSON with this exact structure (no extra text, no explanations):
{
"style": "visual style description",
"tone": "communication tone",
"elements": "design elements",
"colors": "color palette"
}

Use double quotes only. No trailing commas. Make it category-specific."""

        user_prompt = f"Generate JSON styling config for: {topic_category}"

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm(messages)
            response_text = response.content.strip()
            
            # More aggressive cleaning
            response_text = response_text.replace('```json', '').replace('```', '')
            response_text = response_text.strip()
            
            # Find JSON object bounds
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_text = response_text[start_idx:end_idx]
                
                # Clean up common JSON issues
                json_text = json_text.replace("'", '"')  # Single to double quotes
                json_text = json_text.replace(',}', '}')  # Remove trailing commas
                json_text = json_text.replace(',]', ']')   # Remove trailing commas in arrays

                # Remove extra quotes inside color arrays or values
                # e.g. '"colors": "\'Navy Blue\', \'White\', and \'Light Gray\'"' -> '"colors": "Navy Blue, White, and Light Gray"'
                json_text = re.sub(r'\\?"colors\\?"\s*:\s*"(.*?)"', lambda m: '"colors": "' + re.sub(r"[\'\"]", '', m.group(1)) + '"', json_text)

                config = json.loads(json_text)
                
                # Validate required keys exist
                required_keys = ['style', 'tone', 'elements', 'colors']
                if all(key in config for key in required_keys):
                    return config
                else:
                    raise ValueError("Missing required keys")
            else:
                raise ValueError("No JSON object found")
                
        except Exception as e:
            print(f"LLM response for {topic_category}: {response.content if 'response' in locals() else 'No response'}")
            print(f"Error: {e}")
            # Return fallback general config
            return {
                "style": "modern, clean, professional",
                "tone": "informative, engaging",
                "elements": "clean, modern elements",
                "colors": "balanced color palette"
            }

    def generate_complete_carousel_content(self, topic_data: Dict, preferences: Dict, topic_category: str = "general") -> CarouselContent:
        """Generate all carousel content with full context understanding for any topic."""
        
        category_config = self.get_category_specific_prompts(topic_category)
        
        complete_carousel_prompt = PromptTemplate.from_template(
            """You are creating a cohesive 5-slide Instagram carousel for any topic/industry. 
            Each slide must flow naturally into the next, building a complete narrative.
            
            USER PREFERENCES:
            - Niche: {niche}
            - Specific Topic: {specific_topic}
            - Goal: {goal}
            - Target Age: {age_group}
            - Skill Level: {skill_level}
            - Tone: {tone}
            - CTA Style: {cta_style}
            
            TOPIC CATEGORY: {topic_category}
            TOPIC CONCEPTS:
            - Hook (Slide 1): "{hook_concept}"
            - Content Slide 2: "{slide2_concept}" 
            - Content Slide 3: "{slide3_concept}"
            - Content Slide 4: "{slide4_concept}"
            - CTA (Slide 5): "{cta_concept}"
            
            CATEGORY STYLE GUIDE:
            - Visual Style: {visual_style}
            - Tone: {content_tone}
            - Design Elements: {design_elements}
            
            REQUIREMENTS:
            1. Create a unified narrative that flows from hook → education → solution → action
            2. Each slide should reference or build upon previous slides
            3. Maintain consistent terminology and theme throughout
            4. Adapt content style to match the user's preferences and topic category
            5. Hook should create curiosity, content slides should educate progressively, CTA should convert
            6. Use transition phrases that connect slides naturally
            7. Make content relevant and valuable for the specific topic/industry
            8. Personalize content based on user's target audience ({age_group}, {skill_level})
            
            SLIDE SPECIFICATIONS:
            - Hook: Title (4-6 words, ALL CAPS), Content (8-12 words only that create curiosity)
            - Content Slides: Title (4-6 words only, ALL CAPS), Content (20-40 words only, educational but engaging)
            - CTA: Title (3-5 words, ALL CAPS), Content (8-16 words, clear call-to-action matching {cta_style})
            
            Create a cohesive carousel where someone reading all 5 slides gets a complete, flowing story.
            
            Respond with ONLY valid JSON in this exact format:
            {{
                "overall_theme": "One sentence describing the main theme",
                "narrative_flow": "Brief description of how slides connect",
                "topic_category": "{niche}",
                "hook": {{
                    "title": "HOOK TITLE HERE",
                    "content": "Hook content here",
                    "image_prompt": "A detailed prompt for generating a relevant background image suitable for Instagram carousel (4:5 aspect ratio). Style: {visual_style}. Colors: {color_scheme}. Focus visual elements on the RIGHT side, keeping LEFT area suitable for text overlay. Modern, professional, and visually appealing. NO TEXT OR WORDS IN THE IMAGE. Topic: {hook_concept}. Include realistic elements that create curiosity and intrigue about the topic."
                }},
                "slide2": {{
                    "title": "SLIDE 2 TITLE",
                    "content": "Slide 2 content that builds on the hook",
                    "image_prompt": "A detailed prompt for generating a relevant background image suitable for Instagram carousel (4:5 aspect ratio). Style: {visual_style}. Colors: {color_scheme}. Focus visual elements on the TOP area, keeping BOTTOM area clean for text overlay. Modern, professional, and visually appealing. NO TEXT OR WORDS IN THE IMAGE. Topic: {slide2_concept}. Include realistic elements related to the topic."
                }},
                "slide3": {{
                    "title": "SLIDE 3 TITLE", 
                    "content": "Slide 3 content that continues the story",
                    "image_prompt": "A detailed prompt for generating a relevant background image suitable for Instagram carousel (4:5 aspect ratio). Style: {visual_style}. Colors: {color_scheme}. Focus visual elements on the TOP area, keeping BOTTOM area clean for text overlay. Modern, professional, and visually appealing. NO TEXT OR WORDS IN THE IMAGE. Topic: {slide3_concept}. Include realistic elements related to the topic."
                }},
                "slide4": {{
                    "title": "SLIDE 4 TITLE",
                    "content": "Slide 4 content that leads to the CTA",
                    "image_prompt": "A detailed prompt for generating a relevant background image suitable for Instagram carousel (4:5 aspect ratio). Style: {visual_style}. Colors: {color_scheme}. Focus visual elements on the TOP area, keeping BOTTOM area clean for text overlay. Modern, professional, and visually appealing. NO TEXT OR WORDS IN THE IMAGE. Topic: {slide4_concept}. Include realistic elements related to the topic."
                }},
                "cta": {{
                    "title": "CTA TITLE",
                    "content": "Clear call-to-action that follows naturally",
                    "image_prompt": "A detailed prompt for generating a relevant background image suitable for Instagram carousel (4:5 aspect ratio). Style: {visual_style}. Colors: {color_scheme}. Focus visual elements on the RIGHT side, keeping LEFT area suitable for text overlay. Modern, professional, and visually appealing. NO TEXT OR WORDS IN THE IMAGE. Topic: {cta_concept}. Include realistic elements that encourage action."
                }}
            }}
            """
        )
        
        content_chain = LLMChain(llm=self.llm, prompt=complete_carousel_prompt)
        response = content_chain.run(
            niche=preferences.get('niche', ''),
            specific_topic=preferences.get('specific_topic', ''),
            goal=preferences.get('goal', ''),
            age_group=preferences.get('age_group', ''),
            skill_level=preferences.get('skill_level', ''),
            tone=preferences.get('tone', ''),
            cta_style=preferences.get('cta_style', ''),
            topic_category=topic_category,
            hook_concept=topic_data['hook'],
            slide2_concept=topic_data['slide2'],
            slide3_concept=topic_data['slide3'],
            slide4_concept=topic_data['slide4'],
            cta_concept=topic_data['cta'],
            visual_style=category_config.get('style', 'modern, clean, professional'),
            content_tone=category_config.get('tone', 'informative, engaging'),
            design_elements=category_config.get('elements', 'clean, modern elements'),
            color_scheme=category_config.get('colors', 'balanced color palette')
        ).strip()
        print(f"Generated carousel content: {response}")

        # Improved JSON extraction and cleaning
        try:
            # Remove any markdown code block markers
            cleaned = response.replace('```json', '').replace('```', '').strip()
            # Find the first '{' and last '}' to extract the JSON object
            start = cleaned.find('{')
            end = cleaned.rfind('}')
            if start == -1 or end == -1:
                raise ValueError("No JSON object found in response")
            json_str = cleaned[start:end+1]

            # Remove trailing commas before } or ]
            json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
            # Fix common LLM mistakes: duplicate keys, misplaced commas, etc.
            # Optionally, you can add more cleaning here if needed

            result = json.loads(json_str)
            # Extract and validate all content
            carousel_content = CarouselContent(
                hook_title=result.get("hook", {}).get("title", "DISCOVER MORE"),
                hook_content=result.get("hook", {}).get("content", "Learn something amazing today"),
                hook_image_prompt=result.get("hook", {}).get("image_prompt", "Professional background image"),
                slide2_title=result.get("slide2", {}).get("title", "UNDERSTANDING BASICS"),
                slide2_content=result.get("slide2", {}).get("content", "Let's explore the fundamentals"),
                slide2_image_prompt=result.get("slide2", {}).get("image_prompt", "Educational background image"),
                slide3_title=result.get("slide3", {}).get("title", "KEY INSIGHTS"),
                slide3_content=result.get("slide3", {}).get("content", "Here are the important points"),
                slide3_image_prompt=result.get("slide3", {}).get("image_prompt", "Informative background image"),
                slide4_title=result.get("slide4", {}).get("title", "PRACTICAL TIPS"),
                slide4_content=result.get("slide4", {}).get("content", "Apply these strategies effectively"),
                slide4_image_prompt=result.get("slide4", {}).get("image_prompt", "Action-oriented background image"),
                cta_title=result.get("cta", {}).get("title", "TAKE ACTION"),
                cta_content=result.get("cta", {}).get("content", "Start your journey today"),
                cta_image_prompt=result.get("cta", {}).get("image_prompt", "Call-to-action background image"),
                overall_theme=result.get("overall_theme", "Educational content"),
                narrative_flow=result.get("narrative_flow", "Progressive learning journey"),
                topic_category=topic_category
            )
            return carousel_content

        except Exception as e:
            logger.error(f"Error parsing JSON: {e}\nRaw response: {response}")
            # Return fallback content
            return CarouselContent(
                hook_title="DISCOVER MORE",
                hook_content="Learn something amazing today",
                hook_image_prompt="Professional background image",
                slide2_title="UNDERSTANDING BASICS",
                slide2_content="Let's explore the fundamentals",
                slide2_image_prompt="Educational background image",
                slide3_title="KEY INSIGHTS", 
                slide3_content="Here are the important points",
                slide3_image_prompt="Informative background image",
                slide4_title="PRACTICAL TIPS",
                slide4_content="Apply these strategies effectively",
                slide4_image_prompt="Action-oriented background image",
                cta_title="TAKE ACTION",
                cta_content="Start your journey today",
                cta_image_prompt="Call-to-action background image",
                overall_theme="Educational content",
                narrative_flow="Progressive learning",
                topic_category=topic_category
            )

    def generate_image(self, prompt: str, slide_type: str, topic_category: str = "general") -> str:
        """Generate a carousel image using Google Generative AI with category-specific styling."""
        try:
            category_config = self.get_category_specific_prompts(topic_category)
            
            # Enhance prompt based on slide type and category
            if slide_type in ["hook", "cta"]:
                enhanced_prompt = f"{prompt} Generate as a portrait image (4:5 aspect ratio, 1080x1350 pixels). Style: {category_config.get('style', 'modern, professional')}. Colors: {category_config.get('colors', 'balanced palette')}. High quality, Instagram carousel-ready. Suitable for mobile viewing. IMPORTANT: Keep main visual elements on the RIGHT SIDE of the image only, with the LEFT side being cleaner/darker for text overlay. Modern, professional design. NO TEXT OR WORDS IN THE IMAGE."
            else:  # content slides
                enhanced_prompt = f"{prompt} Generate as a portrait image (4:5 aspect ratio, 1080x1350 pixels). Style: {category_config.get('style', 'modern, professional')}. Colors: {category_config.get('colors', 'balanced palette')}. High quality, Instagram carousel-ready. Suitable for mobile viewing. IMPORTANT: Keep main visual elements on the TOP area of the image, with the BOTTOM area being cleaner/darker for text overlay. Modern, professional design. NO TEXT OR WORDS IN THE IMAGE."
            
            response = self.gemini.models.generate_content(
                model="gemini-2.0-flash-exp-image-generation",
                contents=enhanced_prompt,
                config=types.GenerateContentConfig(
                    response_modalities=['Text', 'Image']
                )
            )
            
            image_data = None
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    image_data = part.inline_data.data
                    break
            
            if image_data is None:
                raise ValueError("No image was generated")
            
            timestamp = int(time.time())
            image_path = f"instagram_carousels/bg_{timestamp}.png"
            image = Image.open(BytesIO(image_data))
            
            # Resize to carousel dimensions (4:5 aspect ratio)
            image = image.resize((1080, 1350), Image.Resampling.LANCZOS)
            image.save(image_path)
            
            return image_path
            
        except Exception as e:
            logger.error(f"Error generating image: {str(e)}")
            return self.create_simple_background(slide_type, topic_category)

    def create_simple_background(self, slide_type: str = "hook", topic_category: str = "general") -> str:
        """Create a simple background as fallback for carousel format with category-specific styling."""
        category_config = self.get_category_specific_prompts(topic_category)
        
        # Define category-specific color schemes
        color_schemes = {
            "technology": [(10, 15, 35), (20, 40, 80), (40, 80, 160)],
            "health": [(15, 30, 20), (40, 100, 60), (80, 160, 120)],
            "business": [(20, 20, 35), (40, 40, 80), (80, 80, 160)],
            "education": [(35, 25, 15), (80, 60, 40), (160, 120, 80)],
            "lifestyle": [(35, 20, 25), (80, 50, 60), (160, 100, 120)],
            "science": [(15, 25, 30), (40, 60, 80), (80, 120, 160)],
            "entertainment": [(30, 20, 35), (70, 50, 80), (140, 100, 160)],
            "social": [(25, 30, 20), (60, 80, 50), (120, 160, 100)]
        }
        
        colors = color_schemes.get(topic_category, [(10, 15, 25), (20, 40, 60), (40, 80, 120)])
        
        image = Image.new('RGB', (1080, 1350), colors[0])
        draw = ImageDraw.Draw(image)
        
        if slide_type in ["hook", "cta"]:
            for x in range(1080):
                progress = x / 1080
                color_value = [int(colors[0][i] + (colors[1][i] - colors[0][i]) * progress) for i in range(3)]
                draw.line([(x, 0), (x, 1350)], fill=tuple(color_value))
            
            for i in range(3):
                y_pos = 300 + (i * 400)
                x_pos = 800 + (i * 20)
                size = 100 - (i * 20)
                draw.ellipse([x_pos, y_pos, x_pos + size, y_pos + size], 
                            outline=(*colors[2], 100), width=2)
        else:
            for y in range(1350):
                progress = y / 1350
                color_value = [int(colors[1][i] - (colors[1][i] - colors[0][i]) * progress) for i in range(3)]
                draw.line([(0, y), (1080, y)], fill=tuple(color_value))
            
            for i in range(3):
                x_pos = 200 + (i * 300)
                y_pos = 100 + (i * 50)
                size = 80 - (i * 15)
                draw.ellipse([x_pos, y_pos, x_pos + size, y_pos + size], 
                            outline=(*colors[2], 100), width=2)
        
        timestamp = int(time.time())
        image_path = f"instagram_carousels/bg_{timestamp}.png"
        image.save(image_path)
        return image_path

    def create_enhanced_gradient_overlay(self, width: int, height: int, slide_type: str) -> Image.Image:
        """Create an enhanced gradient overlay for better text visibility."""
        gradient = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(gradient)
            
        if slide_type in ["hook", "cta"]:
            for x in range(width):
                progress = x / width
                alpha = int(255 - (195 * progress))
                for y in range(height):
                    draw.point((x, y), fill=(0, 0, 0, alpha))
        else:
            for y in range(height):
                progress = 1 - (y / height)
                alpha = int(255 - (195 * progress))
                for x in range(width):
                    draw.point((x, y), fill=(0, 0, 0, alpha))
        return gradient

    def get_font(self, base_size: int):
        """Get font for text."""
        font_options = [
            ("fonts/Montserrat-Black.ttf", base_size),
            ("fonts/Roboto-Black.ttf", base_size),
            ("fonts/Poppins-Black.ttf", base_size),
            ("Arial Black.ttf", base_size),
            ("Impact.ttf", base_size),
        ]
        
        for font_path, size in font_options:
            try:
                font = ImageFont.truetype(font_path, size)
                return font, size
            except (IOError, OSError):
                continue
        
        # Fallback to default font
        try:
            font = ImageFont.load_default()
            return font, base_size
        except:
            return None, base_size

    def add_text_to_image(self, image_path: str, title: str, content: str, slide_number: int, slide_type: str, topic_category: str = "general") -> str:
        """Add title and content text overlay with enhanced visual design."""
        background = Image.open(image_path).convert('RGBA')
        img_width, img_height = background.size
        
        overlay = self.create_enhanced_gradient_overlay(img_width, img_height, slide_type)
        background = Image.alpha_composite(background, overlay)
        draw = ImageDraw.Draw(background)
        
        # Category-specific accent colors
        accent_colors = {
            "technology": (0, 162, 255, 255),
            "health": (34, 197, 94, 255),
            "business": (234, 179, 8, 255),
            "education": (249, 115, 22, 255),
            "lifestyle": (236, 72, 153, 255),
            "science": (6, 182, 212, 255),
            "entertainment": (168, 85, 247, 255),
            "social": (34, 197, 94, 255)
        }
        
        title_color = (255, 255, 255, 255)
        content_color = (235, 235, 235, 255)
        accent_color = accent_colors.get(topic_category, (0, 162, 255, 255))
        
        if slide_type in ["hook", "cta"]:
            text_area_width = int(img_width * 0.55)
            padding_left = int(img_width * 0.08)
            padding_top = int(img_height * 0.25)
            text_align = "left"
            title_font_size = int(img_height * 0.062)
            content_font_size = int(img_height * 0.022)
        else:
            text_area_width = int(img_width * 0.85)
            padding_left = int(img_width * 0.075)
            padding_top = int(img_height * 0.45)
            text_align = "center"
            title_font_size = int(img_height * 0.055)
            content_font_size = int(img_height * 0.024)
        
        title_font, actual_title_size = self.get_font(title_font_size)
        content_font, actual_content_size = self.get_font(content_font_size)
        account_font, actual_account_size = self.get_font(int(img_height * 0.022))
        
        available_text_width = text_area_width
        
        # Word wrap title
        title_words = title.split()
        title_lines = []
        current_line = []
        
        for word in title_words:
            test_line = ' '.join(current_line + [word])
            if title_font:
                line_width = draw.textlength(test_line, font=title_font)
            else:
                line_width = len(test_line) * (actual_title_size * 0.6)  # Rough estimation
            
            if line_width <= available_text_width:
                current_line.append(word)
            else:
                if current_line:
                    title_lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    title_lines.append(word)
                    current_line = []
        
        if current_line:
            title_lines.append(' '.join(current_line))
        
        # Word wrap content
        content_words = content.split()
        content_lines = []
        current_line = []
        
        for word in content_words:
            test_line = ' '.join(current_line + [word])
            if content_font:
                line_width = draw.textlength(test_line, font=content_font)
            else:
                line_width = len(test_line) * (actual_content_size * 0.6)  # Rough estimation
            
            if line_width <= available_text_width:
                current_line.append(word)
            else:
                if current_line:
                    content_lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    content_lines.append(word)
                    current_line = []
        
        if current_line:
            content_lines.append(' '.join(current_line))
        
        title_line_height = int(actual_title_size * 1.3)
        content_line_height = int(actual_content_size * 1.4)
        spacing_between_sections = int(img_height * 0.04)
        
        current_y = padding_top
        
        # Draw title lines with shadow
        for line in title_lines:
            if text_align == "center":
                if title_font:
                    line_width = draw.textlength(line, font=title_font)
                else:
                    line_width = len(line) * (actual_title_size * 0.6)
                text_x = (img_width - line_width) // 2
            else:
                text_x = padding_left
            
            # Shadow effects
            shadow_offsets = [(4, 4), (2, 2)]
            shadow_alphas = [120, 180]
            
            for (sx, sy), alpha in zip(shadow_offsets, shadow_alphas):
                if title_font:
                    draw.text((text_x + sx, current_y + sy), line, 
                             font=title_font, fill=(0, 0, 0, alpha))
            
            if title_font:
                draw.text((text_x, current_y), line, font=title_font, fill=title_color)
            current_y += title_line_height
        
        current_y += spacing_between_sections
        
        # Draw content lines with shadow
        for line in content_lines:
            if text_align == "center":
                if content_font:
                    line_width = draw.textlength(line, font=content_font)
                else:
                    line_width = len(line) * (actual_content_size * 0.6)
                text_x = (img_width - line_width) // 2
            else:
                text_x = padding_left
            
            if content_font:
                draw.text((text_x + 2, current_y + 2), line, 
                         font=content_font, fill=(0, 0, 0, 150))
                draw.text((text_x, current_y), line, font=content_font, fill=content_color)
            current_y += content_line_height
        
        # Add slide number indicator
        indicator_size = int(img_height * 0.015)
        indicator_y = img_height - int(img_height * 0.08)
        total_width = (5 * indicator_size) + (4 * indicator_size // 2)
        start_x = (img_width - total_width) // 2
        
        for i in range(5):
            x = start_x + (i * (indicator_size + indicator_size // 2))
            if i + 1 == slide_number:
                draw.ellipse([x, indicator_y, x + indicator_size, indicator_y + indicator_size], 
                           fill=accent_color)
            else:
                draw.ellipse([x, indicator_y, x + indicator_size, indicator_y + indicator_size], 
                           outline=(255, 255, 255, 180), width=2)
        
        # Add account name/branding
        # account_text = "@your_account"
        # if account_font:
        #     account_width = draw.textlength(account_text, font=account_font)
        # else:
        #     account_width = len(account_text) * (actual_account_size * 0.6)
        
        # account_x = img_width - account_width - int(img_width * 0.05)
        # account_y = int(img_height * 0.05)
        
        # if account_font:
        #     draw.text((account_x + 1, account_y + 1), account_text, 
        #              font=account_font, fill=(0, 0, 0, 120))
        #     draw.text((account_x, account_y), account_text, 
        #              font=account_font, fill=(200, 200, 200, 200))
        
        # Save the final image
        final_image = background.convert('RGB')
        timestamp = int(time.time())
        final_path = f"instagram_carousels/slide_{slide_number}_{timestamp}.png"
        final_image.save(final_path, optimize=True, quality=95)
        
        return final_path

    def create_complete_carousel(self, topic_data: Dict, preferences: Dict, topic_category: str = "general") -> Dict:
        """Create a complete Instagram carousel with all slides."""
        try:
            # Generate complete carousel content
            carousel_content = self.generate_complete_carousel_content(topic_data, preferences, topic_category)
            
            carousel_files = []
            
            # Create each slide
            slides_data = [
                ("hook", carousel_content.hook_title, carousel_content.hook_content, carousel_content.hook_image_prompt),
                ("slide2", carousel_content.slide2_title, carousel_content.slide2_content, carousel_content.slide2_image_prompt),
                ("slide3", carousel_content.slide3_title, carousel_content.slide3_content, carousel_content.slide3_image_prompt),
                ("slide4", carousel_content.slide4_title, carousel_content.slide4_content, carousel_content.slide4_image_prompt),
                ("cta", carousel_content.cta_title, carousel_content.cta_content, carousel_content.cta_image_prompt)
            ]
            
            for i, (slide_type, title, content, image_prompt) in enumerate(slides_data, 1):
                logger.info(f"Creating slide {i}: {slide_type}")
                
                # Generate background image
                bg_image_path = self.generate_image(image_prompt, slide_type, topic_category)
                
                # Add text overlay
                final_slide_path = self.add_text_to_image(
                    bg_image_path, title, content, i, slide_type, topic_category
                )
                
                carousel_files.append({
                    'slide_number': i,
                    'slide_type': slide_type,
                    'title': title,
                    'content': content,
                    'file_path': final_slide_path,
                    'filename': os.path.basename(final_slide_path)
                })
                # Clean up background image
                try:
                    os.remove(bg_image_path)
                except:
                    pass
            
            # --- Cleanup: Keep only the latest 5 slide images (png/jpg), delete the rest ---
            # Also delete all bg_*.png and bg_*.jpg, and any .zip files older than 1 hour
            # Handle both .png and .jpg for slides and backgrounds
            slide_images = [f for f in os.listdir("instagram_carousels") if (f.startswith("slide_") and (f.endswith(".png") or f.endswith(".jpg")))]
            slide_images_full = [(f, os.path.getctime(os.path.join("instagram_carousels", f))) for f in slide_images]
            slide_images_full.sort(key=lambda x: x[1])
            if len(slide_images_full) > 5:
                for f, _ in slide_images_full[:-5]:
                    try:
                        os.remove(os.path.join("instagram_carousels", f))
                    except Exception as e:
                        logger.warning(f"Could not delete old image {f}: {e}")

            # Delete all bg_*.png and bg_*.jpg files
            for f in os.listdir("instagram_carousels"):
                if (f.startswith("bg_") and (f.endswith(".png") or f.endswith(".jpg"))):
                    try:
                        os.remove(os.path.join("instagram_carousels", f))
                    except Exception as e:
                        logger.warning(f"Could not delete background image {f}: {e}")

            # Delete .zip files older than 1 hour
            now = time.time()
            for f in os.listdir("instagram_carousels"):
                if f.endswith(".zip"):
                    try:
                        file_path = os.path.join("instagram_carousels", f)
                        if now - os.path.getctime(file_path) > 3600:
                            os.remove(file_path)
                    except Exception as e:
                        logger.warning(f"Could not delete old zip {f}: {e}")
            
            # Create ZIP file with all slides
            zip_filename = f"carousel_{int(time.time())}.zip"
            zip_path = f"instagram_carousels/{zip_filename}"
            
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for slide_data in carousel_files:
                    zipf.write(slide_data['file_path'], slide_data['filename'])
            
            return {
                'success': True,
                'carousel_files': carousel_files,
                'zip_file': zip_path,
                'zip_filename': zip_filename,
                'overall_theme': carousel_content.overall_theme,
                'narrative_flow': carousel_content.narrative_flow,
                'topic_category': carousel_content.topic_category,
                'message': f'Successfully created {len(carousel_files)} carousel slides'
            }
        except Exception as e:
            logger.error(f"Error creating carousel: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to create carousel'
            }

# Initialize the generator
try:
    groq_api_key =  "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"
    google_api_key =  "AIzaSyDlqOFJ5cms_lEdnf7guc_BR2BIA7PNJRU"
    
    if not groq_api_key or not google_api_key:
        logger.error("Missing required API keys")
        carousel_generator = None
    else:
        carousel_generator = CarouselGenerator(groq_api_key, google_api_key)
        logger.info("Carousel generator initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize carousel generator: {e}")
    carousel_generator = None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/generate-topics', methods=['POST'])
def generate_topics():
    """Generate Instagram carousel topics based on preferences."""
    if not carousel_generator:
        return jsonify({'error': 'Carousel generator not initialized'}), 500
    
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['niche', 'specific_topic', 'goal', 'age_group', 'skill_level', 'tone', 'cta_style', 'num_topics'
]
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                'error': 'Missing required fields',
                'missing_fields': missing_fields
            }), 400
        
        # Set default values and validate
        preferences = {
            'niche': data.get('niche', '').strip(),
            'specific_topic': data.get('specific_topic', '').strip(),
            'goal': data.get('goal', '').strip(),
            'age_group': data.get('age_group', '').strip(),
            'skill_level': data.get('skill_level', '').strip(),
            'tone': data.get('tone', '').strip(),
            'focus_areas': data.get('focus_areas', '').strip(),
            'cta_style': data.get('cta_style', '').strip(),
            'num_topics': min(int(data.get('num_topics', 1)), 10)  # Limit to 10 topics max
        }
        
        # Generate topics
        topics = carousel_generator.generate_carousel_topics(preferences)
        
        if not topics:
            return jsonify({
                'error': 'Failed to generate topics',
                'message': 'Please try again with different preferences'
            }), 500
        
        return jsonify({
            'success': True,
            'topics': topics,
            'preferences': preferences,
            'total_topics': len(topics)
        })
        
    except Exception as e:
        logger.error(f"Error in generate_topics: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/create-carousel', methods=['POST'])
def create_carousel():
    """Create a complete Instagram carousel."""
    if not carousel_generator:
        return jsonify({'error': 'Carousel generator not initialized'}), 500
    
    try:
        data = request.get_json()
        
        # Validate required fields
        if 'topic_data' not in data or 'preferences' not in data:
            return jsonify({
                'error': 'Missing required fields: topic_data and preferences'
            }), 400
        
        topic_data = data['topic_data']
        preferences = data['preferences']
        topic_category = data.get('topic_category', 'general')
        
        # Validate topic_data structure
        required_topic_fields = ['hook', 'slide2', 'slide3', 'slide4', 'cta']
        missing_topic_fields = [field for field in required_topic_fields if field not in topic_data]
        
        if missing_topic_fields:
            return jsonify({
                'error': 'Invalid topic_data structure',
                'missing_fields': missing_topic_fields
            }), 400
        
        # Create the carousel
        result = carousel_generator.create_complete_carousel(topic_data, preferences, topic_category)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 500
            
    except Exception as e:
        logger.error(f"Error in create_carousel: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/download-carousel/<filename>', methods=['GET'])
def download_carousel(filename):
    """Download carousel ZIP file."""
    try:
        file_path = f"instagram_carousels/{filename}"
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename,
            mimetype='application/zip'
        )
        
    except Exception as e:
        logger.error(f"Error downloading carousel: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/categorize-topic', methods=['POST'])
def categorize_topic():
    """Categorize a topic to determine the best visual style."""
    try:
        data = request.get_json()
        
        if 'topic' not in data:
            return jsonify({'error': 'Missing topic field'}), 400
        
        topic = data['topic'].lower()
        
        # Simple categorization logic
        categories = {
            'technology': ['tech', 'software', 'programming', 'ai', 'digital', 'computer', 'app', 'coding', 'web', 'data'],
            'health': ['health', 'fitness', 'wellness', 'nutrition', 'medical', 'exercise', 'mental', 'diet', 'healthcare'],
            'business': ['business', 'marketing', 'finance', 'sales', 'entrepreneur', 'leadership', 'management', 'startup', 'money'],
            'education': ['education', 'learning', 'study', 'academic', 'school', 'university', 'knowledge', 'teaching', 'course'],
            'lifestyle': ['lifestyle', 'fashion', 'travel', 'food', 'home', 'beauty', 'relationships', 'personal', 'hobby'],
            'science': ['science', 'research', 'physics', 'chemistry', 'biology', 'environment', 'space', 'nature'],
            'entertainment': ['entertainment', 'movies', 'music', 'gaming', 'sports', 'celebrity', 'tv', 'art', 'culture'],
            'social': ['social', 'community', 'society', 'politics', 'news', 'current', 'events', 'world', 'global']
        }
        
        for category, keywords in categories.items():
            if any(keyword in topic for keyword in keywords):
                return jsonify({
                    'success': True,
                    'category': category,
                    'topic': data['topic']
                })
        
        # Default to general if no specific category found
        return jsonify({
            'success': True,
            'category': 'general',
            'topic': data['topic']
        })
        
    except Exception as e:
        logger.error(f"Error categorizing topic: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/cleanup', methods=['POST'])
def cleanup_files():
    """Clean up old generated files."""
    try:
        cleanup_count = 0
        current_time = time.time()
        
        # Clean up files older than 1 hour
        for filename in os.listdir("instagram_carousels"):
            file_path = os.path.join("instagram_carousels", filename)
            if os.path.isfile(file_path):
                file_age = current_time - os.path.getctime(file_path)
                if file_age > 3600:  # 1 hour
                    os.remove(file_path)
                    cleanup_count += 1
        
        return jsonify({
            'success': True,
            'files_cleaned': cleanup_count,
            'message': f'Cleaned up {cleanup_count} old files'
        })
        
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/preview-carousel/<zip_filename>', methods=['GET'])
def preview_carousel(zip_filename):
    """Preview all images generated for a carousel before/after zipping."""
    try:
        zip_path = f"instagram_carousels/{zip_filename}"
        if not os.path.exists(zip_path):
            return jsonify({'error': 'File not found'}), 404

        # Get the timestamp from the zip filename (format: carousel_<timestamp>.zip)
        try:
            ts_part = zip_filename.split('_')[-1].replace('.zip', '')
            zip_ts = int(ts_part)
        except Exception:
            zip_ts = None

        # List all slide images in the directory that are close in time to the zip file
        image_files = []
        for fname in os.listdir("instagram_carousels"):
            if fname.startswith("slide_") and fname.endswith(".png"):
                try:
                    # slide_{slide_number}_{timestamp}.png
                    parts = fname.split('_')
                    img_ts = int(parts[-1].replace('.png', ''))
                    # If timestamp is within 2 minutes of zip creation, consider it part of this carousel
                    if zip_ts and abs(img_ts - zip_ts) < 120:
                        image_files.append(fname)
                except Exception:
                    continue

        # Sort by slide number
        def slide_sort_key(f):
            try:
                return int(f.split('_')[1])
            except Exception:
                return 0
        image_files.sort(key=slide_sort_key)

        image_urls = [
            request.url_root.rstrip('/') + f'/carousel-image-file/{fname}'
            for fname in image_files
        ]
        # Keep only the latest 5 images, delete the rest
        print(f"Found {len(image_files)} images for preview: {image_files}")
        logger.info(f"Found {len(image_files)} images for preview: {image_files}")
        if len(image_files) > 5:
            to_delete = image_files[:-5]
            for fname in to_delete:
                try:
                    os.remove(os.path.join("instagram_carousels", fname))
                except Exception as e:
                    logger.warning(f"Could not delete old image {fname}: {e}")
            image_files = image_files[-5:]
        return jsonify({
            'success': True,
            'images': image_urls,
            'total_images': len(image_urls),
            'zip_filename': zip_filename
        })
    except Exception as e:
        logger.error(f"Error in preview_carousel: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/carousel-image-file/<image_filename>', methods=['GET'])
def serve_carousel_image_file(image_filename):
    """Serve a single generated image file for preview."""
    try:
        file_path = f"instagram_carousels/{image_filename}"
        if not os.path.exists(file_path):
            return jsonify({'error': 'Image file not found'}), 404
        return send_file(
            file_path,
            mimetype='image/png',
            as_attachment=False,
            download_name=image_filename
        )
    except Exception as e:
        logger.error(f"Error serving carousel image file: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
