import os
import time
import logging
import requests
import json
import base64
import hashlib
import numpy as np
from io import BytesIO
from urllib.parse import urljoin, urlparse
from flask import Flask, request, jsonify, render_template_string, send_from_directory
from bs4 import BeautifulSoup
from datetime import datetime
from typing import Dict, Optional, List
from PIL import Image, ImageOps

# Google Gemini AI client импорт
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logging.warning("Google Gemini client not installed. Install with: pip install google-genai")

app = Flask(__name__, static_folder='.', static_url_path='/static')
logging.basicConfig(level=logging.INFO)

# —— Config —— #
ROOT_URL             = os.getenv("ROOT_URL", "https://kako.mn/")
DELAY_SEC            = float(os.getenv("DELAY_SEC", "0.5"))
ALLOWED_NETLOC       = urlparse(ROOT_URL).netloc
MAX_CRAWL_PAGES      = int(os.getenv("MAX_CRAWL_PAGES", "500"))
CHATWOOT_API_KEY     = os.getenv("CHATWOOT_API_KEY")
ACCOUNT_ID           = os.getenv("ACCOUNT_ID")
CHATWOOT_BASE_URL    = os.getenv("CHATWOOT_BASE_URL", "https://app.chatwoot.com/")
GEMINI_API_KEY       = os.getenv("GEMINI_API_KEY")
AUTO_CRAWL_ON_START  = os.getenv("AUTO_CRAWL_ON_START", "true").lower() == "true"

# Image storage config
IMAGES_DIR = "crawled_images"
os.makedirs(IMAGES_DIR, exist_ok=True)

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY) if (GEMINI_API_KEY and GEMINI_AVAILABLE) else None

# —— Memory Storage —— #
conversation_memory = {}
crawled_data = []
crawl_status = {"status": "not_started", "message": "Crawling has not started yet"}
crawled_images = []  # Store image metadata and features

# —— Crawl & Scrape —— #
def crawl_and_scrape(start_url: str):
    visited = set()
    to_visit = {start_url}
    results = []

    while to_visit and len(visited) < MAX_CRAWL_PAGES:
        url = to_visit.pop()
        if url in visited:
            continue
        visited.add(url)

        try:
            logging.info(f"[Crawling] {url}")
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
        except Exception as e:
            logging.warning(f"Failed to fetch {url}: {e}")
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        title = soup.title.string.strip() if soup.title and soup.title.string else url
        body, images = extract_content(soup, url)

        results.append({
            "url": url,
            "title": title,
            "body": body,
            "images": images
        })

        for a in soup.find_all("a", href=True):
            href = a["href"]
            if is_internal_link(href):
                full = normalize_url(url, href)
                if full.startswith(ROOT_URL) and full not in visited:
                    to_visit.add(full)

        time.sleep(DELAY_SEC)

    return results

# —— Startup Functions —— #
def auto_crawl_on_startup():
    """Automatically crawl the site on startup"""
    global crawled_data, crawl_status
    
    if not AUTO_CRAWL_ON_START:
        crawl_status = {"status": "disabled", "message": "Auto-crawl is disabled"}
        logging.info("Auto-crawl is disabled")
        return
    
    try:
        logging.info(f"🚀 Starting automatic crawl of {ROOT_URL}")
        crawl_status = {"status": "running", "message": f"Crawling {ROOT_URL}..."}
        
        crawled_data = crawl_and_scrape(ROOT_URL)
        
        if crawled_data:
            crawl_status = {
                "status": "completed", 
                "message": f"Successfully crawled {len(crawled_data)} pages",
                "pages_count": len(crawled_data),
                "timestamp": datetime.now().isoformat()
            }
            logging.info(f"✅ Auto-crawl completed: {len(crawled_data)} pages")
        else:
            crawl_status = {"status": "failed", "message": "No pages were crawled"}
            logging.warning("❌ Auto-crawl failed: No pages found")
            
    except Exception as e:
        crawl_status = {"status": "error", "message": f"Crawl error: {str(e)}"}
        logging.error(f"❌ Auto-crawl error: {e}")

# Start auto-crawl in background when app starts
import threading
if AUTO_CRAWL_ON_START:
    threading.Thread(target=auto_crawl_on_startup, daemon=True).start()

# —— Image Processing Functions —— #
def is_product_image(img_url: str, alt_text: str, page_context: str = "") -> bool:
    """Determine if an image is likely a product image"""
    img_url_lower = img_url.lower()
    alt_lower = alt_text.lower()
    page_lower = page_context.lower()
    
    # Skip common non-product images
    skip_patterns = [
        'logo', 'banner', 'header', 'footer', 'menu', 'icon', 'button',
        'social', 'facebook', 'instagram', 'twitter', 'youtube',
        'avatar', 'profile', 'user', 'author', 'team',
        'background', 'bg-', 'hero', 'slider', 'carousel',
        'arrow', 'chevron', 'close', 'search', 'cart-icon',
        'payment', 'visa', 'mastercard', 'paypal',
        'placeholder', 'loading', 'spinner', 'no-image',
        'about-us', 'contact', 'location', 'map'
    ]
    
    # Check if URL or alt contains skip patterns
    for pattern in skip_patterns:
        if pattern in img_url_lower or pattern in alt_lower:
            return False
    
    # Product image indicators
    product_indicators = [
        # URL patterns
        'product', 'item', 'goods', 'merchandise', 'catalog',
        'shop', 'store', 'buy', 'sale', 'price',
        'thumbnail', 'thumb', 'gallery', 'image',
        
        # Mongolian product terms
        'бүтээгдэхүүн', 'барааг', 'зураг', 'зүйл',
        'хувцас', 'гутал', 'цүнх', 'эмэгтэй', 'эрэгтэй',
        'зуны', 'өвлийн', 'хаврын', 'намрын',
        
        # Common product categories
        'clothing', 'shoes', 'bag', 'watch', 'jewelry',
        'electronics', 'phone', 'laptop', 'headphone',
        'book', 'toy', 'game', 'sport', 'fitness',
        'beauty', 'cosmetic', 'perfume', 'makeup',
        'home', 'kitchen', 'furniture', 'decor'
    ]
    
    # Check for product indicators
    product_score = 0
    for indicator in product_indicators:
        if indicator in img_url_lower:
            product_score += 2
        if indicator in alt_lower:
            product_score += 3
        if indicator in page_lower:
            product_score += 1
    
    # Additional checks
    # Images in specific folders are likely products
    if any(folder in img_url_lower for folder in ['/products/', '/items/', '/catalog/', '/shop/', '/store/']):
        product_score += 5
    
    # Images with product-like alt text
    if any(term in alt_lower for term in ['төгрөг', '₮', 'price', 'sale', 'buy', 'order']):
        product_score += 3
    
    # Images with dimensions in filename (often product images)
    import re
    if re.search(r'\d+x\d+', img_url_lower):
        product_score += 2
    
    # File format preferences (product images often in these formats)
    if any(ext in img_url_lower for ext in ['.jpg', '.jpeg', '.png', '.webp']):
        product_score += 1
    
    # Return True if score is high enough
    return product_score >= 3

def download_and_save_image(img_url: str, base_url: str) -> Optional[Dict]:
    """Download image and save it locally with metadata"""
    try:
        # Create full URL
        full_url = urljoin(base_url, img_url)
        
        # Create hash for filename
        url_hash = hashlib.md5(full_url.encode()).hexdigest()
        
        # Download image
        response = requests.get(full_url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        response.raise_for_status()
        
        # Check if it's actually an image
        content_type = response.headers.get('content-type', '')
        if not content_type.startswith('image/'):
            return None
            
        # Save original image
        file_extension = '.jpg'  # Default
        if 'png' in content_type:
            file_extension = '.png'
        elif 'webp' in content_type:
            file_extension = '.webp'
        elif 'gif' in content_type:
            file_extension = '.gif'
            
        filename = f"{url_hash}{file_extension}"
        filepath = os.path.join(IMAGES_DIR, filename)
        
        with open(filepath, 'wb') as f:
            f.write(response.content)
            
        # Load and process image
        try:
            with Image.open(filepath) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Get basic features
                width, height = img.size
                
                # Create thumbnail for faster processing
                thumb_size = (128, 128)
                thumb = img.copy()
                thumb.thumbnail(thumb_size, Image.Resampling.LANCZOS)
                
                # Calculate basic color histogram (simple feature)
                hist_r = thumb.histogram()[0:256]    # Red channel
                hist_g = thumb.histogram()[256:512]  # Green channel
                hist_b = thumb.histogram()[512:768]  # Blue channel
                
                # Normalize histograms
                total_pixels = thumb.size[0] * thumb.size[1]
                hist_r = [x/total_pixels for x in hist_r]
                hist_g = [x/total_pixels for x in hist_g]
                hist_b = [x/total_pixels for x in hist_b]
                
                return {
                    'url': full_url,
                    'filename': filename,
                    'filepath': filepath,
                    'width': width,
                    'height': height,
                    'file_size': len(response.content),
                    'content_type': content_type,
                    'hist_r': hist_r,
                    'hist_g': hist_g,
                    'hist_b': hist_b,
                    'thumb_size': thumb_size
                }
                
        except Exception as img_error:
            logging.warning(f"Could not process image {full_url}: {img_error}")
            # Clean up the file if processing failed
            if os.path.exists(filepath):
                os.remove(filepath)
            return None
            
    except Exception as e:
        logging.warning(f"Could not download image {img_url}: {e}")
        return None

def calculate_image_similarity(img1_features: Dict, img2_features: Dict) -> float:
    """Calculate similarity between two images using histogram comparison"""
    try:
        # Compare histograms using Chi-squared distance
        def chi_squared_distance(h1, h2):
            distance = 0
            for i in range(len(h1)):
                if h1[i] + h2[i] > 0:
                    distance += ((h1[i] - h2[i]) ** 2) / (h1[i] + h2[i])
            return distance
        
        # Calculate distances for each color channel
        dist_r = chi_squared_distance(img1_features['hist_r'], img2_features['hist_r'])
        dist_g = chi_squared_distance(img1_features['hist_g'], img2_features['hist_g'])
        dist_b = chi_squared_distance(img1_features['hist_b'], img2_features['hist_b'])
        
        # Average distance
        avg_distance = (dist_r + dist_g + dist_b) / 3
        
        # Convert to similarity (0-1, where 1 is identical)
        similarity = 1 / (1 + avg_distance)
        
        return similarity
        
    except Exception as e:
        logging.error(f"Error calculating image similarity: {e}")
        return 0.0

def process_user_image_features(image_bytes: bytes) -> Optional[Dict]:
    """Process user uploaded image and extract features"""
    try:
        # Load image from bytes
        img = Image.open(BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Get basic info
        width, height = img.size
        
        # Create thumbnail for processing
        thumb_size = (128, 128)
        thumb = img.copy()
        thumb.thumbnail(thumb_size, Image.Resampling.LANCZOS)
        
        # Calculate color histograms
        hist_r = thumb.histogram()[0:256]
        hist_g = thumb.histogram()[256:512]
        hist_b = thumb.histogram()[512:768]
        
        # Normalize
        total_pixels = thumb.size[0] * thumb.size[1]
        hist_r = [x/total_pixels for x in hist_r]
        hist_g = [x/total_pixels for x in hist_g]
        hist_b = [x/total_pixels for x in hist_b]
        
        return {
            'width': width,
            'height': height,
            'hist_r': hist_r,
            'hist_g': hist_g,
            'hist_b': hist_b,
            'thumb_size': thumb_size
        }
        
    except Exception as e:
        logging.error(f"Error processing user image: {e}")
        return None

def find_similar_crawled_images(user_image_features: Dict, similarity_threshold: float = 0.3) -> List[Dict]:
    """Find similar images from crawled data"""
    similar_images = []
    
    for img_data in crawled_images:
        similarity = calculate_image_similarity(user_image_features, img_data)
        
        if similarity >= similarity_threshold:
            similar_images.append({
                'url': img_data['url'],
                'similarity': similarity,
                'filename': img_data['filename'],
                'page_url': img_data.get('page_url', ''),
                'alt': img_data.get('alt', ''),
                'page_title': img_data.get('page_title', '')
            })
    
    # Sort by similarity (highest first) and return top 3 matches
    similar_images.sort(key=lambda x: x['similarity'], reverse=True)
    
    return similar_images[:3]  # Return top 3 matches

# —— Content Extraction —— #
def extract_content(soup: BeautifulSoup, base_url: str):
    main = soup.find("main") or soup
    texts = []
    images = []

    for tag in main.find_all(["h1", "h2", "h3", "h4", "p", "li", "code"]):
        text = tag.get_text(strip=True)
        if text:
            texts.append(text)

    # Get page title and text content for context
    page_title = soup.title.string.strip() if soup.title and soup.title.string else base_url
    page_text = " ".join(texts[:5])  # Use first 5 text elements for context

    for img in main.find_all("img"):
        src = img.get("src")
        alt = img.get("alt", "").strip()
        if src:
            full_img_url = urljoin(base_url, src)
            entry = f"[Image] {alt} — {full_img_url}" if alt else f"[Image] {full_img_url}"
            texts.append(entry)
            images.append({"url": full_img_url, "alt": alt})

            # Only download and process product images
            if is_product_image(full_img_url, alt, page_text):
                logging.info(f"Downloading product image: {full_img_url}")
                image_data = download_and_save_image(src, base_url)
                if image_data:
                    # Add page context to image data
                    image_data['page_url'] = base_url
                    image_data['page_title'] = page_title
                    image_data['alt'] = alt
                    
                    # Add to global crawled images list
                    crawled_images.append(image_data)
                    logging.info(f"Successfully processed product image: {image_data['filename']}")
                else:
                    logging.warning(f"Failed to process product image: {full_img_url}")
            else:
                logging.info(f"Skipping non-product image: {full_img_url}")

    return "\n\n".join(texts), images

def is_internal_link(href: str) -> bool:
    if not href:
        return False
    parsed = urlparse(href)
    return not parsed.netloc or parsed.netloc == ALLOWED_NETLOC

def normalize_url(base: str, link: str) -> str:
    return urljoin(base, link.split("#")[0])

def scrape_single(url: str):
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    title = soup.title.string.strip() if soup.title and soup.title.string else url
    body, images = extract_content(soup, url)
    return {"url": url, "title": title, "body": body, "images": images}


# —— AI Assistant Functions —— #
def get_ai_response(user_message: str, conversation_id: int, context_data: list = None, image_data: dict = None):
    """Enhanced AI response with Google Gemini for text and image understanding"""
    
    if not client:
        return "🔑 Google Gemini API түлхүүр тохируулагдаагүй байна. Системийн админтай холбогдоно уу."
    
    # Handle empty message when no image is provided
    if not user_message or not user_message.strip():
        if image_data:
            user_message = "Энэ зураг дээр юу байгааг тайлбарлаад өгнө үү?"
        else:
            return "📝 Таны мессеж хоосон байна. Асуулт эсвэл хайж байгаа зүйлээ бичээд илгээнэ үү. Би танд туслахад бэлэн байна! 😊"
    
    # Get conversation history
    history = conversation_memory.get(conversation_id, [])
    
    # Build context from crawled data if available
    context = ""
    similar_images_context = ""
    
    # Process image similarity if user sent an image
    if image_data and crawled_images:
        user_image_features = process_user_image_features(image_data.get('data'))
        if user_image_features:
            similar_images = find_similar_crawled_images(user_image_features, similarity_threshold=0.2)
            if similar_images:
                similar_images_info = []
                for i, sim_img in enumerate(similar_images, 1):
                    similarity_percent = round(sim_img['similarity'] * 100, 1)
                    similar_images_info.append(
                        f"Ижил төстэй зураг {i} ({similarity_percent}% ижил):\n"
                        f"• Хуудас: {sim_img['page_title']}\n"
                        f"• URL: {sim_img['page_url']}\n"
                        f"• Зургийн тайлбар: {sim_img['alt']}\n"
                        f"• Зургийн холбоос: {sim_img['url']}\n"
                        f"{'='*50}"
                    )
                similar_images_context = "\n\n".join(similar_images_info)
                logging.info(f"Found {len(similar_images)} similar images for user image")
    
    if crawled_data and not image_data:  # Only search context for text queries
        # Search for relevant content with more results
        search_results = search_in_crawled_data(user_message, max_results=5)
        if search_results:
            relevant_pages = []
            for i, result in enumerate(search_results, 1):
                relevant_pages.append(
                    f"Хуудас {i}: {result['title']}\n"
                    f"URL: {result['url']}\n"
                    f"Холбогдох агуулга: {result['snippet']}\n"
                    f"{'='*50}"
                )
            context = "\n\n".join(relevant_pages)
    
    # Build system message with context
    system_content = """Та онлайн дэлгүүрийн AI туслах бот юм. Хэрэглэгчдэд бүтээгдэхүүний мэдээлэл, үнэ, дэлгэрэнгүй мэдээлэл хайж олоход тусалдаг.
    Хэрэглэгчтэй монгол хэлээр найрсаг, тусламжтай ярилцаарай. Та өөрийн мэдэх мэдээллээр дамжуулан бүхий л асуултад хариулах чадвартай.
    
    ЗУРАГ ШИНЖИЛГЭЭНИЙ ТУХАЙ:
    Хэрэв хэрэглэгч зураг илгээвэл, зургийг сайтар үзээд дараах зүйлсийг хийнэ үү:
    • Зураг дээрх гол объект, зүйлсийг тодорхойлно
    • Зураг дээрх бүтээгдэхүүн байвал, түүний нэр, загвар, онцлогийг хэлнэ
    • Өнгө, хэлбэр, хэмжээ зэрэг дэлгэрэнгүй мэдээллийг өгнө
    • Хэрэв бүтээгдэхүүн таньж болвол, түүний үнэ болон худалдан авах боломжийн талаар мэдээлэл өгнө
    • Зургийн чанар муу эсвэл тодорхойгүй байвал, илүү тод зураг оруулахыг санал болгоно
    
    ИЖИЛ ТӨСТЭЙ ЗУРГИЙН МЭДЭЭЛЭЛ:
    Хэрэв хэрэглэгчийн илгээсэн зурагтай ижил төстэй зургууд олдсон бол:
    • Эдгээр хамгийн ижил төстэй 3 зургийн мэдээллийг ашиглан илүү нарийвчлалтай хариулт өгнө үү
    • Тухайн бүтээгдэхүүний хуудасны мэдээллийг дурдаж, холбоосыг өгнө үү
    • Ижил төстэй зургуудын ялгааг тайлбарлаж өгнө үү
    • Хамгийн тохирох бүтээгдэхүүнийг санал болгоно уу
    
    ЭНГИЙН МЭНДЧИЛГЭЭНИЙ ТУХАЙ:
    Хэрэв хэрэглэгч энгийн мэндчилгээ хийж байвал (жишээ: "сайн байна уу", "сайн уу", "мэнд", "hello", "hi", "сайн уу байна", "hey", "sn bnu", "snu" гэх мэт), дараах байдлаар хариулаарай:
    
    "Сайн байна уу! Танд хэрхэн туслах вэ?
    
    Би дараах зүйлсээр танд туслаж чадна:
    • 🔍 Бүтээгдэхүүн хайх болон олох
    • 💰 Үнийн мэдээлэл өгөх  
    • 📝 Бүтээгдэхүүний дэлгэрэнгүй мэдээлэл
    • 📷 Зураг танилцуулах, зураг дээрх бүтээгдэхүүн тодорхойлох
    • 🔄 Ижил төстэй бүтээгдэхүүн олох (зураг илгээвэл)
    • 🛒 Худалдан авалтын зөвлөгөө
    • 📞 Холбоо барих мэдээлэл
    • ❓ Бүхий л төрлийн асуултад хариулах
    
    Хайж байгаа бүтээгдэхүүнээ хэлээрэй, зураг илгээгээрэй эсвэл асуултаа чөлөөтэй асуугаарай!"
    
    БҮТЭЭГДЭХҮҮН ХАЙХ ЗАА ЗААВАР:
    1. Хэрэглэгч бүтээгдэхүүн хайж байвал, холбогдох бүтээгдэхүүний мэдээллийг хайж олоорой
    2. Үнэ, загвар, өнгө, хэмжээ зэрэг дэлгэрэнгүй мэдээллийг өгөөрөй
    3. Хэрэв олон төстэй бүтээгдэхүүн байвал, тэдгээрийг жагсааж харьцуулга хийж өгөөрөй
    4. Бүтээгдэхүүний зургийг байвал дурдаарай
    5. Худалдан авах холбоос эсвэл холбоо барих мэдээллийг өгөөрөй
    
    ХАРИУЛТЫН ЗАГВАР:
    - Эхлээд тухайн бүтээгдэхүүний нэр болон товч тайлбарыг өгөөрөй
    - Үнэ болон боломжтой сонголтуудыг (өнгө, хэмжээ г.м) дурдаарай  
    - Онцлог шинж чанарууд болон давуу талуудыг тайлбарлаарай
    - Хэрэв байвал холбогдох линк эсвэл холбоо барих мэдээллийг өгөөрөй
    - Найрсаг, худалдааны амжилттай хэв маягаар хариулаарай
    
    ТУСГАЙ ТОХИОЛДЛУУД:
    - Хэрэв тухайн бүтээгдэхүүн олдохгүй бол, ижил төстэй бүтээгдэхүүн санал болгооройй
    - Үнийн асуултад тодорхой хариулт өгөөрөй
    - Хэрэглэгчийн сонирхлын дагуу нэмэлт санал болгооройй
    - Худалдан авах процессын талаар тайлбарлаж өгөөрөй
    
    ЧУХАЛ: Та бүх төрлийн асуултад хариулах чадвартай. Хэрэв баримт бичгээс тодорхой мэдээлэл олдохгүй байвал, ерөнхий мэдлэг, туршлагаараа тусалж, хэрэглэгчид хамгийн сайн зөвлөгөө өгөөрөй. Дандаа найрсаг, тусламжтай байж, хэрэглэгчийн асуултыг бүрэн хариулахыг хичээрэй."""
    
    if context:
        system_content += f"\n\nКонтекст мэдээлэл:\n{context}"
        
    if similar_images_context:
        system_content += f"\n\nИжил төстэй зургийн мэдээлэл (хэрэглэгчийн илгээсэн зурагтай харьцуулах):\n{similar_images_context}"
    
    # Prepare content for Gemini
    contents = []
    
    # Add user message
    if user_message:
        contents.append(user_message)
    
    # Add image if provided
    if image_data:
        try:
            # Get image bytes and mime type
            image_bytes = image_data.get('data')
            mime_type = image_data.get('mime_type', 'image/jpeg')
            
            if image_bytes:
                # Create image part for Gemini
                image_part = types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=mime_type
                )
                contents.append(image_part)
        except Exception as e:
            logging.error(f"Image processing error: {e}")
            return "🖼️ Зураг боловсруулахад алдаа гарлаа. Дахин оролдоно уу."
    
    try:
        # Generate response with Gemini
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_content,
                max_output_tokens=600,
                temperature=0.7,
            )
        )
        
        ai_response = response.text
        
        # Store in memory
        if conversation_id not in conversation_memory:
            conversation_memory[conversation_id] = []
        
        # Store user message (include mention of image if present)
        user_content = user_message
        if image_data:
            user_content += " [зураг хавсаргасан]"
            
        conversation_memory[conversation_id].append({"role": "user", "content": user_content})
        conversation_memory[conversation_id].append({"role": "assistant", "content": ai_response})
        
        # Keep only last 8 messages
        if len(conversation_memory[conversation_id]) > 8:
            conversation_memory[conversation_id] = conversation_memory[conversation_id][-8:]
            
        return ai_response
        
    except Exception as e:
        logging.error(f"Gemini API алдаа: {e}")
        return f"🔧 Өөрийн системээс хариулт авахад саад гарлаа. Та дараах зүйлсийг туршиж үзнэ үү:\n\n• Асуултаа арай өөрөөр томъёолж илгээнэ үү\n• Илүү тодорхой, тусгай нөхцөлөөр асуугаарай\n• Хайж байгаа зүйлийнхээ нэрийг өөрөөр бичиж үзнэ үү\n• Хэдэн секундын дараа дахин оролдоно уу\n\nБи танд туслахад бэлэн байна! 💪"

def process_chatwoot_attachment(attachment_data):
    """Process image attachment from Chatwoot"""
    try:
        file_type = attachment_data.get('file_type', '')
        file_url = attachment_data.get('data_url', '')
        
        # Check if it's an image
        if not file_type.startswith('image/'):
            return None
            
        # Download the image
        response = requests.get(file_url, timeout=10)
        response.raise_for_status()
        
        image_bytes = response.content
        
        return {
            'data': image_bytes,
            'mime_type': file_type
        }
        
    except Exception as e:
        logging.error(f"Error processing attachment: {e}")
        return None

def search_in_crawled_data(query: str, max_results: int = 3):
    """Enhanced search through crawled data with multiple strategies"""
    if not crawled_data:
        return []
    
    query_lower = query.lower()
    results = []
    scored_results = []
    
    for page in crawled_data:
        title = page['title'].lower()
        body = page['body'].lower()
        
        # Calculate relevance score
        score = 0
        
        # Exact phrase match in title (highest score)
        if query_lower in title:
            score += 10
        
        # Exact phrase match in body
        if query_lower in body:
            score += 5
        
        # Individual word matches
        query_words = query_lower.split()
        for word in query_words:
            if len(word) > 2:  # Skip very short words
                if word in title:
                    score += 3
                if word in body:
                    score += 1
        
        # Partial matches (for Mongolian words)
        for word in query_words:
            if len(word) > 3:
                for title_word in title.split():
                    if word in title_word or title_word in word:
                        score += 2
                for body_word in body.split():
                    if word in body_word or body_word in word:
                        score += 0.5
        
        # Only include pages with some relevance
        if score > 0:
            # Find the best snippet
            best_snippet = ""
            max_context = 400
            
            # Look for best matching context around query words
            for word in query_words:
                if word in body.lower():
                    word_pos = body.lower().find(word)
                    start = max(0, word_pos - 150)
                    end = min(len(body), word_pos + 250)
                    snippet = body[start:end].strip()
                    if len(snippet) > len(best_snippet):
                        best_snippet = snippet
            
            if not best_snippet:
                best_snippet = body[:max_context] + "..." if len(body) > max_context else body
                
            scored_results.append({
                'title': page['title'],
                'url': page['url'],
                'snippet': best_snippet,
                'score': score
            })
    
    # Sort by score (highest first) and return top results
    scored_results.sort(key=lambda x: x['score'], reverse=True)
    
    # Remove score from final results
    for result in scored_results[:max_results]:
        del result['score']
        results.append(result)
            
    return results

# def scrape_single(url: str):
#     resp = requests.get(url, timeout=10)
#     resp.raise_for_status()
#     soup = BeautifulSoup(resp.text, "html.parser")
#     title = soup.title.string.strip() if soup.title else url
#     body, images = extract_content(soup, url)
#     return {"url": url, "title": title, "body": body, "images": images}


# —— Enhanced Chatwoot Integration —— #
def send_to_chatwoot(conv_id: int, content: str, message_type: str = "outgoing"):
    """Enhanced chatwoot message sending with better error handling"""
    api_url = (
        f"{CHATWOOT_BASE_URL}/api/v1/accounts/{ACCOUNT_ID}"
        f"/conversations/{conv_id}/messages"
    )
    headers = {
        "api_access_token": CHATWOOT_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "content": content, 
        "message_type": message_type,
        "private": False
    }
    
    try:
        resp = requests.post(api_url, json=payload, headers=headers, timeout=10)
        resp.raise_for_status()
        logging.info(f"Message sent to conversation {conv_id}")
        return True
    except Exception as e:
        logging.error(f"Failed to send message to chatwoot: {e}")
        return False

def get_conversation_info(conv_id: int):
    """Get conversation details from Chatwoot"""
    api_url = f"{CHATWOOT_BASE_URL}/api/v1/accounts/{ACCOUNT_ID}/conversations/{conv_id}"
    headers = {"api_access_token": CHATWOOT_API_KEY}
    
    try:
        resp = requests.get(api_url, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logging.error(f"Failed to get conversation info: {e}")
        return None

def mark_conversation_resolved(conv_id: int):
    """Mark conversation as resolved"""
    api_url = f"{CHATWOOT_BASE_URL}/api/v1/accounts/{ACCOUNT_ID}/conversations/{conv_id}/toggle_status"
    headers = {"api_access_token": CHATWOOT_API_KEY}
    payload = {"status": "resolved"}
    
    try:
        resp = requests.post(api_url, json=payload, headers=headers, timeout=10)
        resp.raise_for_status()
        return True
    except Exception as e:
        logging.error(f"Failed to mark conversation as resolved: {e}")
        return False


# —— API Endpoints —— #
@app.route("/", methods=["GET"])
def index():
    """Serve the main HTML page with Chatwoot widget"""
    html_content = """<!DOCTYPE html>
<html lang="mn">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Chatwoot Widget</title>
    <script>
      (function (d, t) {
        var BASE_URL = "https://app.chatwoot.com";
        var g = d.createElement(t),
          s = d.getElementsByTagName(t)[0];
        g.src = BASE_URL + "/packs/js/sdk.js";
        g.defer = true;
        g.async = true;
        s.parentNode.insertBefore(g, s);
        g.onload = function () {
          window.chatwootSDK.run({
            websiteToken: "HEpoGsGiY59Tqew6S4yhZbnf",
            baseUrl: BASE_URL,
          });
        };
      })(document, "script");
    </script>
  </head>
  <body>
    <h1>Баруун доор kako.mn AI chatbot-ын хэсэг байна...</h1>
    <!-- Chatwoot widget автоматаар энд гарч ирнэ -->
  </body>
</html>"""
    return html_content

@app.route("/api/scrape", methods=["POST"])
def api_scrape():
    data = request.get_json(force=True)
    url = data.get("url")
    if not url:
        return jsonify({"error": "Missing 'url' in JSON body"}), 400
    try:
        page = scrape_single(url)
        return jsonify(page)
    except Exception as e:
        return jsonify({"error": f"Fetch/Scrape failed: {e}"}), 502

@app.route("/api/crawl", methods=["POST"])
def api_crawl():
    pages = crawl_and_scrape(ROOT_URL)
    return jsonify(pages)


# —— Enhanced Chatwoot Webhook —— #
@app.route("/webhook/chatwoot", methods=["POST"])
def chatwoot_webhook():
    """Enhanced webhook with AI integration using RAG system and image recognition"""
    global crawled_data, crawl_status
    
    data = request.json or {}
    
    # Only process incoming messages
    if data.get("message_type") != "incoming":
        return jsonify({}), 200

    conv_id = data["conversation"]["id"]
    text = data.get("content", "").strip()
    contact = data.get("conversation", {}).get("contact", {})
    contact_name = contact.get("name", "Хэрэглэгч")
    
    # Check for image attachments
    attachments = data.get("attachments", [])
    image_data = None
    
    if attachments:
        for attachment in attachments:
            # Process the first image attachment
            image_data = process_chatwoot_attachment(attachment)
            if image_data:
                logging.info(f"Image attachment received from {contact_name} in conversation {conv_id}")
                break
    
    logging.info(f"Received message from {contact_name} in conversation {conv_id}: {text} {'[with image]' if image_data else ''}")
    
    # Use AI with image support
    ai_response = get_ai_response(text, conv_id, crawled_data, image_data)
    
    # Send AI response directly
    send_to_chatwoot(conv_id, ai_response)

    return jsonify({"status": "success"}), 200


# —— Additional API Endpoints —— #
@app.route("/api/crawl-status", methods=["GET"])
def get_crawl_status():
    """Get current crawl status"""
    return jsonify({
        "crawl_status": crawl_status,
        "crawled_pages": len(crawled_data),
        "config": {
            "root_url": ROOT_URL,
            "auto_crawl_enabled": AUTO_CRAWL_ON_START,
            "max_pages": MAX_CRAWL_PAGES
        }
    })

@app.route("/api/force-crawl", methods=["POST"])
def force_crawl():
    """Force start a new crawl"""
    global crawled_data, crawl_status
    
    # Check if already running
    if crawl_status["status"] == "running":
        return jsonify({"error": "Crawl is already running"}), 409
    
    try:
        crawl_status = {"status": "running", "message": "Force crawl started via API"}
        crawled_data = crawl_and_scrape(ROOT_URL)
        
        if crawled_data:
            crawl_status = {
                "status": "completed",
                "message": f"Force crawl completed via API",
                "pages_count": len(crawled_data),
                "timestamp": datetime.now().isoformat()
            }
            return jsonify({
                "status": "success",
                "pages_crawled": len(crawled_data),
                "crawl_status": crawl_status
            })
        else:
            crawl_status = {"status": "failed", "message": "Force crawl failed - no pages found"}
            return jsonify({"error": "No pages were crawled"}), 500
            
    except Exception as e:
        crawl_status = {"status": "error", "message": f"Force crawl error: {str(e)}"}
        return jsonify({"error": f"Crawl failed: {e}"}), 500

@app.route("/api/search", methods=["POST"])
def api_search():
    """Search through crawled data via API"""
    data = request.get_json(force=True)
    query = data.get("query", "").strip()
    max_results = data.get("max_results", 5)
    
    if not query:
        return jsonify({"error": "Missing 'query' in request body"}), 400
    
    if crawl_status["status"] == "running":
        return jsonify({"error": "Crawl is currently running, please wait"}), 409
    
    if not crawled_data:
        return jsonify({"error": "No crawled data available. Run crawl first."}), 404
    
    results = search_in_crawled_data(query, max_results)
    return jsonify({
        "query": query,
        "results_count": len(results),
        "results": results,
        "crawl_status": crawl_status
    })

@app.route("/api/conversation/<int:conv_id>/memory", methods=["GET"])
def get_conversation_memory(conv_id):
    """Get conversation memory for debugging"""
    memory = conversation_memory.get(conv_id, [])
    return jsonify({"conversation_id": conv_id, "memory": memory, "system": "gemini_multimodal_rag"})

@app.route("/api/conversation/<int:conv_id>/clear", methods=["POST"])
def clear_conversation_memory(conv_id):
    """Clear conversation memory"""
    if conv_id in conversation_memory:
        del conversation_memory[conv_id]
    return jsonify({"status": "cleared", "conversation_id": conv_id, "system": "gemini_multimodal_rag"})

@app.route("/api/crawled-data", methods=["GET"])
def get_crawled_data():
    """Get current crawled data"""
    page_limit = request.args.get('limit', 10, type=int)
    return jsonify({
        "total_pages": len(crawled_data), 
        "crawl_status": crawl_status,
        "data": crawled_data[:page_limit]
    })

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "system_type": "gemini_multimodal_rag_with_image_similarity",
        "timestamp": datetime.now().isoformat(),
        "crawl_status": crawl_status,
        "crawled_pages": len(crawled_data),
        "crawled_images": len(crawled_images),
        "active_conversations": len(conversation_memory),
        "config": {
            "root_url": ROOT_URL,
            "auto_crawl_enabled": AUTO_CRAWL_ON_START,
            "gemini_configured": client is not None,
            "chatwoot_configured": bool(CHATWOOT_API_KEY and ACCOUNT_ID),
            "image_recognition": True,
            "image_similarity_matching": True,
            "images_directory": IMAGES_DIR
        }
    })

@app.route("/api/crawled-images", methods=["GET"])
def get_crawled_images():
    """Get information about crawled images"""
    page_limit = request.args.get('limit', 20, type=int)
    return jsonify({
        "total_images": len(crawled_images),
        "crawl_status": crawl_status,
        "images": [
            {
                "url": img["url"],
                "filename": img["filename"],
                "page_url": img.get("page_url", ""),
                "page_title": img.get("page_title", ""),
                "alt": img.get("alt", ""),
                "width": img.get("width", 0),
                "height": img.get("height", 0),
                "file_size": img.get("file_size", 0),
                "content_type": img.get("content_type", "")
            } for img in crawled_images[:page_limit]
        ]
    })

@app.route("/api/similar-images", methods=["POST"])
def find_similar_images_api():
    """Find similar images to uploaded image via API"""
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No image file selected"}), 400
    
    try:
        # Read image data
        image_bytes = file.read()
        
        # Process image features
        user_image_features = process_user_image_features(image_bytes)
        if not user_image_features:
            return jsonify({"error": "Could not process uploaded image"}), 400
        
        # Find similar images
        similarity_threshold = request.form.get('threshold', 0.3, type=float)
        similar_images = find_similar_crawled_images(user_image_features, similarity_threshold)
        
        return jsonify({
            "similar_images_count": len(similar_images),
            "threshold_used": similarity_threshold,
            "similar_images": similar_images
        })
        
    except Exception as e:
        logging.error(f"Error in similar images API: {e}")
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

@app.route("/api/clear-images", methods=["POST"])
def clear_crawled_images():
    """Clear all crawled images"""
    global crawled_images
    
    try:
        # Remove image files
        if os.path.exists(IMAGES_DIR):
            import shutil
            shutil.rmtree(IMAGES_DIR)
            os.makedirs(IMAGES_DIR, exist_ok=True)
        
        # Clear memory
        crawled_images = []
        
        return jsonify({
            "status": "success",
            "message": "All crawled images cleared",
            "images_remaining": len(crawled_images)
        })
        
    except Exception as e:
        logging.error(f"Error clearing images: {e}")
        return jsonify({"error": f"Failed to clear images: {str(e)}"}), 500

@app.route("/images/<filename>", methods=["GET"])
def serve_crawled_image(filename):
    """Serve crawled images"""
    try:
        return send_from_directory(IMAGES_DIR, filename)
    except Exception as e:
        logging.error(f"Error serving image {filename}: {e}")
        return jsonify({"error": "Image not found"}), 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)