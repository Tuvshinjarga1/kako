import os
import time
import logging
import requests
import json
import base64
from urllib.parse import urljoin, urlparse
from flask import Flask, request, jsonify, render_template_string, send_from_directory
from bs4 import BeautifulSoup
from datetime import datetime
from typing import Dict, Optional, List
from PIL import Image
import io

# OpenAI client импорт
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI client not installed. Install with: pip install openai")

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
OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY")
AUTO_CRAWL_ON_START  = os.getenv("AUTO_CRAWL_ON_START", "true").lower() == "true"

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY) if (OPENAI_API_KEY and OPENAI_AVAILABLE) else None

# —— Memory Storage —— #
conversation_memory = {}
crawled_data = []
crawl_status = {"status": "not_started", "message": "Crawling has not started yet"}

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

# —— Content Extraction —— #
def extract_content(soup: BeautifulSoup, base_url: str):
    main = soup.find("main") or soup
    texts = []
    images = []

    for tag in main.find_all(["h1", "h2", "h3", "h4", "p", "li", "code"]):
        text = tag.get_text(strip=True)
        if text:
            texts.append(text)

    for img in main.find_all("img"):
        src = img.get("src")
        alt = img.get("alt", "").strip()
        if src:
            full_img_url = urljoin(base_url, src)
            entry = f"[Image] {alt} — {full_img_url}" if alt else f"[Image] {full_img_url}"
            texts.append(entry)
            images.append({"url": full_img_url, "alt": alt})

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


# —— Image Processing Functions —— #
def encode_image_to_base64(image_url: str) -> Optional[str]:
    """Download and encode image to base64"""
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        
        # Convert to base64
        base64_image = base64.b64encode(response.content).decode('utf-8')
        return base64_image
    except Exception as e:
        logging.error(f"Failed to encode image {image_url}: {e}")
        return None

def analyze_image_with_gpt4(image_data: str, user_question: str = "") -> str:
    """Analyze image using GPT-4 Vision"""
    if not client:
        return "🔑 OpenAI API түлхүүр тохируулагдаагүй байна."
    
    try:
        messages = [
            {
                "role": "system",
                "content": """Та онлайн дэлгүүрийн AI туслах бот юм. Хэрэглэгчээс ирсэн зургийг танин мэдэж, тухайн зургтай холбоотой бүтээгдэхүүний мэдээлэл, үнэ, онцлог шинж чанарыг монгол хэлээр тайлбарлаарай.

ЗУРГИЙН ШИНЖИЛГЭЭНИЙ ЗААВАР:
1. Зурагт юу харагдаж байгааг тодорхой дурдаарай
2. Хэрэв бүтээгдэхүүн бол, нэр, загвар, өнгө, материалыг тайлбарлаарай
3. Худалдан авах боломжтой эсэхийг дурдаарай
4. Ижил төстэй бүтээгдэхүүн санал болгооройй
5. Үнийн мэдээлэл байвал дурдаарай

Найрсаг, тусламжтай хариулт өгөөрөй."""
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Энэ зургийг үзээд тайлбарлаж өгнө үү? {user_question}" if user_question else "Энэ зургийг үзээд тайлбарлаж өгнө үү?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}"
                        }
                    }
                ]
            }
        ]
        
        response = client.chat.completions.create(
            model="gpt-4o",  # Use GPT-4o for better vision and text capabilities
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logging.error(f"GPT-4 Vision error: {e}")
        return f"🔧 Зургийг шинжлэхэд алдаа гарлаа: {str(e)[:100]}"

def extract_images_from_chatwoot_message(message_data: dict) -> List[str]:
    """Extract image URLs from Chatwoot message attachments"""
    images = []
    
    # Check for attachments
    attachments = message_data.get("attachments", [])
    for attachment in attachments:
        if attachment.get("file_type") and attachment["file_type"].startswith("image/"):
            data_url = attachment.get("data_url")
            if data_url:
                images.append(data_url)
    
    return images

# —— AI Assistant Functions —— #
def get_ai_response(user_message: str, conversation_id: int, context_data: list = None, images: List[str] = None):
    """Enhanced AI response with OpenAI's GPT-4 Vision for text and image support"""
    
    print(user_message)

    if not client:
        return "🔑 OpenAI API түлхүүр тохируулагдаагүй байна. Админтай холбогдоно уу."
    
    # Handle empty or None messages
    if not user_message and not images:
        return "📝 Мессежийн агуулга алга байна. Асуултаа дахин илгээнэ үү эсвэл зураг оруулна уу."
    
    # Use default message for image-only requests
    if not user_message and images:
        user_message = "Энэ зургийг шинжилж тайлбарлаж өгнө үү?"
    
    # Handle image analysis first if images are provided
    if images:
        image_responses = []
        for image_url in images:
            # Encode image to base64
            base64_image = encode_image_to_base64(image_url)
            if base64_image:
                image_analysis = analyze_image_with_gpt4(base64_image, user_message)
                image_responses.append(image_analysis)
        
        if image_responses:
            combined_response = "\n\n".join(image_responses)
            
            # Store in memory
            if conversation_id not in conversation_memory:
                conversation_memory[conversation_id] = []
            
            conversation_memory[conversation_id].append({
                "role": "user", 
                "content": f"{user_message} [Зурагтай]"
            })
            conversation_memory[conversation_id].append({
                "role": "assistant", 
                "content": combined_response
            })
            
            # Keep only last 8 messages
            if len(conversation_memory[conversation_id]) > 8:
                conversation_memory[conversation_id] = conversation_memory[conversation_id][-8:]
                
            return combined_response
    
    # Get conversation history
    history = conversation_memory.get(conversation_id, [])
    
    # Build context from crawled data if available
    context = ""
    if crawled_data:
        # Search for relevant content
        search_results = search_in_crawled_data(user_message, max_results=3)
        if search_results:
            relevant_pages = []
            for result in search_results:
                relevant_pages.append(
                    f"Хуудас: {result['title']}\n"
                    f"URL: {result['url']}\n"
                    f"Холбогдох агуулга: {result['snippet']}\n"
                )
            context = "\n\n".join(relevant_pages)
    
    # Build system message with context
    system_content = """Та онлайн дэлгүүрийн AI туслах бот юм. Хэрэглэгчдэд бүтээгдэхүүний мэдээлэл, үнэ, дэлгэрэнгүй мэдээлэл хайж олоход тусалдаг.
    Хэрэглэгчтэй монгол хэлээр найрсаг, тусламжтай ярилцаарай.
    
    ЭНГИЙН МЭНДЧИЛГЭЭНИЙ ТУХАЙ:
    Хэрэв хэрэглэгч энгийн мэндчилгээ хийж байвал (жишээ: "сайн байна уу", "сайн уу", "мэнд", "hello", "hi", "сайн уу байна", "hey", "sn bnu", "snu" гэх мэт), дараах байдлаар хариулаарай:
    
    "Сайн байна уу! Танд хэрхэн туслах вэ?
    
    Би дараах зүйлсээр танд туслаж чадна:
    • 🔍 Бүтээгдэхүүн хайх болон олох
    • 💰 Үнийн мэдээлэл өгөх  
    • 📝 Бүтээгдэхүүний дэлгэрэнгүй мэдээлэл
    • 🛒 Худалдан авалтын зөвлөгөө
    • 📞 Холбоо барих мэдээлэл
    • 📸 Зураг танин бүтээгдэхүүн олох
    
    Хайж байгаа бүтээгдэхүүнээ хэлээрэй эсвэл зургийг илгээгээрэй!"
    
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
    - Худалдан авах процессын талаар тайлбарлаж өгөөрөй"""
    
    if context:
        system_content += f"\n\nКонтекст мэдээлэл:\n{context}"
    
    # Build conversation messages for OpenAI
    messages = [
        {
            "role": "system",
            "content": system_content
        }
    ]
    
    # Add conversation history
    for msg in history[-4:]:  # Last 4 messages
        if msg.get("role") == "user":
            messages.append({"role": "user", "content": msg["content"]})
        elif msg.get("role") == "assistant":
            messages.append({"role": "assistant", "content": msg["content"]})
    
    # Add current message
    messages.append({"role": "user", "content": user_message})
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Use GPT-4o for better vision and text capabilities
            messages=messages,
            max_tokens=600,
            temperature=0.7,
        )
        
        ai_response = response.choices[0].message.content
        
        # Store in memory
        if conversation_id not in conversation_memory:
            conversation_memory[conversation_id] = []
        
        conversation_memory[conversation_id].append({"role": "user", "content": user_message})
        conversation_memory[conversation_id].append({"role": "assistant", "content": ai_response})
        
        # Keep only last 8 messages
        if len(conversation_memory[conversation_id]) > 8:
            conversation_memory[conversation_id] = conversation_memory[conversation_id][-8:]
            
        return ai_response
        
    except Exception as e:
        logging.error(f"OpenAI API алдаа: {e}")
        return f"🔧 AI-тай холбогдоход саад гарлаа. Дараах зүйлсийг туршиж үзнэ үү:\n• Асуултаа дахин илгээнэ үү\n• Асуултаа тодорхой болгоно уу\n• Холбогдох мэдээллийг хайж үзнэ үү\n\nАлдааны дэлгэрэнгүй: {str(e)[:100]}"

def search_in_crawled_data(query: str, max_results: int = 3):
    """Simple search through crawled data"""
    if not crawled_data:
        return []
    
    query_lower = query.lower()
    results = []
    
    for page in crawled_data:
        title = page['title'].lower()
        body = page['body'].lower()
        
        # Check if query matches in title or body
        if (query_lower in title or 
            query_lower in body or 
            any(word in title or word in body for word in query_lower.split())):
            
            # Find the most relevant snippet
            query_words = query_lower.split()
            best_snippet = ""
            max_context = 300
            
            for word in query_words:
                if word in body.lower():
                    start = max(0, body.lower().find(word) - 100)
                    end = min(len(body), body.lower().find(word) + 200)
                    snippet = body[start:end]
                    if len(snippet) > len(best_snippet):
                        best_snippet = snippet
            
            if not best_snippet:
                best_snippet = body[:max_context] + "..." if len(body) > max_context else body
                
            results.append({
                'title': page['title'],
                'url': page['url'],
                'snippet': best_snippet
            })
            
            # Stop when we have enough results
            if len(results) >= max_results:
                break
            
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
    """Enhanced webhook with AI integration and image recognition"""
    global crawled_data, crawl_status
    
    data = request.json or {}
    
    # Only process incoming messages
    if data.get("message_type") != "incoming":
        return jsonify({}), 200

    try:
        conv_id = data["conversation"]["id"]
        # Fix: Handle None values properly
        content = data.get("content") or ""
        text = content.strip() if isinstance(content, str) else ""
        contact = data.get("conversation", {}).get("contact", {})
        contact_name = contact.get("name", "Хэрэглэгч")
        
        # Extract images from message attachments
        images = extract_images_from_chatwoot_message(data)
        
        if images:
            logging.info(f"Received message with {len(images)} image(s) from {contact_name} in conversation {conv_id}: {text}")
        else:
            logging.info(f"Received text message from {contact_name} in conversation {conv_id}: {text}")
        
        # Get conversation history
        history = conversation_memory.get(conv_id, [])
        
        # Try to answer with AI (including image analysis if images present)
        ai_response = get_ai_response(text, conv_id, crawled_data, images)
        
        # Check if AI couldn't find good answer by searching crawled data
        search_results = search_in_crawled_data(text, max_results=3) if text else []
        
        # Check if this user was previously escalated but asking a new question
        was_previously_escalated = any(
            msg.get("role") == "system" and "escalated_to_human" in msg.get("content", "")
            for msg in history
        )
        
        # Let AI evaluate its own response quality and decide if human help is needed
        # Skip escalation check for image messages as AI can handle them well
        needs_human_help = False
        if not images and text:  # Only check for text messages with content
            needs_human_help = should_escalate_to_human(text, search_results, ai_response, history)
        
        # If user was previously escalated but AI can answer this new question, respond with AI
        if was_previously_escalated and not needs_human_help:
            # AI can handle this new question even though user was escalated before
            if images:
                response_with_note = f"{ai_response}\n\n📸 Зургийг амжилттай шинжиллээ! Хэрэв нэмэлт асуулт байвал чөлөөтэй асуугаарай."
            else:
                response_with_note = f"{ai_response}\n\n💡 Хэрэв энэ хариулт хангалтгүй бол, дэмжлэгийн багтай холбогдоно уу."
            send_to_chatwoot(conv_id, response_with_note)
            return jsonify({"status": "success"}), 200
        
        if needs_human_help:
            # Mark this conversation as escalated
            if conv_id not in conversation_memory:
                conversation_memory[conv_id] = []
            conversation_memory[conv_id].append({
                "role": "system", 
                "content": "escalated_to_human"
            })
            
            # AI thinks it can't handle this properly, escalate to human
            escalation_response = """🤝 Би таны асуултад хангалттай хариулт өгч чадахгүй байна. Удахгүй таны асуултад ажилтан хариулт өгөх болно.

Тусламжийн баг удахгүй танд хариулт өгөх болно."""
            
            send_to_chatwoot(conv_id, escalation_response)
        else:
            # AI is confident in its response, send it
            if images:
                # Add emoji to indicate image was processed
                ai_response_with_icon = f"📸 {ai_response}"
                send_to_chatwoot(conv_id, ai_response_with_icon)
            else:
                send_to_chatwoot(conv_id, ai_response)

        return jsonify({"status": "success"}), 200
        
    except KeyError as e:
        logging.error(f"Missing required field in webhook data: {e}")
        return jsonify({"error": f"Missing required field: {e}"}), 400
    except Exception as e:
        logging.error(f"Webhook processing error: {e}")
        return jsonify({"error": "Internal server error"}), 500


def should_escalate_to_human(user_message: str, search_results: list, ai_response: str, history: list) -> bool:
    """AI evaluates its own response and decides if human help is needed using OpenAI"""
    
    # Handle empty parameters
    if not user_message or not ai_response:
        return False  # Don't escalate if no content to evaluate
    
    # Use OpenAI to evaluate its own response quality
    if not client:
        # Fallback without AI evaluation - be more lenient
        return len(user_message) > 50 and (not search_results or len(search_results) == 0)
    
    # Build context for AI self-evaluation
    context = f"""Хэрэглэгчийн асуулт: "{user_message}"

Манай баримт бичгээс хайсан үр дүн:
{f"Олдсон: {len(search_results)} үр дүн" if search_results else "Мэдээлэл олдсонгүй"}

Миний өгсөн хариулт: "{ai_response}"

Ярилцлагын сүүлийн мессежүүд:"""
    
    if history:
        recent_messages = [msg.get("content", "")[:100] for msg in history[-3:] if msg.get("role") == "user"]
        if recent_messages:
            context += "\n" + "\n".join(recent_messages)
    
    try:
        messages = [
            {
                "role": "system",
                "content": """Та өөрийн өгсөн хариултыг үнэлж, хэрэглэгчид хангалттай эсэхийг шийднэ.

Дараах тохиолдлуудад л хүний ажилтны тусламж шаардлагатай:
- Хэрэглэгч техникийн алдаа, тохиргооны асуудлаар тусламж хүсэж байгаа
- Акаунт, төлбөр, хостинг, домэйн зэрэг онлайн дэлгүүрийн үйлчилгээтэй холбоотой асуудал
- Тусгай хүсэлт, гомдол, шуурхай тусламж хэрэгтэй асуудал
- Хэрэглэгч өөрөө "ажилтныг хүсэж байна" гэж тодорхой хэлсэн тохиолдол
- Миний хариулт нь хэрэглэгчийн асуултын үндсэн сэдвээс огт холдсон бол

Дараах тохиолдлуудад хүний тусламж ШААРДЛАГАГҮЙ:
- Энгийн мэдээлэл асуух (бүтээгдэхүүний тухай)
- Ерөнхий зөвлөгөө авах
- Худалдан авах мэдлэг судлах
- Би хангалттай хариулт өгч чадсан тохиолдол
- Хэрэглэгч зүгээр л мэдээлэл хайж байгаа

Өөрийнхөө хариултанд итгэлтэй байж, хэрэглэгч дахин асууж болно гэдгийг санаарай.

Хариултаа зөвхөн 'YES' (хүний тусламж хэрэгтэй) эсвэл 'NO' (миний хариулт хангалттай) гэж өгнө үү."""
            },
            {
                "role": "user", 
                "content": context
            }
        ]
        
        response = client.chat.completions.create(
            model="gpt-4o",  # Use GPT-4o for better vision and text capabilities
            messages=messages,
            max_tokens=10,
            temperature=0.2
        )
        
        ai_decision = response.choices[0].message.content.strip().upper()
        logging.info(f"OpenAI self-evaluation for '{user_message[:30]}...': {ai_decision}")
        return ai_decision == "YES"
        
    except Exception as e:
        logging.error(f"OpenAI self-evaluation error: {e}")
        # More lenient fallback - don't escalate by default
        return False


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
    return jsonify({"conversation_id": conv_id, "memory": memory})

@app.route("/api/conversation/<int:conv_id>/clear", methods=["POST"])
def clear_conversation_memory(conv_id):
    """Clear conversation memory"""
    if conv_id in conversation_memory:
        del conversation_memory[conv_id]
    return jsonify({"status": "cleared", "conversation_id": conv_id})

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
        "timestamp": datetime.now().isoformat(),
        "crawl_status": crawl_status,
        "crawled_pages": len(crawled_data),
        "active_conversations": len(conversation_memory),
        "config": {
            "root_url": ROOT_URL,
            "auto_crawl_enabled": AUTO_CRAWL_ON_START,
            "openai_configured": client is not None,
            "model": "gpt-4o",
            "image_recognition": True,
            "chatwoot_configured": bool(CHATWOOT_API_KEY and ACCOUNT_ID)
        }
    })

@app.route("/api/analyze-image", methods=["POST"])
def api_analyze_image():
    """Analyze image via API"""
    try:
        data = request.get_json(force=True)
        image_url = data.get("image_url")
        question = data.get("question", "")
        
        if not image_url:
            return jsonify({"error": "Missing 'image_url' in request body"}), 400
        
        if not client:
            return jsonify({"error": "OpenAI API not configured"}), 500
            
        # Encode image to base64
        base64_image = encode_image_to_base64(image_url)
        if not base64_image:
            return jsonify({"error": "Failed to process image"}), 400
            
        # Analyze image
        analysis = analyze_image_with_gpt4(base64_image, question)
        
        return jsonify({
            "image_url": image_url,
            "question": question,
            "analysis": analysis,
            "model": "gpt-4o",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logging.error(f"Image analysis API error: {e}")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)