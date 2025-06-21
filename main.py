import os
import time
import logging
import requests
from openai import OpenAI
import json
from urllib.parse import urljoin, urlparse
from flask import Flask, request, jsonify
from bs4 import BeautifulSoup
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import re
import random
from typing import Dict, Optional

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# —— Config —— #
ROOT_URL             = os.getenv("ROOT_URL", "https://kako.mn/")
DELAY_SEC            = float(os.getenv("DELAY_SEC", "0.5"))
ALLOWED_NETLOC       = urlparse(ROOT_URL).netloc
MAX_CRAWL_PAGES      = int(os.getenv("MAX_CRAWL_PAGES", "50"))
CHATWOOT_API_KEY     = os.getenv("CHATWOOT_API_KEY")
ACCOUNT_ID           = os.getenv("ACCOUNT_ID")
CHATWOOT_BASE_URL    = os.getenv("CHATWOOT_BASE_URL", "https://app.chatwoot.com")
OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY")
AUTO_CRAWL_ON_START  = os.getenv("AUTO_CRAWL_ON_START", "true").lower() == "true"

# SMTP тохиргоо
SMTP_SERVER          = os.getenv("SMTP_SERVER")
SMTP_PORT            = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME        = os.getenv("SENDER_EMAIL")
SMTP_PASSWORD        = os.getenv("SENDER_PASSWORD")
SMTP_FROM_EMAIL      = os.getenv("SENDER_EMAIL")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

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
        title = soup.title.string.strip() if soup.title else url
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
    title = soup.title.string.strip() if soup.title else url
    body, images = extract_content(soup, url)
    return {"url": url, "title": title, "body": body, "images": images}


# —— AI Assistant Functions —— #
def get_ai_response(user_message: str, conversation_id: int, context_data: list = None):
    """Enhanced AI response with better context awareness"""
    
    if not client:
        return "🔑 OpenAI API түлхүүр тохируулагдаагүй байна. Админтай холбогдоно уу."
    
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
    system_content = """Та Cloud.mn-ийн баримт бичгийн талаар асуултад хариулдаг Монгол AI туслах юм. 
    Хэрэглэгчтэй монгол хэлээр ярилцаарай. Хариултаа товч бөгөөд ойлгомжтой байлгаарай.
    
    ЭНГИЙН МЭНДЧИЛГЭЭНИЙ ТУХАЙ:
    Хэрэв хэрэглэгч энгийн мэндчилгээ хийж байвал (жишээ: "сайн байна уу", "сайн уу", "мэнд", "hello", "hi", "сайн уу байна", "hey", "sn bnu", "snu" гэх мэт), дараах байдлаар хариулаарай:
    
    "Сайн байна уу! 👋 Би Cloud.mn-ийн AI туслах юм. Танд хэрхэн туслах вэ?
    
    Би дараах зүйлсээр танд туслаж чадна:
    • 📚 Cloud.mn баримт бичгээс мэдээлэл хайх
    • ❓ Техникийн асуултад хариулах  
    • 💬 Ерөнхий зөвлөгөө өгөх
    
    Асуултаа чөлөөтэй асуугаарай!"
    
    Хариулахдаа дараах зүйлсийг анхаарна уу:
    1. Хариултаа холбогдох баримт бичгийн линкээр дэмжүүлээрэй
    2. Хэрэв ойлгомжгүй бол тодорхой асууна уу
    3. Хариултаа бүтэцтэй, цэгцтэй байлгаарай
    4. Техникийн нэр томъёог монгол хэлээр тайлбарлаарай
    
    Хэрэглэгчийн хүсэлтийг автоматаар таньж, дараах үйлдлүүдийг хийх боломжтой:
    - Хэрэглэгч мэдээлэл хайхыг хүсвэл, холбогдох мэдээллийг хайж олж хариулна
    - Хэрэглэгч тодорхой хуудсыг шүүрдэхийг хүсвэл, тухайн хуудсыг шүүрдэж хариулна
    - Хэрэглэгч тусламж хүсвэл, боломжтой үйлдлүүдийн талаар тайлбарлана
    - Хэрэглэгч бүх сайтыг шүүрдэхийг хүсвэл, шүүрдэлтийг эхлүүлнэ"""
    
    if context:
        system_content += f"\n\nКонтекст мэдээлэл:\n{context}"
    
    # Build conversation context
    messages = [
        {
            "role": "system", 
            "content": system_content
        }
    ]
    
    # Add conversation history
    for msg in history[-4:]:  # Last 4 messages
        messages.append(msg)
    
    # Add current message
    messages.append({"role": "user", "content": user_message})
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=500,  # Increased token limit for better responses
            temperature=0.7
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
    """Enhanced webhook with AI integration"""
    global crawled_data, crawl_status
    
    data = request.json or {}
    
    # Only process incoming messages
    if data.get("message_type") != "incoming":
        return jsonify({}), 200

    conv_id = data["conversation"]["id"]
    text = data.get("content", "").strip()
    contact = data.get("conversation", {}).get("contact", {})
    contact_name = contact.get("name", "Хэрэглэгч")
    
    logging.info(f"Received message from {contact_name} in conversation {conv_id}: {text}")
    
    # Get conversation history
    history = conversation_memory.get(conv_id, [])
    
    # Check if this is an email address
    if "@" in text and is_valid_email(text.strip()):
        # Store email for confirmation
        if conv_id not in conversation_memory:
            conversation_memory[conv_id] = []
        conversation_memory[conv_id].append({
            "role": "system", 
            "content": f"pending_email:{text.strip()}"
        })
        
        response = f"📧 Таны оруулсан имэйл хаяг: {text.strip()}\n\nТа дахин шалгана уу, зөв бол 'y' буруу бол 'n' гэж бичнэ үү."
        send_to_chatwoot(conv_id, response)
        return jsonify({"status": "success"}), 200
    
    # Check if user is confirming email with 'tiim' or 'ugui'
    if text.lower() in ['tiim', 'тийм', 'yes', 'y']:
        # Look for pending email
        pending_email = None
        for msg in history:
            if msg.get("role") == "system" and "pending_email:" in msg.get("content", ""):
                pending_email = msg.get("content").split(":")[1]
                break
        
        if pending_email:
            verification_code = send_verification_email(pending_email)
            if verification_code:
                # Remove pending email and add verification code
                conversation_memory[conv_id] = [msg for msg in conversation_memory[conv_id] 
                                               if not (msg.get("role") == "system" and "pending_email:" in msg.get("content", ""))]
                conversation_memory[conv_id].append({
                    "role": "system", 
                    "content": f"verification_code:{verification_code},email:{pending_email}"
                })
                
                response = "📧 Таны имэйл хаяг руу баталгаажуулах 6 оронтой код илгээлээ. Уг кодыг оруулна уу."
                send_to_chatwoot(conv_id, response)
                return jsonify({"status": "success"}), 200
            else:
                response = "❌ Имэйл илгээхэд алдаа гарлаа. Дахин оролдоно уу эсвэл өөр имэйл хаяг оруулна уу."
                send_to_chatwoot(conv_id, response)
                return jsonify({"status": "success"}), 200
        else:
            response = "⚠️ Баталгаажуулах имэйл хаяг олдсонгүй. Эхлээд имэйл хаягаа оруулна уу."
            send_to_chatwoot(conv_id, response)
            return jsonify({"status": "success"}), 200
    
    # Check if user is rejecting email with 'ugui'
    if text.lower() in ['ugui', 'үгүй', 'no', 'n']:
        # Remove pending email
        if conv_id in conversation_memory:
            conversation_memory[conv_id] = [msg for msg in conversation_memory[conv_id] 
                                           if not (msg.get("role") == "system" and "pending_email:" in msg.get("content", ""))]
        
        response = "❌ Имэйл хаяг буруу байлаа. Зөв имэйл хаягаа дахин оруулна уу."
        send_to_chatwoot(conv_id, response)
        return jsonify({"status": "success"}), 200
    
    # Check if this is a verification code (6 digits)
    if len(text) == 6 and text.isdigit():
        verification_info = None
        for msg in history:
            if msg.get("role") == "system" and "verification_code:" in msg.get("content", ""):
                verification_info = msg.get("content")
                break
        
        if verification_info:
            parts = verification_info.split(",")
            stored_code = parts[0].split(":")[1]
            email = parts[1].split(":")[1]
            
            # Count failed attempts
            failed_attempts = sum(1 for msg in history 
                                if msg.get("role") == "assistant" 
                                and "❌ Баталгаажуулах код буруу байна" in msg.get("content", ""))
            
            if text == stored_code:
                response = "✅ Баталгаажуулалт амжилттай! Одоо асуудлаа дэлгэрэнгүй бичнэ үү."
                send_to_chatwoot(conv_id, response)
                
                conversation_memory[conv_id].append({
                    "role": "system", 
                    "content": f"verified_email:{email}"
                })
                return jsonify({"status": "success"}), 200
            else:
                # Handle failed verification attempts
                if failed_attempts >= 2:  # Allow 3 total attempts (0, 1, 2)
                    response = """❌ Баталгаажуулах кодыг 3 удаа буруу оруулсан тул шинэ код авах шаардлагатай. 
                    
Шинэ код авахын тулд имэйл хаягаа дахин оруулна уу."""
                    send_to_chatwoot(conv_id, response)
                    
                    # Remove old verification code from memory
                    conversation_memory[conv_id] = [msg for msg in conversation_memory[conv_id] 
                                                   if not (msg.get("role") == "system" and "verification_code:" in msg.get("content", ""))]
                    return jsonify({"status": "success"}), 200
                else:
                    remaining_attempts = 2 - failed_attempts
                    response = f"""❌ Баталгаажуулах код буруу байна. 
                    
Танд {remaining_attempts} удаа оролдох боломж үлдлээ. Имэйлээ шалгаж, зөв кодыг оруулна уу."""
                    send_to_chatwoot(conv_id, response)
                    return jsonify({"status": "success"}), 200
        else:
            # No verification code found in memory
            response = """⚠️ Баталгаажуулах код олдсонгүй. 
            
Эхлээд имэйл хаягаа оруулж, баталгаажуулах код авна уу."""
            send_to_chatwoot(conv_id, response)
            return jsonify({"status": "success"}), 200
    
    # Check if user has verified email and is describing an issue
    verified_email = None
    for msg in history:
        if msg.get("role") == "system" and "verified_email:" in msg.get("content", ""):
            verified_email = msg.get("content").split(":")[1]
            break
    
    if verified_email and len(text) > 15:  # User has verified email and writing detailed message
        # Send confirmation email to user
        confirmation_sent = send_confirmation_email(verified_email, text[:100] + "..." if len(text) > 100 else text)
        
        response = "✅ Таны асуудлыг хүлээн авлаа. Бид тантай удахгүй холбогдох болно. Баярлалаа!"
        
        if confirmation_sent:
            response += "\n📧 Танд баталгаажуулах мэйл илгээлээ."
        send_to_chatwoot(conv_id, response)
        return jsonify({"status": "success"}), 200
    
    # Try to answer with AI first
    ai_response = get_ai_response(text, conv_id, crawled_data)
    
    # Check if AI couldn't find good answer by searching crawled data
    search_results = search_in_crawled_data(text, max_results=3)
    
    # Check if this user was previously escalated but asking a new question
    was_previously_escalated = any(
        msg.get("role") == "system" and "escalated_to_human" in msg.get("content", "")
        for msg in history
    )
    
    # Let AI evaluate its own response quality and decide if human help is needed
    needs_human_help = should_escalate_to_human(text, search_results, ai_response, history)
    
    # If user was previously escalated but AI can answer this new question, respond with AI
    if was_previously_escalated and not needs_human_help:
        # AI can handle this new question even though user was escalated before
        response_with_note = f"{ai_response}\n\n💡 Хэрэв энэ хариулт хангалтгүй бол, имэйл хаягаа оруулж дэмжлэгийн багтай холбогдоно уу."
        send_to_chatwoot(conv_id, response_with_note)
        return jsonify({"status": "success"}), 200
    
    if needs_human_help and not verified_email:
        # Mark this conversation as escalated
        if conv_id not in conversation_memory:
            conversation_memory[conv_id] = []
        conversation_memory[conv_id].append({
            "role": "system", 
            "content": "escalated_to_human"
        })
        
        # AI thinks it can't handle this properly, escalate to human
        escalation_response = """🤝 Би таны асуултад хангалттай хариулт өгч чадахгүй байна. Дэмжлэгийн багийн тусламж авахыг санал болгож байна.

Тусламж авахын тулд имэйл хаягаа оруулна уу. Бид таны имэйл хаягийг баталгаажуулсны дараа асуудлыг шийдвэрлэх болно."""
        
        send_to_chatwoot(conv_id, escalation_response)
    else:
        # AI is confident in its response, send it
        send_to_chatwoot(conv_id, ai_response)

    return jsonify({"status": "success"}), 200


def should_escalate_to_human(user_message: str, search_results: list, ai_response: str, history: list) -> bool:
    """AI evaluates its own response and decides if human help is needed"""
    
    # Use AI to evaluate its own response quality
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
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": """Та өөрийн өгсөн хариултыг үнэлж, хэрэглэгчид хангалттай эсэхийг шийднэ.

Дараах тохиолдлуудад л хүний ажилтны тусламж шаардлагатай:
- Хэрэглэгч техникийн алдаа, тохиргооны асуудлаар тусламж хүсэж байгаа
- Акаунт, төлбөр, хостинг, домэйн зэрэг Cloud.mn-ийн үйлчилгээтэй холбоотой асуудал
- Тусгай хүсэлт, гомдол, шуурхай тусламж хэрэгтэй асуудал
- Хэрэглэгч өөрөө "ажилтныг хүсэж байна" гэж тодорхой хэлсэн тохиолдол
- Миний хариулт нь хэрэглэгчийн асуултын үндсэн сэдвээс огт холдсон бол

Дараах тохиолдлуудад хүний тусламж ШААРДЛАГАГҮЙ:
- Энгийн мэдээлэл асуух (Cloud.mn docs-ийн тухай)
- Ерөнхий зөвлөгөө авах
- Техникийн мэдлэг судлах
- Би хангалттай хариулт өгч чадсан тохиолдол
- Хэрэглэгч зүгээр л мэдээлэл хайж байгаа

Өөрийнхөө хариултанд итгэлтэй байж, хэрэглэгч дахин асууж болно гэдгийг санаарай.

Хариултаа зөвхөн 'YES' (хүний тусламж хэрэгтэй) эсвэл 'NO' (миний хариулт хангалттай) гэж өгнө үү."""
                },
                {
                    "role": "user", 
                    "content": context
                }
            ],
            max_tokens=10,
            temperature=0.2
        )
        
        ai_decision = response.choices[0].message.content.strip().upper()
        logging.info(f"AI self-evaluation for '{user_message[:30]}...': {ai_decision}")
        return ai_decision == "YES"
        
    except Exception as e:
        logging.error(f"AI self-evaluation error: {e}")
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
            "chatwoot_configured": bool(CHATWOOT_API_KEY and ACCOUNT_ID),
            "smtp_configured": bool(SMTP_SERVER and SMTP_USERNAME and SMTP_PASSWORD)
        }
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

# —— Email Verification Functions —— #
def is_valid_email(email: str) -> bool:
    """Check if email format is valid"""
    email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(email_regex, email))

def send_verification_email(email: str) -> str:
    """Send verification email with code and return the code"""
    if not SMTP_FROM_EMAIL or not SMTP_PASSWORD or not SMTP_SERVER:
        logging.error("SMTP credentials not configured")
        return None
        
    # Generate verification code
    verification_code = ''.join([str(random.randint(0, 9)) for _ in range(6)])
    
    # Create email
    msg = MIMEMultipart()
    msg['From'] = SMTP_FROM_EMAIL
    msg['To'] = email
    msg['Subject'] = "Cloud.mn баталгаажуулах код"
    
    body = f"""Сайн байна уу,

Таны Cloud.mn-д хандсан хүсэлтийг баталгаажуулахын тулд доорх кодыг оруулна уу:

{verification_code}

Хэрэв та энэ хүсэлтийг илгээгээгүй бол мэдэгдэнэ үү.

Хүндэтгэсэн,
Cloud.mn тусламжийн үйлчилгээ"""
    
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.send_message(msg)
        server.quit()
        logging.info(f"Verification email sent to {email}")
        return verification_code
    except Exception as e:
        logging.error(f"Failed to send verification email: {e}")
        return None

def send_confirmation_email(email: str, problem: str) -> bool:
    """Send confirmation email after issue is sent to support team"""
    if not SMTP_FROM_EMAIL or not SMTP_PASSWORD or not SMTP_SERVER:
        logging.error("SMTP credentials not configured")
        return False
        
    # Create email
    msg = MIMEMultipart()
    msg['From'] = SMTP_FROM_EMAIL
    msg['To'] = email
    msg['Subject'] = "Cloud.mn - Таны хүсэлтийг хүлээн авлаа"
    
    body = f"""Сайн байна уу,

Таны "{problem}" асуудлыг тусламжийн баг руу амжилттай илгээлээ.

Бид таны хүсэлтийг хүлээн авч, удахгүй танд хариу өгөх болно.

Хүндэтгэсэн,
Cloud.mn тусламжийн үйлчилгээ"""
    
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.send_message(msg)
        server.quit()
        logging.info(f"Confirmation email sent to {email}")
        return True
    except Exception as e:
        logging.error(f"Failed to send confirmation email: {e}")
        return False