import os
import time
import logging
import requests
import json
from urllib.parse import urljoin, urlparse
from flask import Flask, request, jsonify, render_template_string, send_from_directory
from bs4 import BeautifulSoup
from datetime import datetime
from typing import Dict, Optional

# Groq AI client импорт
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logging.warning("Groq client not installed. Install with: pip install groq")

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
GROQ_API_KEY         = os.getenv("GROQ_API_KEY")
AUTO_CRAWL_ON_START  = os.getenv("AUTO_CRAWL_ON_START", "true").lower() == "true"

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY) if (GROQ_API_KEY and GROQ_AVAILABLE) else None

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


# —— AI Assistant Functions —— #
def get_ai_response(user_message: str, conversation_id: int, context_data: list = None):
    """Enhanced AI response with Groq's Llama models for better Mongolian support"""
    
    if not client:
        return "🔑 Groq API түлхүүр тохируулагдаагүй байна. Системийн админтай холбогдоно уу."
    
    # Handle empty message
    if not user_message or not user_message.strip():
        return "📝 Таны мессеж хоосон байна. Асуулт эсвэл хайж байгаа зүйлээ бичээд илгээнэ үү. Би танд туслахад бэлэн байна! 😊"
    
    # Get conversation history
    history = conversation_memory.get(conversation_id, [])
    
    # Build context from crawled data if available
    context = ""
    if crawled_data:
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
            
            # Add statistics about available data
            # context += f"\n\n📊 Нийт {len(crawled_data)} хуудаснаас {len(search_results)} хамгийн холбогдохтой хуудас олдлоо."
        else:
            # context = f"📋 {len(crawled_data)} хуудаснаас хайлт хийсэн боловч тухайн сэдвээр шууд тохирох мэдээлэл олдсонгүй. Ерөнхий мэдлэгээрээ хариулж байна."
            pass  # No context found, will use general knowledge
    
    # Build system message with context
    system_content = """Та онлайн дэлгүүрийн AI туслах бот юм. Хэрэглэгчдэд бүтээгдэхүүний мэдээлэл, үнэ, дэлгэрэнгүй мэдээлэл хайж олоход тусалдаг.
    Хэрэглэгчтэй монгол хэлээр найрсаг, тусламжтай ярилцаарай. Та өөрийн мэдэх мэдээллээр дамжуулан бүхий л асуултад хариулах чадвартай.
    
    ЭНГИЙН МЭНДЧИЛГЭЭНИЙ ТУХАЙ:
    Хэрэв хэрэглэгч энгийн мэндчилгээ хийж байвал (жишээ: "сайн байна уу", "сайн уу", "мэнд", "hello", "hi", "сайн уу байна", "hey", "sn bnu", "snu" гэх мэт), дараах байдлаар хариулаарай:
    
    "Сайн байна уу! Танд хэрхэн туслах вэ?
    
    Би дараах зүйлсээр танд туслаж чадна:
    • 🔍 Бүтээгдэхүүн хайх болон олох
    • 💰 Үнийн мэдээлэл өгөх  
    • 📝 Бүтээгдэхүүний дэлгэрэнгүй мэдээлэл
    • 🛒 Худалдан авалтын зөвлөгөө
    • 📞 Холбоо барих мэдээлэл
    • ❓ Бүхий л төрлийн асуултад хариулах
    
    Хайж байгаа бүтээгдэхүүнээ хэлээрэй эсвэл асуултаа чөлөөтэй асуугаарай!"
    
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
    
    # Build conversation messages for Groq
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
            model="llama-3.3-70b-versatile",  # Groq's best model for Mongolian
            messages=messages,
            max_tokens=600,
            temperature=0.7,
            top_p=0.9
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
        logging.error(f"Groq API алдаа: {e}")
        return f"🔧 Өөрийн системээс хариулт авахад саад гарлаа. Та дараах зүйлсийг туршиж үзнэ үү:\n\n• Асуултаа арай өөрөөр томъёолж илгээнэ үү\n• Илүү тодорхой, тусгай нөхцөлөөр асуугаарай\n• Хайж байгаа зүйлийнхээ нэрийг өөрөөр бичиж үзнэ үү\n• Хэдэн секундын дараа дахин оролдоно уу\n\nБи танд туслахад бэлэн байна! 💪"

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
    """Enhanced webhook with AI integration using RAG system"""
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
    
    # Skip empty messages
    if not text:
        logging.warning(f"Skipping empty message in conversation {conv_id}")
        return jsonify({"status": "skipped", "reason": "empty_message"}), 200
    
    # Always use AI with RAG system - no human escalation
    ai_response = get_ai_response(text, conv_id, crawled_data)
    
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
    return jsonify({"conversation_id": conv_id, "memory": memory, "system": "pure_rag_no_escalation"})

@app.route("/api/conversation/<int:conv_id>/clear", methods=["POST"])
def clear_conversation_memory(conv_id):
    """Clear conversation memory"""
    if conv_id in conversation_memory:
        del conversation_memory[conv_id]
    return jsonify({"status": "cleared", "conversation_id": conv_id, "system": "pure_rag_no_escalation"})

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
        "system_type": "pure_rag_no_escalation",
        "timestamp": datetime.now().isoformat(),
        "crawl_status": crawl_status,
        "crawled_pages": len(crawled_data),
        "active_conversations": len(conversation_memory),
        "config": {
            "root_url": ROOT_URL,
            "auto_crawl_enabled": AUTO_CRAWL_ON_START,
            "groq_configured": client is not None,
            "chatwoot_configured": bool(CHATWOOT_API_KEY and ACCOUNT_ID),
            "human_escalation": False
        }
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)