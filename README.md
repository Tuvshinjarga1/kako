# Kako.mn AI Chatbot - Google Gemini Vision + Зураг харьцуулах систем

Энэ бол онлайн дэлгүүрийн AI туслах бот юм. Google Gemini 2.5 Flash ашиглан монгол хэлийг сайн ойлгодог, зургийг танидаг, хурдан хариулт өгдөг. **Шинэ боломж**: Crawl хийсэн зургуудтай хэрэглэгчийн илгээсэн зургийг харьцуулж ижил төстэй бүтээгдэхүүн олдог!

## 🚀 Онцлог шинж чанарууд

- **Google Gemini 2.5 Flash**: Google-ийн хамгийн сүүлийн үеийн multimodal загвар ашиглан монгол хэлийг маш сайн ойлгодог
- **🖼️ Зураг танин мэдэх**: Хэрэглэгчээс ирсэн зургийг танин бүтээгдэхүүний мэдээлэл өгдөг
- **🔄 Зураг харьцуулах**: Хэрэглэгчийн илгээсэн зурагтай ижил төстэй бүтээгдэхүүнийг crawl хийсэн зургуудаас олдог
- **Multimodal**: Текст болон зураг хоёуланг нэг зэрэг боловсруулдаг
- **Вебсайт crawling**: Автоматаар вебсайтын агуулга болон зургуудыг татаж хадгалдаг
- **Chatwoot интеграци**: Chatwoot widget-тэй холбогдоно
- **RAG систем**: Crawl хийсэн мэдээллээс хамгийн холбогдохтой агуулгыг олж AI-д өгдөг
- **Image Database**: Зургийн мэдээллийн сан үүсгэж, өнгөний histogram ашиглан харьцуулдаг

## 📋 Шаардлага

- Python 3.9+
- Google Gemini API түлхүүр
- Docker (сонголттой)

## 🔧 Суулгах заавар

### 1. Repository clone хийх

```bash
git clone <repository-url>
cd kako
```

### 2. Virtual environment үүсгэх

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Dependencies суулгах

```bash
pip install -r requirements.txt
```

### 4. Google Gemini API түлхүүр авах

1. [Google AI Studio](https://aistudio.google.com/app/apikey) руу орно уу
2. "Create API key" дээр дарж шинэ API key үүсгэнэ үү
3. API key-г аюулгүй газар хадгална уу

### 5. Environment variables тохируулах

`.env` файл үүсгэж дараах мэдээллийг оруулна уу:

```bash
# Онлайн дэлгүүрийн тохиргоо
ROOT_URL=https://kako.mn/
MAX_CRAWL_PAGES=500
AUTO_CRAWL_ON_START=true
DELAY_SEC=0.5

# Google Gemini API тохиргоо (зураг таних чадвартай)
GEMINI_API_KEY=your_google_gemini_api_key

# Chatwoot тохиргоо
CHATWOOT_API_KEY=your_chatwoot_api_key
ACCOUNT_ID=your_account_id
CHATWOOT_BASE_URL=https://app.chatwoot.com/
```

### 6. Аппликейшн ажиллуулах

```bash
python main.py
```

Эсвэл Docker ашиглан:

```bash
docker build -t kako-chatbot .
docker run -p 8000:8000 --env-file .env kako-chatbot
```

## 🔥 Шинэ боломжууд

### 📸 Зураг танин мэдэх (Google Gemini Vision)

Хэрэглэгч Chatwoot widget дээр зураг илгээхэд, бот автоматаар:

- Зурагт харагдаж байгаа бүтээгдэхүүнийг тодорхойлно
- Загвар, өнгө, материалын мэдээлэл өгнө
- Үнэ болон худалдан авах мэдээлэл хайж олно
- Ижил төстэй бүтээгдэхүүн санал болгоно
- Object detection болон segmentation хийдэг

### 🔄 Зураг харьцуулах систем (ШИНЭ!)

**Crawl хийсэн зургуудтай харьцуулах**:

- Хэрэглэгчийн илгээсэн зурагтай ижил төстэй бүтээгдэхүүнийг автоматаар олдог
- Өнгөний histogram ашиглан зургийн ижлээ тооцдог
- Similarity threshold: 20% (тохируулах боломжтой)
- Хамгийн ойролцоо 3 зургийг олж AI-д мэдээлэл өгдөг

**Ажиллах зарчим**:

1. Crawl хийхэд бүх зургийг татаж `crawled_images/` folder-т хадгална
2. Зураг тус бүрээс өнгөний histogram гаргана (RGB каналууд)
3. Хэрэглэгч зураг илгээхэд, Chi-squared distance ашиглан харьцуулна
4. Хамгийн ижил төстэй 3 зургийн хуудасны мэдээллийг AI-д өгнө

### 🧠 RAG (Retrieval-Augmented Generation) систем

- Crawl хийсэн мэдээллээс хамгийн холбогдохтой агуулгыг автоматаар хайдаг
- Semantic search ашиглан монгол хэлээр хайлт хийдэг
- Контекст мэдээллийг AI-д өгч илүү нарийвчлалтай хариулт авдаг

### 🔧 API Endpoints

#### Системийн төлөв шалгах

```bash
GET /health
```

#### Хайлт хийх

```bash
POST /api/search
{
  "query": "хайх үг",
  "max_results": 5
}
```

#### Crawl-ийн төлөв шалгах

```bash
GET /api/crawl-status
```

#### Шинэ crawl эхлүүлэх

```bash
POST /api/force-crawl
```

#### Crawl хийсэн зургуудыг харах

```bash
GET /api/crawled-images?limit=20
```

#### Зураг харьцуулах (API)

```bash
POST /api/similar-images
Content-Type: multipart/form-data

# Form data:
image: [image file]
threshold: 0.3  # optional, default 0.3
```

#### Зургийн мэдээллийн санг цэвэрлэх

```bash
POST /api/clear-images
```

#### Зураг үзэх

```bash
GET /images/[filename]
```

## 🛠️ Тохиргоо

### GEMINI_API_KEY

Google Gemini API түлхүүр (ai.google.dev-ээс авах)

### MAX_CRAWL_PAGES

Crawl хийх хуудасны тоо (default: 500)

### AUTO_CRAWL_ON_START

Аппликейшн эхлэхэд автоматаар crawl хийх эсэх (default: true)

## 🐛 Алдаа засах

### Gemini API алдаа

```bash
# API key шалгах
echo $GEMINI_API_KEY

# API quota шалгах (Google AI Studio дээр)
```

### Docker алдаа

```bash
# Container logs шалгах
docker logs kako-chatbot

# Container дотор орох
docker exec -it kako-chatbot bash
```

## 📝 Техникийн дэлгэрэнгүй

### Google Gemini-ийн давуу талууд

- **Multimodal**: Зураг болон текстийг нэг зэрэг ойлгодог
- **Хурдан**: OpenAI-оос илүү хурдан response
- **Үнэ хямд**: Token үнэ илүү хямд
- **Монгол хэл**: Монгол хэлийг сайн дэмждэг
- **Vision чадвар**: Зураг танин мэдэх чадвар өндөр

### Дэмжигдэх зургийн форматууд

- PNG - `image/png`
- JPEG - `image/jpeg`
- WebP - `image/webp`
- HEIC - `image/heic`
- HEIF - `image/heif`

### Code structure

```
├── main.py              # Үндсэн Flask аппликейшн
├── requirements.txt     # Python dependencies
├── Dockerfile          # Docker тохиргоо
├── env.example         # Environment variables жишээ
└── README.md           # Энэ файл
```

### Шинэ функц нэмэх

1. `main.py` файлд функц нэмнэ үү
2. Route нэмэх бол `@app.route` decorator ашиглана уу
3. AI logic өөрчлөх бол `get_ai_response` функцийг засна уу
4. Зураг боловсруулах бол `process_chatwoot_attachment` функцийг засна уу

## 🖼️ Зураг боловсруулах

### Жишээ ашиглалт

1. Chatwoot widget дээр зураг upload хийх
2. "Энэ юу вэ?" гэж асуух эсвэл зураг л илгээх
3. Gemini AI автоматаар зургийг шинжилж хариулт өгнө

### Зургийн хязгаарлалтууд

- Нийт request хэмжээ: 20MB
- Зургийн тоо: Request тутамд 3,600 хүртэл
- Зургийн хэмжээ: Автоматаар оптимальчлагдана

## 🚀 Performance

- **Token тооцоолол**: 384px хүртэл зураг 258 token
- **Response хугацаа**: Ихэнхдээ 2-5 секунд
- **Crawl хурд**: Секундэд 2 хуудас (DELAY_SEC тохиргооноос хамаарна)

## 🤝 Хувь нэмэр оруулах

1. Fork хийнэ үү
2. Feature branch үүсгэнэ үү
3. Changes хийнэ үү
4. Test хийнэ үү
5. Pull request илгээнэ үү

## 📞 Тусламж

- Асуудалтай бол GitHub Issues ашиглана уу
- Google Gemini API: [ai.google.dev](https://ai.google.dev)
- Chatwoot setup: [chatwoot.com](https://chatwoot.com)
