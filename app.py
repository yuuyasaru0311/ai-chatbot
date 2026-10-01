import os
import io
import base64
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

app = Flask(__name__)
client = genai.Client(api_key=api_key)

# グローバル変数：一度だけ読み込んでキャッシュするナレッジとベクトル
cached_paragraphs = []
cached_vectors = []
_is_initialized = False

def init_knowledge_base():
    """必要になったタイミングで一度だけknowledge.txtを読み込み、ベクトル化する（起動時クラッシュ防止）"""
    global cached_paragraphs, cached_vectors, _is_initialized
    if _is_initialized:
        return
    _is_initialized = True

    file_path = "knowledge.txt"
    if not os.path.exists(file_path):
        print("Warning: knowledge.txt が見つかりません。")
        return

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            cached_paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

        if cached_paragraphs:
            print(f"ナレッジを {len(cached_paragraphs)} 件読み込み中...")
            vectors = []
            for p in cached_paragraphs:
                res = client.models.embed_content(
                    model="text-embedding-2",
                    contents=p
                )
                vector = np.array(res.embeddings[0].values)
                vectors.append(vector)
            cached_vectors = vectors
            print("ナレッジの事前ベクトル化が完了しました！")
    except Exception as e:
        print(f"初期ベクトル化エラー（通常のチャットは利用できます）: {e}")

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search_relevant_knowledge_smart(query):
    """事前計算されたベクトルと照合する"""
    # 初回リクエスト時にここで安全に初期化を行う
    init_knowledge_base()
    
    if not cached_paragraphs or not cached_vectors or not query:
        return ""
    
    try:
        query_res = client.models.embed_content(
            model="text-embedding-2",
            contents=query
        )
        query_vector = np.array(query_res.embeddings[0].values)

        best_paragraph = ""
        highest_score = -1.0
        
        for p, p_vector in zip(cached_paragraphs, cached_vectors):
            score = cosine_similarity(query_vector, p_vector)
            if score > highest_score:
                highest_score = score
                best_paragraph = p
        
        if highest_score > 0.45:
            return best_paragraph
            
    except Exception as e:
        print(f"Embedding Search Error: {e}")
        
    return ""

sys_instruct = (
    "あなたは、とても親切で温かいAIアシスタントです。"
    "【参考情報】が提示されている場合はそれを最優先に参考にして回答し、"
    "【参考情報】がない場合は、あなたが持っている一般的な知識を使って分かりやすく答えてください。"
    "画像が送られてきた場合は、それがスマホの画面や仕事に関することだけでなく、建物、人物、景色、物などどんな写真であっても、何が写っているかを優しく丁寧に教えてあげてください。"
    "「大丈夫ですよ」「〜ですね」といった安心感を与える温かい言葉を交えながら、3文以内で短く優しく答えてください。"
)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat_endpoint():
    data = request.get_json() or {}
    user_message = data.get("message", "")
    image_b64 = data.get("image", None)

    if not user_message and not image_b64:
        return jsonify({"response": "メッセージまたは写真を入力してくださいね。"})

    try:
        context = search_relevant_knowledge_smart(user_message)
        
        if context:
            prompt_text = f"（【参考情報】\n{context}）\n\n{user_message}"
        else:
            prompt_text = user_message if user_message else "この画像について分かりやすく教えてください。"

        contents = []
        if image_b64:
            if "," in image_b64:
                image_b64 = image_b64.split(",")[1]
            image_bytes = base64.b64decode(image_b64)
            pil_img = Image.open(io.BytesIO(image_bytes))
            contents.append(pil_img)
        
        contents.append(prompt_text)

        user_chat = client.chats.create(
            model="gemini-3.8-flash",
            config={"system_instruction": sys_instruct}
        )

        response = user_chat.send_message(contents)
        return jsonify({"response": response.text})

    except Exception as e:
        error_msg = str(e)
        if "503" in error_msg or "UNAVAILABLE" in error_msg:
            friendly_message = "申し訳ありません。ただいま少し込み合っているようです。大丈夫ですよ、少し時間を置いてからもう一度お試しくださいね。"
        elif "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            friendly_message = "少しお話しするのが早すぎたようです。1〜2分ほどゆっくりお休みしてから、また声をかけてくださいね。"
        else:
            friendly_message = "うまくお返事ができませんでした。もう一度短くお話ししていただけますか？一緒にやっていきましょうね。"
            
        print(f"API Error Log: {e}")
        return jsonify({"response": friendly_message})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)