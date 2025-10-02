from flask import Flask, request, jsonify, render_template
import os
import PyPDF2
import joblib
from modelo import EnhancedEmailClassifier
from huggingface_hub import InferenceClient

app = Flask(__name__)

# Configurações
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf'}
MODEL_PATH = 'email_classifier_model.pkl'  # Caminho para o modelo de classificação salvo
TOKEN = 'Seu-Hugging-Face-API-token'  # Substitua pelo seu token Hugging Face <<<--------------------CONFIGURE AQUI-------------------
MODEL_IA = 'deepseek-ai/DeepSeek-V3-0324'  # Modelo de geração de texto, pode ser ajustado
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# geração de texto API config


# Carregar modelos de classificação e geração de texto
try:
    # modelo para classificação de emails do pkl salvo
    classifier = joblib.load(MODEL_PATH)
    print(f"Modelo de classificação carregado: {classifier.model_type}")
    
    # Modelo para geração de respostas
    if TOKEN == 'sua_token_api_hugging_faces':
        generator = "Por favor, configure seu token Hugging Face na variável TOKEN."
    else:
        generator = InferenceClient(model=MODEL_IA, token=TOKEN)
        print("Cliente de geração de texto carregado.")

except Exception as e:
    print(f"Erro ao carregar modelos: {e}")
    classifier = None
    generator = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    """Extrai texto de arquivos PDF"""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
    except Exception as e:
        return f"Erro ao extrair texto do PDF: {e}"

def classify_email(text):
    """Classifica o email como Produtivo ou Improdutivo"""
    if not classifier:
        return "Modelo de classificação não disponível"
    
    # Classificar

    result, confidence = classifier.predict(text)
    
    # Retornar a categoria com maior score
    return result

def generate_response(email_text, category):
    """Gera uma resposta automática baseada na categoria"""
    if not generator:
        return "Cliente de geração não disponível"
    
    # Definir prompts baseados na categoria
    if category.lower() == "produtivo":
        message = {'role': 'user', 'content': f"Email recebido: {email_text}\n\nResposta profissional e útil:"}
    else:
        message = {'role': 'user', 'content':f"Email recebido: {email_text}\n\nResposta cordial e agradecendo:"}
    
    # Gerar resposta
    response = generator.chat_completion(messages=[message], temperature=0.7, top_p=0.9)
    
    # Extrair apenas a parte gerada
    generated_text = response['choices'][0]['message']['content']
    response_text = generated_text.replace(message['content'],"").strip()
    
    return response_text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classificar', methods=['POST'])
def classify_email_route():
    """Endpoint para classificar email e gerar resposta"""
    try:
        # Verificar se foi enviado texto ou arquivo
        if 'email_text' in request.form and request.form['email_text']:
            email_text = request.form['email_text']
        elif 'email_file' in request.files:
            file = request.files['email_file']
            if file.filename == '':
                return jsonify({'error': 'Nenhum arquivo selecionado'})
            
            if file and allowed_file(file.filename):
                # Salvar arquivo temporariamente
                filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filename)
                
                # Extrair texto do arquivo
                if filename.endswith('.pdf'):
                    email_text = extract_text_from_pdf(filename)
                else:  # .txt
                    with open(filename, 'r', encoding='utf-8') as f:
                        email_text = f.read()
                
                # Limpar arquivo temporário
                os.remove(filename)
            else:
                return jsonify({'error': 'Tipo de arquivo não permitido'})
        else:
            return jsonify({'error': 'Nenhum conteúdo de email fornecido'})
        
        # Classificar email
        category = classify_email(email_text)
        # Gerar resposta
        if generator == "Por favor, configure seu token Hugging Face na variável TOKEN.":
            return jsonify({'error': generator})
        response = generate_response(email_text, category)
        
        return jsonify({
            'category': category,
            'response': response,
            'original_text': email_text
        })
    
    except Exception as e:
        return jsonify({'error': f'Erro no processamento: {str(e)}'})

if __name__ == '__main__':
    # Criar pasta de uploads se não existir
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    app.run()