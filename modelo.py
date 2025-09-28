import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import joblib
import re
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime
nltk.download('stopwords')

class EnhancedEmailClassifier:
    def __init__(self, model_type='auto'):
        self.model_type = model_type
        self.pipeline = None
        self.vectorizer = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9
        )
        self.best_model = None
        self.model_comparison = None
        self._initialize_model()
        
    def _initialize_model(self):
        """Inicializa o modelo baseado no tipo selecionado"""
        if self.model_type == 'naive_bayes':
            classifier = MultinomialNB(alpha=0.1)
        elif self.model_type == 'logistic_regression':
            classifier = LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42
            )
        elif self.model_type == 'svm':
            classifier = SVC(
                kernel='linear',
                C=1.0,
                probability=True,
                random_state=42
            )
        elif self.model_type == 'random_forest':
            classifier = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10
            )
        elif self.model_type == 'gradient_boosting':
            classifier = GradientBoostingClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=5
            )
        else:
            # Para 'auto', usaremos um placeholder
            classifier = MultinomialNB()
            
        self.pipeline = Pipeline([
            ('tfidf', self.vectorizer),
            ('classifier', classifier)
        ])
    
    def preprocess_text(self, text):
        """Pré-processa o texto do email"""
        stopwords_pt = stopwords.words("portuguese")
        stemmer = RSLPStemmer()
        # Remover cabeçalhos de email
        text = re.sub(r'De:.*?\n', '', text)
        text = re.sub(r'Para:.*?\n', '', text)
        text = re.sub(r'Assunto:.*?\n', '', text)
        
        # Converter para minúsculas
        text = text.lower()
        
        # Remover caracteres especiais e números
        text = re.sub(r'[^a-zA-Záéíóúâêîôûãõç\s]', ' ', text)
        
        # Remover espaços extras
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remover stopwords
        text = ' '.join([word for word in text.split() if word not in stopwords_pt])

        # lematização
        text = ' '.join([stemmer.stem(word) for word in text.split()])
        
        return text
    
    def load_dataset(self, filepath):
        """Carrega o dataset de emails"""
        df = pd.read_csv(filepath)
        return df
    
    def prepare_data(self, df):
        """Prepara os dados para treinamento"""
        # Pré-processar textos
        X = [self.preprocess_text(text) for text in df['email_text']]
        
        # Converter categorias para numérico
        y = df['category'].map({'Produtivo': 1, 'Improdutivo': 0}).values
        
        return X, y
    
    def find_best_model(self, X, y, test_size=0.2, cv=5, models_to_test='all'):
        """
        Encontra o melhor modelo entre os algoritmos disponíveis
        
        Args:
            X: Lista de textos de emails
            y: Lista de labels
            test_size: Proporção do conjunto de teste
            cv: Número de folds para validação cruzada
            models_to_test: Lista de modelos a testar ou 'all' para testar todos
        
        Returns:
            Dicionário com resultados da comparação e o melhor modelo
        """
        # Definir modelos a testar
        if models_to_test == 'all':
            models_to_test = [
                'naive_bayes',
                'logistic_regression', 
                'svm',
                'random_forest',
                'gradient_boosting'
            ]
        
        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        results = []
        best_score = 0
        best_model_name = None
        best_pipeline = None
        
        print("="*60)
        print("COMPARAÇÃO DE MODELOS")
        print("="*60)
        
        for model_name in models_to_test:
            print(f"\n Treinando {model_name}...")
            start_time = time.time()
            
            # Criar e treinar modelo
            temp_classifier = EnhancedEmailClassifier(model_name)
            temp_classifier.pipeline.fit(X_train, y_train)
            
            # Fazer predições
            y_pred = temp_classifier.pipeline.predict(X_test)
            
            # Calcular métricas
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Validação cruzada
            cv_scores = cross_val_score(
                temp_classifier.pipeline, X_train, y_train, 
                cv=cv, scoring='accuracy'
            )
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            training_time = time.time() - start_time
            
            # Armazenar resultados
            result = {
                'model': model_name,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'training_time': training_time
            }
            results.append(result)
            
            print(f"✅ {model_name}:")
            print(f"   Acurácia: {accuracy:.4f}")
            print(f"   F1-Score: {f1:.4f}")
            print(f"   Precisão: {precision:.4f}")
            print(f"   Recall: {recall:.4f}")
            print(f"   Validação Cruzada: {cv_mean:.4f} (±{cv_std:.4f})")
            print(f"   Tempo de Treino: {training_time:.2f}s")
            
            # Verificar se é o melhor modelo
            if f1 > best_score:  # Usando F1-Score como critério principal
                best_score = f1
                best_model_name = model_name
                best_pipeline = temp_classifier.pipeline
        
        # Criar DataFrame com resultados
        results_df = pd.DataFrame(results)
        
        # Ordenar por F1-Score (critério principal)
        results_df = results_df.sort_values('f1_score', ascending=False)
        
        print("\n" + "="*60)
        print("MELHOR MODELO ENCONTRADO")
        print("="*60)
        best_result = results_df.iloc[0]
        print(f"Modelo: {best_result['model']}")
        print(f"F1-Score: {best_result['f1_score']:.4f}")
        print(f"Acurácia: {best_result['accuracy']:.4f}")
        print(f"Precisão: {best_result['precision']:.4f}")
        print(f"Recall: {best_result['recall']:.4f}")
        
        # Salvar informações
        self.best_model = best_pipeline
        self.model_comparison = results_df
        self.model_type = best_model_name
        
        return {
            'best_model': best_pipeline,
            'best_model_name': best_model_name,
            'comparison_results': results_df,
            'best_metrics': best_result.to_dict()
        }
    
    def train_with_best_model(self, filepath, test_size=0.2, cv=5):
        """Treina usando o melhor modelo encontrado"""
        # Carregar dados
        df = self.load_dataset(filepath)
        X, y = self.prepare_data(df)
        
        # Encontrar melhor modelo
        comparison_result = self.find_best_model(X, y, test_size, cv)
        
        # Usar o melhor modelo encontrado
        self.pipeline = comparison_result['best_model']
        self.model_type = comparison_result['best_model_name']
        
        # Fazer avaliação final no conjunto de teste
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        y_pred = self.pipeline.predict(X_test)
        
        print("\n" + "="*60)
        print("AVALIAÇÃO FINAL DO MELHOR MODELO")
        print("="*60)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"Acurácia: {accuracy:.4f}")
        print(f"Precisão: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Matriz de confusão
        self.plot_confusion_matrix(y_test, y_pred)
        
        # Plotar comparação de modelos
        self.plot_model_comparison()
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'best_model': self.model_type,
            'comparison': self.model_comparison
        }
    
    def plot_model_comparison(self):
        """Plota gráfico comparativo dos modelos"""
        if self.model_comparison is None:
            print("Nenhuma comparação de modelos disponível.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Comparação de Modelos de Classificação', fontsize=16)
        
        # Gráfico 1: Acurácia e F1-Score
        models = self.model_comparison['model']
        accuracy = self.model_comparison['accuracy']
        f1_scores = self.model_comparison['f1_score']
        
        x = np.arange(len(models))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, accuracy, width, label='Acurácia', alpha=0.7)
        axes[0, 0].bar(x + width/2, f1_scores, width, label='F1-Score', alpha=0.7)
        axes[0, 0].set_xlabel('Modelos')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Acurácia vs F1-Score')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(models, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Gráfico 2: Precisão e Recall
        precision = self.model_comparison['precision']
        recall = self.model_comparison['recall']
        
        axes[0, 1].bar(x - width/2, precision, width, label='Precisão', alpha=0.7)
        axes[0, 1].bar(x + width/2, recall, width, label='Recall', alpha=0.7)
        axes[0, 1].set_xlabel('Modelos')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_title('Precisão vs Recall')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(models, rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Gráfico 3: Validação Cruzada
        cv_mean = self.model_comparison['cv_mean']
        cv_std = self.model_comparison['cv_std']
        
        axes[1, 0].bar(x, cv_mean, yerr=cv_std, capsize=5, alpha=0.7)
        axes[1, 0].set_xlabel('Modelos')
        axes[1, 0].set_ylabel('Acurácia Média')
        axes[1, 0].set_title('Validação Cruzada (5-fold)')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(models, rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Gráfico 4: Tempo de Treinamento
        training_time = self.model_comparison['training_time']
        
        axes[1, 1].bar(x, training_time, alpha=0.7)
        axes[1, 1].set_xlabel('Modelos')
        axes[1, 1].set_ylabel('Tempo (segundos)')
        axes[1, 1].set_title('Tempo de Treinamento')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(models, rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plota a matriz de confusão"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Improdutivo', 'Produtivo'],
                   yticklabels=['Improdutivo', 'Produtivo'])
        plt.title('Matriz de Confusão - Melhor Modelo')
        plt.ylabel('Verdadeiro')
        plt.xlabel('Predito')
        plt.show()
    
    def predict(self, email_text):
        """Faz a predição para um único email"""
        if self.pipeline is None:
            raise ValueError("Modelo não treinado. Execute train_with_best_model() primeiro.")
        
        processed_text = self.preprocess_text(email_text)
        prediction = self.pipeline.predict([processed_text])[0]
        probability = self.pipeline.predict_proba([processed_text])[0]
        
        category = "produtivo" if prediction == 1 else "improdutivo"
        confidence = probability[prediction]
        
        return category, confidence
    
    def save_model(self, filepath):
        """Salva o modelo treinado"""
        
        joblib.dump(self, filepath)
        print(f"Modelo salvo como {filepath}")

# Função de exemplo de uso
def example_usage(dataset_path):
    """Exemplo de como usar a função de encontrar o melhor modelo"""
    
    # Carregar dados
    classifier = EnhancedEmailClassifier()
    df = classifier.load_dataset(dataset_path)
    X, y = classifier.prepare_data(df)
    
    print("Dataset Carregado:")
    print(f"Total de emails: {len(X)}")
    print(f"Distribuição: {np.unique(y, return_counts=True)}")
    
    # Encontrar o melhor modelo
    classifier.train_with_best_model(dataset_path)
    
    # Testar com exemplos
    test_emails = [
        "Preciso de suporte técnico urgente para problema no sistema de login",
        "Obrigado pelo excelente trabalho da equipe de desenvolvimento",
        "Erro 404 ao acessar o painel administrativo, preciso de ajuda",
        "Desejo um feliz natal e um próspero ano novo para todos"
    ]
    
    print("\n" + "="*60)
    print("TESTE COM EXEMPLOS")
    print("="*60)
    
    for i, email in enumerate(test_emails, 1):
        category, confidence = classifier.predict(email)
        print(f"Exemplo {i}:")
        print(f"Email: {email[:80]}...")
        print(f"Predição: {category} (confiança: {confidence:.2%})")
        print("-" * 40)
    
    # Salvar modelo
    classifier.save_model('email_classifier_model.pkl')
    
    return classifier

if __name__ == "__main__":
    # Executar exemplo
    classifier = example_usage('email_dataset.csv')