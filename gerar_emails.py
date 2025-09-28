import pandas as pd
import random
from faker import Faker

# Inicializar Faker para geração de dados realistas
fake = Faker('pt_BR')

class EmailDatasetGenerator:
    def __init__(self):
        self.productive_keywords = [
            'suporte', 'problema', 'erro', 'ajuda', 'solicitação', 'urgente',
            'configuração', 'senha', 'acesso', 'sistema', 'tecnico', 'suporte técnico',
            'bug', 'falha', 'atualização', 'migração', 'backup', 'restauração',
            'integracao', 'api', 'relatorio', 'dashboard', 'login', 'cadastro',
            'pagamento', 'fatura', 'contrato', 'proposta', 'orcamento', 'venda'
        ]
        
        self.unproductive_keywords = [
            'obrigado', 'agradeço', 'parabéns', 'felicitações', 'feliz', 'natal',
            'ano novo', 'final de semana', 'ótimo', 'excelente', 'bom trabalho',
            'agradecimento', 'cumprimentos', 'saudações', 'abraço', 'atenciosamente',
            'cordialmente', 'grato', 'reconhecimento', 'elogio', 'comemoração',
            'festividade', 'feriado', 'férias', 'descanso', 'comemoração'
        ]
        
        self.productive_templates = [
            "Preciso de suporte técnico para {problema} no sistema {sistema}.",
            "Encontrei um erro ao {acao}. Podem me ajudar?",
            "Solicito assistência para configurar o {recurso}.",
            "Problema com {funcionalidade}: {descricao_problema}",
            "Como faço para {acao} no sistema?",
            "Erro {codigo_erro} ao tentar acessar {modulo}.",
            "Necessito de suporte urgente para {situacao}.",
            "Dúvida sobre como usar o {recurso}.",
            "Solicitação de {servico} para o projeto {projeto}.",
            "Problema de performance no {componente}.",
            "Não consigo {acao} devido a {problema}.",
            "Preciso de ajuda com a integração do {sistema_externo}.",
            "Relatório de bug: {descricao_bug}",
            "Solicito atualização do {sistema} para a versão {versao}.",
            "Problema com o acesso ao {recurso}.",
            "Como resetar minha senha de acesso?",
            "Erro ao tentar {acao} no módulo {modulo}.",
            "Solicito informações sobre {topico}.",
            "Problema com a funcionalidade de {feature}.",
            "Necessito de treinamento para usar o {sistema}."
        ]
        
        self.unproductive_templates = [
            "Agradeço pelo {servico} prestado, foi muito útil.",
            "Parabéns pelo excelente trabalho no {projeto}.",
            "Desejo um {evento} maravilhoso para toda a equipe.",
            "Muito obrigado pelo atendimento {qualidade}.",
            "Feliz {feriado} para todos!",
            "Aprecio muito a dedicação da equipe de {departamento}.",
            "Excelente trabalho no {projeto_recente}.",
            "Agradecimentos pelo bom atendimento {situacao}.",
            "Que tenham um {periodo} abençoado!",
            "Reconhecimento pelo esforço da equipe.",
            "Cumprimentos pela qualidade do {servico}.",
            "Agradeço pela paciência e atenção.",
            "Parabéns pelo aniversário da empresa!",
            "Desejo sucesso no {evento_futuro}.",
            "Agradeço pela colaboração no {projeto}.",
            "Feliz {data_comemorativa}!",
            "Muito obrigado pela parceria.",
            "Cumprimentos pela equipe competente.",
            "Agradeço pelo suporte sempre eficiente.",
            "Desejo um final de semana tranquilo."
        ]
        
        self.systems = ['SAP', 'Salesforce', 'Oracle', 'Microsoft Dynamics', ' sistema interno', 'plataforma web', 'app mobile']
        self.problems = ['login', 'acesso', 'performance', 'configuração', 'integração', 'relatórios']
        self.actions = ['acessar', 'configurar', 'atualizar', 'migrar', 'exportar', 'importar']
    
    def generate_productive_email(self):
        """Gera um email produtivo"""
        template = random.choice(self.productive_templates)
        
        # Preencher placeholders
        email_text = template.format(
            problema=random.choice(self.problems),
            sistema=random.choice(self.systems),
            acao=random.choice(self.actions),
            recurso=random.choice(['módulo de relatórios', 'dashboard', 'painel administrativo']),
            funcionalidade=random.choice(['login', 'exportação', 'importação', 'relatórios']),
            descricao_problema=random.choice([
                'não carrega corretamente',
                'apresenta erro inesperado',
                'está muito lento',
                'não salva as alterações'
            ]),
            codigo_erro=f"ERRO_{random.randint(100, 999)}",
            modulo=random.choice(['usuários', 'relatórios', 'configurações']),
            situacao=random.choice(['critica', 'urgente', 'importante']),
            servico=random.choice(['suporte', 'treinamento', 'consultoria']),
            projeto=f"{fake.word().capitalize()}",
            componente=random.choice(['banco de dados', 'servidor', 'aplicação']),
            sistema_externo=random.choice(['API externa', 'sistema legado', 'ERP']),
            descricao_bug=random.choice([
                'ao clicar no botão salvar, o sistema fecha',
                'os valores não são calculados corretamente',
                'a interface não responde',
                'os dados não são exibidos'
            ]),
            versao=f"{random.randint(1, 5)}.{random.randint(0, 9)}",
            topico=random.choice(['licenciamento', 'atualizações', 'novas funcionalidades']),
            feature=random.choice(['pesquisa', 'filtro', 'exportação'])
        )
        
        # Adicionar saudação e assinatura
        full_email = f"""
De: {fake.email()}
Para: suporte@empresa.com
Assunto: {random.choice(['Solicitação de Suporte', 'Problema Técnico', 'Dúvida sobre Sistema', 'Relatório de Erro'])}

Prezados,

{email_text}

Atenciosamente,
{fake.name()}
{fake.job()}
        """.strip()
        
        return full_email
    
    def generate_unproductive_email(self):
        """Gera um email improdutivo"""
        template = random.choice(self.unproductive_templates)
        
        # Preencher placeholders
        email_text = template.format(
            servico=random.choice(['suporte', 'atendimento', 'trabalho']),
            projeto=random.choice(['projeto recente', 'última demanda', 'tarefa concluída']),
            evento=random.choice(['final de semana', 'feriado', 'natal', 'ano novo']),
            qualidade=random.choice(['rápido', 'eficiente', 'prestativo']),
            feriado=random.choice(['Natal', 'Páscoa', 'Ano Novo', 'Carnaval']),
            departamento=random.choice(['suporte', 'tecnologia', 'vendas']),
            projeto_recente=random.choice(['implementação do sistema', 'migração de dados']),
            situacao=random.choice(['prestado', 'oferecido', 'realizado']),
            periodo=random.choice(['final de semana', 'feriado', 'mês']),
            data_comemorativa=random.choice(['Natal', 'Páscoa', 'Ano Novo']),
            evento_futuro=random.choice(['evento próximo', 'lançamento', 'implementação'])
        )
        
        # Adicionar saudação e assinatura
        full_email = f"""
De: {fake.email()}
Para: equipe@empresa.com
Assunto: {random.choice(['Agradecimento', 'Felitações', 'Cumprimentos', 'Reconhecimento'])}

Olá equipe,

{email_text}

Atenciosamente,
{fake.name()}
{fake.job()}
        """.strip()
        
        return full_email
    
    def generate_dataset(self, num_emails=200):
        """Gera o dataset completo"""
        emails = []
        categories = []
        
        # Balancear entre categorias
        num_productive = num_emails // 2
        num_unproductive = num_emails - num_productive
        
        print(f"Gerando {num_productive} emails produtivos...")
        for _ in range(num_productive):
            emails.append(self.generate_productive_email())
            categories.append("Produtivo")
        
        print(f"Gerando {num_unproductive} emails improdutivos...")
        for _ in range(num_unproductive):
            emails.append(self.generate_unproductive_email())
            categories.append("Improdutivo")
        
        # Criar DataFrame
        df = pd.DataFrame({
            'email_text': emails,
            'category': categories
        })
        
        # Embaralhar o dataset
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return df
    
    def save_dataset(self, filename='email_dataset.csv', num_emails=200):
        """Salva o dataset em arquivo CSV"""
        df = self.generate_dataset(num_emails)
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"Dataset salvo como {filename}")
        print(f"Shape: {df.shape}")
        print(f"Distribuição por categoria:")
        print(df['category'].value_counts())
        
        return df

# Função para análise do dataset gerado
def analyze_dataset(df):
    """Analisa o dataset gerado"""
    print("\n" + "="*50)
    print("ANÁLISE DO DATASET")
    print("="*50)
    
    print(f"Total de emails: {len(df)}")
    print(f"Distribuição por categoria:")
    print(df['category'].value_counts())
    
    print("\nExemplo de email Produtivo:")
    productive_example = df[df['category'] == 'Produtivo'].iloc[0]['email_text']
    print(productive_example[:300] + "...")
    
    print("\nExemplo de email Improdutivo:")
    unproductive_example = df[df['category'] == 'Improdutivo'].iloc[0]['email_text']
    print(unproductive_example[:300] + "...")
    
    # Estatísticas básicas de texto
    df['text_length'] = df['email_text'].apply(len)
    df['word_count'] = df['email_text'].apply(lambda x: len(x.split()))
    
    print(f"\nEstatísticas de texto:")
    print(f"Comprimento médio: {df['text_length'].mean():.0f} caracteres")
    print(f"Palavras médias: {df['word_count'].mean():.0f} palavras")
    print(f"Comprimento mínimo: {df['text_length'].min()} caracteres")
    print(f"Comprimento máximo: {df['text_length'].max()} caracteres")

if __name__ == "__main__":
    # Gerar o dataset
    n = 500  # Número total de emails a serem gerados
    generator = EmailDatasetGenerator()
    df = generator.save_dataset('email_dataset.csv', num_emails=n)
    
    # Analisar o dataset
    analyze_dataset(df)
    
    # Mostrar primeiras linhas
    print("\nPrimeiras 5 linhas do dataset:")
    print(df.head())