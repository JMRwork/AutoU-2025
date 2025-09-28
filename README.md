# Documentação de Utilização dos Arquivos `gerar_emails.py`, `modelo.py` e `app.py`

## Visão Geral

Este projeto é composto por três arquivos principais:

- **`gerar_emails.py`**: Responsável pela geração de e-mails de exemplo ou dados de entrada para o modelo.
- **`modelo.py`**: Contém a definição, treinamento e carregamento do modelo de processamento de e-mails.
- **`app.py`**: Implementa a aplicação principal, integrando a geração de e-mails e o modelo para fornecer funcionalidades ao usuário final.

## Como Utilizar
### 1. Baixar os pacotes do `requirements.txt`
> `pip install -r requirements.txt`

### 2. Gerar Dados de E-mail

Antes de executar a aplicação principal, utilize o `gerar_emails.py` para criar os dados de e-mails necessários para treinar ou testar o modelo. Definindo o parâmetro `n` para o número de dados de emails desejados para o dataset a ser criado.

### 3. Crie o Modelo com `modelo.py`

Execute o arquivo `modelo.py` após gerar os Dados de email. Gerando o modelo apartir dos dados `email_dataset.csv` gerado na etapa anterior.

### 4. Iniciando a Aplicação Web

Primeiro abra o arquivo `app.py` e configure a constante `TOKEN` para valor apropriado para acesso. Então Execute o arquivo. 

### 5. Acessando localmente Aplicativo

Depois que estiver tudo carregado. [Acesso ao aplicativo -> http://127.0.0.1:5000](http://127.0.0.1:5000)