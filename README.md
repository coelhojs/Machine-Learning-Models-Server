# Machine-Learning-Models-Server

## Servidor ML_Server:

### Criação do ambiente dentro da pasta raiz:

	`python -m venv models_server`

### Ativação do ambiente virtual:
	
	`models_server\Scripts\activate`
	
### Instalar o Flask e dependências do projeto:

	`pip install Flask numpy requests tensorflow`
	
### Para subir o servidor:

	`set FLASK_ENV=development && flask run --host=0.0.0.0`