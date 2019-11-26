# Machine-Learning-Models-Server

## Servidor ML_Server:

### Criação do ambiente dentro da pasta raiz:

	python -m venv envs/models_server
	
### Ativação do ambiente virtual:
	
	envs\models_server\Scripts\activate
	
### Instalar o Flask e dependências do projeto:

	pip install Flask numpy requests tensorflow
	
### Para subir o servidor:

	set FLASK_ENV=development && flask run --host=0.0.0.0

### Para testar, com o servidor e o TF-Serving ativo:

	curl --request POST \
  	--url http://localhost:5000/vera_species \
  	--header 'content-type: application/json' \
  	--data '{"data": "c:\\development\\tensorflow-serving\\mangueira.jpg"}'
