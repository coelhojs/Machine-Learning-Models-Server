https://github.com/tensorflow/tensorflow/issues/18503#issuecomment-385624410

# Machine-Learning-Models-Server

## Servidor ML_Server:

### Criação do ambiente dentro da pasta raiz:

	python -m venv env/models_server
	
### Ativação do ambiente virtual:
	
	env\models_server\Scripts\activate
	
### Instalar o Flask e dependências do projeto:

	pip install -r requirements.txt
	
### Para subir o servidor:

	set FLASK_ENV=development && flask run --host=0.0.0.0

### Para testar, com o servidor e o TF-Serving ativo:

	curl --request POST --url http:/localhost:5000/vera_species/classify --header 'content-type: application/json' --data '{"data": "c:\\development\\tensorflow-serving\\mangueira.jpg"}'


	curl --request POST --url http://localhost:5000/vera_poles_trees/detect --header 'content-type: application/json' --data '{"data": "C:/Machine-Learning-Models-Server/test_images/1.png"}'
