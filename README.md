https://github.com/tensorflow/tensorflow/issues/18503#issuecomment-385624410

# Machine-Learning-Models-Server

## Servidor ML_Server:

### Criação do ambiente dentro da pasta raiz:

	python -m venv env/models_server
	
### Ativação do ambiente virtual:
	WINDOWS Anaconda:
		conda activate models_server
	WINDOWS Venv:
		env\models_server\Scripts\activate
	LINUX:
		source ./env/models_server/bin/activate

	
### Instalar o Flask e dependências do projeto:

	pip install -r requirements.txt
	
### Para subir o servidor:

	set FLASK_ENV=development && flask run --host=0.0.0.0

### Para testar, com o servidor e o TF-Serving ativo:

	
