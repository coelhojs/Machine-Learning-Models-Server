# TF-Serving

## vera_species

### Download da Imagem Docker do TF-Serving:

    `docker pull tensorflow/serving`

### Servidor de classificação de espécies do Vera:

    docker run -p 8501:8501 --mount type=bind,source=C:\development\Machine-Learning-Models-Server\tensorflow_serving\vera_species,target=/models/vera_species -e MODEL_NAME=vera_species -t tensorflow/serving &

### Servidor de detecção de objetos do Vera:

    docker run -p 8500:8500 --mount type=bind,source=C:\development\Machine-Learning-Models-Server\tensorflow_serving\vera_poles_trees,target=/models/vera_poles_trees -e MODEL_NAME=vera_poles_trees -t tensorflow/serving &