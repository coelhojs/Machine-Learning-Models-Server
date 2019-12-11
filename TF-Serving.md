# TF-Serving

## vera_species

### Atualizar certificados do Docker Toolbox

    docker-machine regenerate-certs default

### Adicionar o disco local à maquina virtual do Oracle VirtualBox

    Abra o VirtualBox e acesse as configurações da VM default. Adicione o disco local C: a lista de diretórios reconhecidos e selecione a opção permanente

### Download da Imagem Docker do TF-Serving:

    `docker pull tensorflow/serving`

### Servidor de classificação de espécies do Vera:

    `docker run -p 8501:8501 --mount type=bind,source=C:\development\Machine-Learning-Models-Server\tensorflow_serving\vera_species,target=/models/vera_species -e MODEL_NAME=vera_species -t tensorflow/serving &`