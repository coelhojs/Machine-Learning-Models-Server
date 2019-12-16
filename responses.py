import requests


def send_object_detection_results (result):
    try:
        headers = {"content-type": "application/json"}
        server_url = 'http://localhost:3857/MachineLearning/Poles_Trees_Results'
        print(f'\n\nEnviando resultados para {server_url}...\n')
        server_response = requests.post(server_url, data=result, headers=headers)
        print ('response from server:')
        print (server_response.text)
            
    except:
        print ('Houve um erro ao tentar enviar ao Vera os resultados')