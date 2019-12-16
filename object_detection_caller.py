import asyncio
import requests
from PIL import Image
from tensorflow_scripts.utils import img_util, label_map_util
from tensorflow_scripts.object_detection.object_detection import objects_detector
from responses import send_object_detection_results

async def object_detection_batch_serving(prediction_obj, splitted_list_images, model_path, labels, server_url):
    try:
        for list_images in splitted_list_images:
            for image_path in list_images:
                # Build input data
                print(f'\n\nPre-processing input file {image_path}...\n')
                formatted_json_input = img_util.object_detection_pre_process(image_path)
                print('Pre-processing done! \n')

                # Call tensorflow server
                headers = {"content-type": "application/json"}
                print(f'\n\nMaking request to {server_url}...\n')
                server_response = requests.post(server_url, data=formatted_json_input, headers=headers)
                print(f'Request returned\n')
                print(server_response)

                # Post process output
                print(f'\n\nPost-processing server response...\n')
                image = Image.open(image_path).convert("RGB")
                image_np = img_util.load_image_into_numpy_array(image)
                output_dict = img_util.post_process(server_response, image_np.shape, labels)

                # Formatando resultado para o modelo esperado pelo Vera
                inference_dict = {}
                inference_dict['ImagePath'] = image_path
                inference_dict['Classes'] = output_dict['detection_classes']
                inference_dict['BoundingBoxes'] = output_dict['detection_boxes']
                inference_dict['Scores'] = np.array(output_dict['detection_scores']).tolist()
                inference_dict['NumDetections'] = output_dict['num_detections']

                print(f'Post-processing done!\n')

                prediction_obj['Detections'].append(inference_dict)

            send_object_detection_results(prediction_obj)

        return "Detecção em lote finalizada"
    
    except:
        return "Houve um erro na detecção de objetos em lote"

async def object_detection_batch_script(prediction_obj, splitted_list_images, model_path, labels):
    try:
        for list_images in splitted_list_images:
            #Limpa a lista de resultados antes de iniciar um lote de detecções
            prediction_obj['Detections'] = []
            
            images_predictions = objects_detector(list_images, model_path, labels)

            for x in range(len(images_predictions)):
                prediction_obj['Detections'].append(images_predictions[x])

            send_object_detection_results(prediction_obj)

        return "Detecção em lote finalizada"

    except:
        return "Houve um erro na detecção de objetos em lote"