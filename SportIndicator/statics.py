import json
def read_data(predict_path):   
    with open(predict_path, 'r', encoding='utf-8') as f:
        predictions = json.load(f)

    return predictions

'''
body_part_file      = "./results/boxing_aligned/BodyPart_Detection_geval_epoch_115.json"
causation_file      = "./results/boxing_aligned/Causation_Detection_geval_epoch_115.json"
coordination_file   = "./results/boxing_aligned/Coordination_Detection_geval_epoch_115.json"
error_file          = "./results/boxing_aligned/Error_Detection_geval_epoch_115.json"
method_file         = "./results/boxing_aligned/Method_Detection_geval_epoch_115.json"
time_file           = "./results/boxing_aligned/Time_Detection_geval_epoch_115.json"

body_part_file      = "./results/boxing_llama/BodyPart_Detection_geval_epoch_31.json"
causation_file      = "./results/boxing_llama/Causation_Detection_geval_epoch_31.json"
coordination_file   = "./results/boxing_llama/Coordination_Detection_geval_epoch_31.json"
error_file          = "./results/boxing_llama/Error_Detection_geval_epoch_31.json"
method_file         = "./results/boxing_llama/Method_Detection_geval_epoch_31.json"
time_file           = "./results/boxing_llama/Time_Detection_geval_epoch_31.json"
'''
body_part_file      = "./results/GT_FS/BodyPart_Detection_results.json"
causation_file      = "./results/GT_FS/Causation_Detection_results.json"
coordination_file   = "./results/GT_FS/Coordination_Detection_results.json"
error_file          = "./results/GT_FS/Error_Detection_results.json"
method_file         = "./results/GT_FS/Method_Detection_results.json"
time_file           = "./results/GT_FS/Time_Detection_results.json"

if __name__ == '__main__':
    body_part = read_data(body_part_file)
    causation = read_data(causation_file)
    coordination = read_data(coordination_file)
    error = read_data(error_file)
    method = read_data(method_file)
    time = read_data(time_file)

    metric1 = { "total": 0, "score_1" : 0, "score_2" : 0, "score_3" : 0, "score_4" : 0, "score_5" : 0}
    metric2 = { "total": 0, "score_1" : 0, "score_2" : 0, "score_3" : 0, "score_4" : 0, "score_5" : 0}
    metric3 = { "total": 0, "score_1" : 0, "score_2" : 0, "score_3" : 0, "score_4" : 0, "score_5" : 0}
    metric4 = { "total": 0, "score_1" : 0, "score_2" : 0, "score_3" : 0, "score_4" : 0, "score_5" : 0}
    metric5 = { "total": 0, "score_1" : 0, "score_2" : 0, "score_3" : 0, "score_4" : 0, "score_5" : 0}
    metric6 = { "total": 0, "score_1" : 0, "score_2" : 0, "score_3" : 0, "score_4" : 0, "score_5" : 0}
    
    all_instructions = {}
    for item in body_part:
        all_instructions[item["file_name"]] = { 
            "score" : item["score"],
            "1" :item["BodyPart_Detection_score"]}

        if item["BodyPart_Detection_score"] == 1 :
            metric3["total"] += 1
            if isinstance(item["score"], list):
                item["score"] = max(item["score"])
            if item["score"] == 1: metric3["score_1"] += 1
            elif item["score"] == 2: metric3["score_2"] += 1
            elif item["score"] == 3: metric3["score_3"] += 1
            elif item["score"] == 4: metric3["score_4"] += 1
            elif item["score"] == 5: metric3["score_5"] += 1

    for item in causation:
        all_instructions[item["file_name"]]["2"] = item["Causation_Detection_score"]
        if item["Causation_Detection_score"] == 1 :
            metric4["total"] += 1
            if isinstance(item["score"], list):
                item["score"] = max(item["score"])
            if item["score"] == 1: metric4["score_1"] += 1
            elif item["score"] == 2: metric4["score_2"] += 1
            elif item["score"] == 3: metric4["score_3"] += 1
            elif item["score"] == 4: metric4["score_4"] += 1
            elif item["score"] == 5: metric4["score_5"] += 1

    for item in coordination:
        all_instructions[item["file_name"]]["3"] = item["Coordination_Detection_score"]
        if item["Coordination_Detection_score"] == 1 :
            metric6["total"] += 1
            if isinstance(item["score"], list):
                item["score"] = max(item["score"])
            if item["score"] == 1: metric6["score_1"] += 1
            elif item["score"] == 2: metric6["score_2"] += 1
            elif item["score"] == 3: metric6["score_3"] += 1
            elif item["score"] == 4: metric6["score_4"] += 1
            elif item["score"] == 5: metric6["score_5"] += 1

    for item in error:
        all_instructions[item["file_name"]]["4"] = item["Error_Detection_score"]
        if item["Error_Detection_score"] == 1 :
            metric1["total"] += 1
            if isinstance(item["score"], list):
                item["score"] = max(item["score"])
            if item["score"] == 1: metric1["score_1"] += 1
            elif item["score"] == 2: metric1["score_2"] += 1
            elif item["score"] == 3: metric1["score_3"] += 1
            elif item["score"] == 4: metric1["score_4"] += 1
            elif item["score"] == 5: metric1["score_5"] += 1

    for item in method:
        all_instructions[item["file_name"]]["5"] = item["Method_Detection_score"]
        if item["Method_Detection_score"] == 1 :
            metric5["total"] += 1
            if isinstance(item["score"], list):
                item["score"] = max(item["score"])
            if item["score"] == 1: metric5["score_1"] += 1
            elif item["score"] == 2: metric5["score_2"] += 1
            elif item["score"] == 3: metric5["score_3"] += 1
            elif item["score"] == 4: metric5["score_4"] += 1
            elif item["score"] == 5: metric5["score_5"] += 1
    
    for item in time:
        all_instructions[item["file_name"]]["6"] = item["Time_Detection_score"]
        if item["Time_Detection_score"] == 1 :
            metric2["total"] += 1
            if isinstance(item["score"], list):
                item["score"] = max(item["score"])
            if item["score"] == 1: metric2["score_1"] += 1
            elif item["score"] == 2: metric2["score_2"] += 1
            elif item["score"] == 3: metric2["score_3"] += 1
            elif item["score"] == 4: metric2["score_4"] += 1
            elif item["score"] == 5: metric2["score_5"] += 1

    # create a 6x6 matrix
    matrix = [[0] * 6 for _ in range(6)]

    for i in range(0,6):
        for j in range(0,6):
            if i == j: continue
            for item in all_instructions:
                data = all_instructions[item]
                key1 = str(i + 1)
                key2 = str(j + 1)

                if key1 in data and key2 in data:
                    if isinstance(data["score"], list) :
                        bestscore = max(data["score"])
                    else :
                        bestscore = data["score"]
                    if data[key1] > 0 and data[key2] > 0 and data["score"] > 1:
                        matrix[i][j] += data["score"]

    data = {
        "metric1": metric1,
        "metric2": metric2,
        "metric3": metric3,
        "metric4": metric4,
        "metric5": metric5,
        "metric6": metric6,
        "matrix" : matrix
    }       
    output_filepath = "analyze_boxing.json"
    with open(output_filepath, 'w') as f:
        json.dump(data, f, indent=4)
