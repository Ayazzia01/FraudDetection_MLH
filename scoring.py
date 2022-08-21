import json
import joblib

def init():
    global model

    model_path = 'fraud_detection_model.pkl'
    model = joblib.load(model_path)

def run(raw_data, request_headers):
    data = json.loads(raw_data)["data"]
    result = model.predict(data)
    
    print(('{{"RequestId":"{0}", '
           '"TraceParent":"{1}", '
           '"NumberOfPredictions":{2}}}'
           ).format(
               request_headers.get("X-Ms-Request-Id", ""),
               request_headers.get("Traceparent", ""),
               len(result)
    ))

    return {"result": result.tolist()}

init()
test_row = '{"data":[[12, PAYMENT, 734.52, C1088161828, 35628, 34893.48, M476137774, 0, 0],[85, CASH_OUT, 1529614.71, C955071434, 1529614.71, 0, C865727492, 0, 1529614.71, 0]]}'
request_header = {}
prediction = run(test_row, request_header)
print("Test result: ", prediction)