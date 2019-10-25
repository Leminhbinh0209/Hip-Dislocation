from utils import *
# Hip Processing


graph = None
hp = None
# server port
my_port = '5000'
# Init server
app = Flask(__name__)
CORS(app)


def load_model():
    # Get default graph:
    global graph
    graph = tf.get_default_graph()
    global hp
    hp = HipProcessing()

# request index
@app.route('/')
@cross_origin()
def index():
    return "Welcome to flask API!"

# request hello_word
@app.route('/hello_world', methods=['GET'])
@cross_origin()
def hello_world():
    # Get staff id of client 
    staff_id = request.args.get('staff_id')
    # return greeting Hello
    return "Hello "  + str(staff_id)

# Khai bao ham xu ly request detect
@app.route('/detect', methods=['POST'])
@cross_origin()
def detect():
    if request.method == 'POST':
        # Get image B64 data and transform to image
        # image_b64 = request.files["image"].read() # demo python
        # image = np.asarray(bytearray(image_b64), dtype=np.uint8) # demo python

        image_b64 = request.form['image']
        image = np.fromstring(base64.b64decode(image_b64), dtype=np.uint8) # incase C#

        img_opencv = cv2.imdecode(image, cv2.IMREAD_ANYCOLOR)
        #  To Gray scale
        gray = cv2.cvtColor(img_opencv, cv2.COLOR_BGR2GRAY)
        # Image size
        Width = gray.shape[1]
        Height = gray.shape[0]

        pred_points = hp.hip_points_detection(gray)
        retString = str(int(np.round(pred_points[0])))
        for p in pred_points[1:]:
                retString += ","
                retString += str(int(np.round(p)))
        # return jsonify({"status": "ok", "result": retString}), 200
        print(retString)
        return retString


# Run server
if __name__ == '__main__':
    load_model()
    app.run(debug=True, host='0.0.0.0',port=my_port)
