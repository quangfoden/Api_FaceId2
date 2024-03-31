import cv2
import os
from flask import jsonify
from flask import Flask, request, render_template,redirect, url_for,send_file
from datetime import date
import mysql.connector
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import time
import io
from PIL import Image
import torch
from flask_cors import CORS
import shutil

app = Flask(__name__)
CORS(app)

nimgs = 10
camera_opened = False
imgBackground = cv2.imread("bg3.jpg")

# 'haarcascade_frontalface_default.xml' chứa một mô hình để phát hiện khuôn mặt phía trước trong ảnh
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')

connection = mysql.connector.connect(
    host='localhost',
    user='root',
    password='',
    database='social_network'
)
cursor = connection.cursor(buffered=True)

def totalreg():
    return len(os.listdir('static/faces'))

def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except:
        return []

def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)

def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

def login_user(name):
    username, user_id = name.split('_')
    cursor.execute("SELECT * FROM user_face_regs WHERE user_id = %s", (user_id,))
    user_record = cursor.fetchone()
    if user_record:
        print("Đăng nhập thành công cho người dùng:", user_record[2])
        return True
    else:
        print("Không tìm thấy thông tin người dùng.")
        return False
    
def is_face_registered(user_id):
    cursor.execute("SELECT * FROM face_ids WHERE user_id = %s", (user_id,))
    result = cursor.fetchone()
    if result:
        return True
    else:
        return False
 
def delete_registered_face(user_id, username):
    try:
        # Xóa bản ghi từ cơ sở dữ liệu
        cursor.execute("DELETE FROM face_ids WHERE user_id = %s", (user_id,))
        cursor.execute("DELETE FROM user_face_regs WHERE user_id = %s", (user_id,))
        connection.commit()
        # Xóa hình ảnh từ thư mục
        user_folder = f'static/faces/{username}_{user_id}'
        if os.path.exists(user_folder):
            shutil.rmtree(user_folder)
            print("Thư mục khuôn mặt đã được xóa.")
        else:
            print("Không tìm thấy thư mục khuôn mặt.")
    except Exception as e:
        print("Lỗi khi xóa khuôn mặt:", repr(e))
        connection.rollback()

@app.route('/start', methods=['POST'])
def start():
    ret = True
    count=1
    cap = cv2.VideoCapture(0)
    success = False
    while ret:
        ret, frame = cap.read()
        if len(extract_faces(frame)) > 0:
            (x, y, w, h) = extract_faces(frame)[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 128, 0), 1)
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            if login_user(identified_person):
                username, user_id = identified_person.split('_')
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 128, 0), 1)
                cv2.putText(frame, f'{username}', (x,y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 128, 0), 1)
                success = True
            else:
                # Thực hiện hành động sau khi đăng nhập thất bại
                success = False
           
        new_width = 400  # Đặt chiều rộng mới
        new_height = 400  # Đặt chiều cao mới

        # Thay đổi kích thước của hình ảnh cam
        small_frame = cv2.resize(frame, (new_width, new_height))

        # Tạo mask trắng có hình dạng hình vuông
        mask = np.ones((new_height, new_width), dtype=np.uint8) * 255

        # Điều chỉnh vị trí của hình ảnh cam trong imgBackground
        top = 200  # chỉ mục hàng đầu tiên của vùng muốn gán hình ảnh cam vào
        bottom = top + new_height  # chỉ mục hàng cuối cùng của vùng muốn gán hình ảnh cam vào
        left = 450  # chỉ mục cột đầu tiên của vùng muốn gán hình ảnh cam vào
        right = left + new_width  # chỉ mục cột cuối cùng của vùng muốn gán hình ảnh cam vào

        # Gán hình ảnh cam vào imgBackground
        imgBackground[top:bottom, left:right] = small_frame

        cv2.imshow('login', imgBackground)
        if cv2.waitKey(1) == 27:
        # thoát khỏi vòng lặp
            break 
        # đóng cammera
        count+=1
        if count > 100:
            break
    cap.release()
    cv2.destroyAllWindows()
    if success:
        return jsonify({'status': True, 'message': 'Xác thực thành công', 'user': {'username': username, 'user_id': user_id}}),200
    else:
        return jsonify({'status': False, 'message': 'xác thực thất bại'}),500
@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.json['userName']
    newuserid = request.json['userId']
    userimagefolder = 'static/faces/' + newusername + '_' + str(newuserid)
    # Kiểm tra xem người dùng đã tồn tại trong cơ sở dữ liệu chưa
    cursor.execute("SELECT * FROM user_face_regs WHERE user_id = %s OR user_name = %s", (newuserid, newusername))
    existing_user = cursor.fetchone()
    if existing_user:
        # Người dùng đã tồn tại, trả về phản hồi JSON
        return jsonify({'status': 'error', 'message': 'Người dùng đã đăng ký khuôn mặt trước đó'}), 400
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    i, j = 0, 0
    cap = cv2.VideoCapture(0) 
    # Thêm người dùng vào cơ sở dữ liệu
    try:
        cursor.execute("INSERT INTO user_face_regs (user_id, user_name) VALUES (%s, %s)", (newuserid, newusername))
        connection.commit()
        print("Người dùng đã được thêm vào cơ sở dữ liệu.")
    except Exception as e:
        print("Lỗi khi thêm người dùng vào cơ sở dữ liệu:", e)
        connection.rollback()
    
    while 1:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 128, 0), 1)
            cv2.putText(frame, f'{i}/{nimgs}', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 0), 2, cv2.LINE_AA)
            if j % 5 == 0:
                name = newusername + '_' + str(i) + '.jpg'
                face_image_path = os.path.join(userimagefolder, name)
                cv2.imwrite(face_image_path, frame[y:y+h, x:x+w])
                
                # Lưu đường dẫn hình ảnh khuôn mặt vào cơ sở dữ liệu
                try:
                    cursor.execute("INSERT INTO face_ids (user_id, face_image) VALUES (%s, %s)", (newuserid, face_image_path))
                    connection.commit()
                    print("Khuôn mặt đã được thêm vào cơ sở dữ liệu.")
                except Exception as e:
                    print("Lỗi khi thêm khuôn mặt vào cơ sở dữ liệu:", e)
                    connection.rollback()
                i += 1
            j += 1
        if j == nimgs*5:
            break

        new_width = 400  # Đặt chiều rộng mới
        new_height = 400  # Đặt chiều cao mới

        # Thay đổi kích thước của hình ảnh cam
        small_frame = cv2.resize(frame, (new_width, new_height))

        # Tạo mask trắng có hình dạng hình vuông
        mask = np.ones((new_height, new_width), dtype=np.uint8) * 255

        # Điều chỉnh vị trí của hình ảnh cam trong imgBackground
        top = 200  # chỉ mục hàng đầu tiên của vùng muốn gán hình ảnh cam vào
        bottom = top + new_height  # chỉ mục hàng cuối cùng của vùng muốn gán hình ảnh cam vào
        left = 450  # chỉ mục cột đầu tiên của vùng muốn gán hình ảnh cam vào
        right = left + new_width  # chỉ mục cột cuối cùng của vùng muốn gán hình ảnh cam vào

        # Gán hình ảnh cam vào imgBackground
        imgBackground[top:bottom, left:right] = small_frame

        cv2.imshow('add', imgBackground)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    return jsonify({'status':True, 'message': 'Đăng ký khuôn mặt thành công'})

@app.route('/delete_face', methods=['POST'])
def delete_face():
    user_id = request.json['user_id']
    username = request.json['username']
    print(user_id)
    print(username)
    try:
        # Kiểm tra xem khuôn mặt đã đăng ký hay chưa
        if is_face_registered(user_id):
            # Xóa khuôn mặt đã đăng ký
            delete_registered_face(user_id, username)
            return jsonify({'status': 'success', 'message': 'Khuôn mặt đã được xóa thành công'}), 200
        else:
            return jsonify({'status': 'error', 'message': 'Khuôn mặt không tồn tại trong cơ sở dữ liệu'}), 404
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
# Load YOLOv5 model
model = torch.hub.load("ultralytics/yolov5", "yolov5s", force_reload=True, skip_validation=True)

@app.route('/checkimage', methods=["POST"])
def check_image():
    """Receive an image from the client, perform object detection, and return the annotated image."""
    if request.method != "POST":
        return "Method not allowed", 405

    if request.files.get("image"):
        # Read the image from the client
        image_file = request.files["image"]
        image_bytes = image_file.read()
        image_pil = Image.open(io.BytesIO(image_bytes))

        # Perform object detection
        results = model(image_pil, size=640)

        # Get annotated image (take the first element in the results list)
        annotated_image = results.render()[0]

        # Convert the annotated image to bytes
        image_output = io.BytesIO()
        Image.fromarray(annotated_image).save(image_output, format="JPEG")
        image_output.seek(0)

        # Return the annotated image back to the client
        return send_file(image_output, mimetype="image/jpeg")

    return "No image file provided", 400

if __name__ == '__main__':
    app.run(debug=True)
