from flask import Flask, Response
import cv2
import time

app = Flask(__name__)

def gen_frames():
    cap = cv2.VideoCapture(0)

    prev_time = time.time()
    frame_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        current_time = time.time()

        # 5초마다 FPS 출력
        if current_time - prev_time >= 5:
            fps = frame_count / (current_time - prev_time)
            print(f" 송출 FPS: {fps:.2f}")
            prev_time = current_time
            frame_count = 0

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)
