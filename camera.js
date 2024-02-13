document.addEventListener('DOMContentLoaded', () => {
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const countdownElement = document.getElementById('countdown');
    const instructionsElement = document.getElementById('instructions');
    const timerElement = document.getElementById('timer');

    let countdown = 3;
    let intervalId;

    // Function to start the countdown and capture images
    function startCountdown() {
        intervalId = setInterval(() => {
            countdown--;
            countdownElement.textContent = countdown;

            if (countdown <= 0) {
                clearInterval(intervalId);
                captureImage();
                countdown = 3;
                startCountdown();
            }
        }, 1000);
    }

    // Function to capture an image and process face and hand detection
    async function captureImage() {
        const context = canvas.getContext('2d');
        context.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Perform face and hand detection using face-api.js and OpenCV
        const img = cv.imread(canvas);
        const gray = new cv.Mat();
        cv.cvtColor(img, gray, cv.COLOR_RGBA2GRAY);

        // Use face-api.js for face detection
        const faces = await faceapi.detectAllFaces(gray).withFaceLandmarks();
        faces.forEach(face => {
            // Draw rectangle around the detected face
            const box = face.detection.box;
            context.beginPath();
            context.rect(box.x, box.y, box.width, box.height);
            context.strokeStyle = 'red';
            context.lineWidth = 2;
            context.stroke();
        });

        // Perform hand detection (you may need a hand detection model)
        // ...

        cv.imshow(canvas, img);
        gray.delete();
        img.delete();
    }

    // Get user media for camera
    navigator.mediaDevices.getUserMedia({ video: true })
        .then((stream) => {
            video.srcObject = stream;
            video.play();
            startCountdown();
        })
        .catch((error) => {
            console.error('Error accessing camera: ', error);
        });
});