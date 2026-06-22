const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const fileInput = document.getElementById("fileInput");

let imageBlob = null;

// Activar cámara
document
    .getElementById("startCamera")
    .addEventListener("click", async () => {

        try {

            const stream =
                await navigator.mediaDevices.getUserMedia({
                    video: true
                });

            video.srcObject = stream;

        } catch (error) {

            alert("No se pudo acceder a la cámara");
            console.error(error);

        }

    });

// Capturar foto
document
    .getElementById("capture")
    .addEventListener("click", () => {

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        const ctx = canvas.getContext("2d");

        ctx.drawImage(
            video,
            0,
            0,
            canvas.width,
            canvas.height
        );

        canvas.toBlob(blob => {

            imageBlob = blob;

            alert("Foto capturada correctamente");

        }, "image/jpeg");

    });

// Enviar imagen
document
    .getElementById("sendImage")
    .addEventListener("click", async () => {

        const formData = new FormData();

        // Prioridad 1: imagen seleccionada
        if (fileInput.files.length > 0) {

            formData.append(
                "file",
                fileInput.files[0]
            );

        }

        // Prioridad 2: captura de cámara
        else if (imageBlob) {

            formData.append(
                "file",
                imageBlob,
                "captura.jpg"
            );

        }

        else {

            alert(
                "Seleccione una imagen o capture una foto"
            );

            return;
        }

        try {

            const response = await fetch("/detect", {
                method: "POST",
                body: formData
            });

            const data =
                await response.json();

            console.log(data);

            alert("Imagen enviada correctamente");

        } catch (error) {

            console.error(error);

            alert(
                "Error al conectar con la API"
            );

        }

    });