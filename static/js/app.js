const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const fileInput = document.getElementById("fileInput");

const startCameraBtn = document.getElementById("startCamera");
const captureBtn = document.getElementById("capture");
const sendImageBtn = document.getElementById("sendImage");

const previewWrap = document.getElementById("previewWrap");
const previewImg = document.getElementById("previewImg");
const previewMeta = document.getElementById("previewMeta");
const clearPreviewBtn = document.getElementById("clearPreview");

let imageBlob = null;
let mediaStream = null;

function setPreview(blob, filename = "captura.jpg") {
    imageBlob = blob;

    if (!blob) return;

    const url = URL.createObjectURL(blob);
    previewImg.src = url;
    previewImg.alt = "Vista previa";
    previewMeta.textContent = filename;

    if (previewWrap) previewWrap.classList.remove("hidden");
    if (sendImageBtn) sendImageBtn.disabled = false;

    // Si quieres evitar fugas de memoria, revoca el objeto URL cuando sea reemplazado.
    // (En este flujo el usuario puede capturar varias veces.)
}

function stopCameraIfAny() {
    if (mediaStream) {
        mediaStream.getTracks().forEach((t) => t.stop());
        mediaStream = null;
    }
}

function showCameraError(error) {
    // Mensajes más específicos según el tipo de error.
    if (error && error.name === "NotAllowedError") {
        alert("Permiso denegado: habilita el acceso a la cámara en el navegador.");
        return;
    }
    if (error && error.name === "NotFoundError") {
        alert("No se encontró una cámara disponible en este dispositivo.");
        return;
    }
    if (error && error.name === "NotReadableError") {
        alert("La cámara está siendo usada por otra app (o está bloqueada). Cierra otras apps e inténtalo.");
        return;
    }

    alert("No se pudo acceder a la cámara.");
}

// Activar cámara
startCameraBtn.addEventListener("click", async () => {
    try {
        stopCameraIfAny();

        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: "user"
            }
        });

        mediaStream = stream;
        video.srcObject = stream;

        // Asegurar que el botón Capturar quede habilitado cuando ya hay stream.
        captureBtn.disabled = false;

        // Mantener el botón de analizar bloqueado hasta capturar/seleccionar.
        sendImageBtn.disabled = true;

        // ocultar preview anterior (si aplica)
        if (previewWrap) previewWrap.classList.add("hidden");
        imageBlob = null;

    } catch (error) {
        captureBtn.disabled = true;
        showCameraError(error);
        console.error(error);
    }
});

// Capturar foto
captureBtn.addEventListener("click", () => {
    const w = video.videoWidth;
    const h = video.videoHeight;

    if (!w || !h) {
        alert("La cámara aún no está lista. Espera 1–2 segundos e intenta de nuevo.");
        return;
    }

    canvas.width = w;
    canvas.height = h;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(
        (blob) => {
            if (!blob) {
                alert("No se pudo capturar la imagen.");
                return;
            }
            setPreview(blob, "captura.jpg");
            alert("Foto capturada correctamente");
        },
        "image/jpeg",
        0.92
    );
});

// Selección de archivo desde disco
fileInput.addEventListener("change", () => {
    // Si el usuario selecciona desde disco, limpia la captura anterior.
    imageBlob = null;

    if (fileInput.files && fileInput.files[0]) {
        const file = fileInput.files[0];
        const url = URL.createObjectURL(file);

        if (previewWrap) previewWrap.classList.remove("hidden");
        if (previewImg) previewImg.src = url;
        if (previewMeta) previewMeta.textContent = file.name;

        sendImageBtn.disabled = false;
    }
});

// Limpiar preview
if (clearPreviewBtn) {
    clearPreviewBtn.addEventListener("click", () => {
        // Limpiar estado
        imageBlob = null;
        fileInput.value = "";

        if (previewWrap) previewWrap.classList.add("hidden");
        if (previewImg) previewImg.src = "";
        if (previewMeta) previewMeta.textContent = "";

        sendImageBtn.disabled = true;
    });
}

// Enviar imagen
sendImageBtn.addEventListener("click", async () => {
    const formData = new FormData();

    // Prioridad 1: imagen seleccionada desde disco
    if (fileInput.files.length > 0) {
        formData.append("file", fileInput.files[0]);
    }
    // Prioridad 2: captura de cámara
    else if (imageBlob) {
        formData.append("file", imageBlob, "captura.jpg");
    } else {
        alert("Seleccione una imagen o capture una foto");
        return;
    }

    try {
        const response = await fetch("/detect", {
            method: "POST",
            body: formData
        });

        const data = await response.json();
        console.log(data);

        if (!response.ok) {
            alert(data?.mensaje || "Error al procesar la imagen");
            return;
        }

        alert("Imagen enviada correctamente");
    } catch (error) {
        console.error(error);
        alert("Error al conectar con la API");
    }
});

// Estado inicial (por si el HTML viene con disabled)
captureBtn.disabled = true;
sendImageBtn.disabled = true;

