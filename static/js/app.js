const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const fileInput = document.getElementById("fileInput");

const startCameraBtn = document.getElementById("startCamera");
const switchCameraBtn = document.getElementById("switchCamera");
const captureBtn = document.getElementById("capture");
const sendImageBtn = document.getElementById("sendImage");
const btnSpinner = document.getElementById("btnSpinner");
const btnText = document.querySelector("#sendImage .btn-text");

const previewWrap = document.getElementById("previewWrap");
const previewImg = document.getElementById("previewImg");
const previewMeta = document.getElementById("previewMeta");
const clearPreviewBtn = document.getElementById("clearPreview");
const boxesCanvas = document.getElementById("boxesCanvas");

const cameraOverlay = document.getElementById("cameraOverlay");

// Paneles de resultados (panel derecho)
const emptyState = document.getElementById("emptyState");
const loadingState = document.getElementById("loadingState");
const errorState = document.getElementById("errorState");
const noDetectState = document.getElementById("noDetectState");
const resultsState = document.getElementById("resultsState");

const errorTitle = document.getElementById("errorTitle");
const errorMsg = document.getElementById("errorMsg");
const retryBtn = document.getElementById("retryBtn");
const noDetectMsg = document.getElementById("noDetectMsg");

const eppGrid = document.getElementById("eppGrid");
const resultsList = document.getElementById("resultsList");
const resultsTime = document.getElementById("resultsTime");
const eppAlert = document.getElementById("eppAlert");
const eppAlertTitle = document.getElementById("eppAlertTitle");
const lowResWarning = document.getElementById("lowResWarning");
const eppAlertMsg = document.getElementById("eppAlertMsg");
const eppAlertIcon = document.getElementById("eppAlertIcon");

let imageBlob = null;
let mediaStream = null;
let currentFacingMode = "environment";

// ===============================
// TABS (Subir imagen / Usar cámara)
// ===============================
const tabButtons = document.querySelectorAll(".tab");
const tabPanels = document.querySelectorAll(".tab-panel");

tabButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
        const target = btn.dataset.tab; // "upload" | "camera"

        tabButtons.forEach((b) => b.classList.toggle("active", b === btn));
        tabPanels.forEach((panel) => {
            panel.classList.toggle("active", panel.id === `tab-${target}`);
        });

        // Si salimos de la pestaña de cámara, la apagamos para no dejarla
        // encendida en segundo plano sin que el usuario la vea.
        if (target !== "camera") {
            stopCameraIfAny();
        }
    });
});

// ===============================
// PANEL DE RESULTADOS — helpers
// ===============================
function showResultPanel(panelToShow) {
    [emptyState, loadingState, errorState, noDetectState, resultsState].forEach(
        (panel) => {
            if (panel) panel.classList.toggle("hidden", panel !== panelToShow);
        }
    );
}

// ===============================
// BOUNDING BOXES — dibujar cajas sobre la vista previa
// ===============================
// Colores por tipo de detección (presente = verde, incumplimiento = rojo, neutro = ámbar)
function colorParaClase(clase) {
    if (clase.startsWith("NO-")) return "#ef4444";          // rojo: incumplimiento
    if (clase === "Hardhat" || clase === "Safety Vest" || clase === "Mask") return "#22c55e"; // verde: EPP presente
    return "#f59e0b";                                          // ámbar: otras clases (Person, vehicle, etc.)
}

function limpiarBoxes() {
    if (!boxesCanvas) return;
    const ctx = boxesCanvas.getContext("2d");
    ctx.clearRect(0, 0, boxesCanvas.width, boxesCanvas.height);
}

function dibujarBoxes(resultados) {
    if (!boxesCanvas || !previewImg) return;

    // El contenedor (preview-image-stack) define el tamaño visible real.
    const rect = previewImg.getBoundingClientRect();
    boxesCanvas.width = rect.width;
    boxesCanvas.height = rect.height;

    const ctx = boxesCanvas.getContext("2d");
    ctx.clearRect(0, 0, boxesCanvas.width, boxesCanvas.height);

    const naturalW = previewImg.naturalWidth;
    const naturalH = previewImg.naturalHeight;
    if (!naturalW || !naturalH) return;

    // object-fit: contain → calcular el rectángulo real donde se dibuja
    // la imagen dentro del contenedor (puede haber franjas negras).
    const containerW = rect.width;
    const containerH = rect.height;
    const scale = Math.min(containerW / naturalW, containerH / naturalH);
    const renderedW = naturalW * scale;
    const renderedH = naturalH * scale;
    const offsetX = (containerW - renderedW) / 2;
    const offsetY = (containerH - renderedH) / 2;

    resultados.forEach((r) => {
        if (!r.box) return;
        const [x1, y1, x2, y2] = r.box;

        const px1 = offsetX + x1 * scale;
        const py1 = offsetY + y1 * scale;
        const px2 = offsetX + x2 * scale;
        const py2 = offsetY + y2 * scale;

        const color = colorParaClase(r.clase);
        const etiqueta = `${r.clase_es || r.clase} ${r.confianza.toFixed(0)}%`;

        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(px1, py1, px2 - px1, py2 - py1);

        ctx.font = "600 12px 'Space Grotesk', sans-serif";
        const textWidth = ctx.measureText(etiqueta).width;
        const labelH = 18;
        const labelY = py1 - labelH >= 0 ? py1 - labelH : py1;

        ctx.fillStyle = color;
        ctx.fillRect(px1, labelY, textWidth + 10, labelH);

        ctx.fillStyle = "#0a0a0a";
        ctx.fillText(etiqueta, px1 + 5, labelY + 13);
    });
}

window.addEventListener("resize", () => {
    if (previewWrap && !previewWrap.classList.contains("hidden") && window._ultimasDetecciones) {
        dibujarBoxes(window._ultimasDetecciones);
    }
});

function setSendingState(isSending) {
    sendImageBtn.disabled = isSending;
    if (btnSpinner) btnSpinner.classList.toggle("hidden", !isSending);
    if (btnText) btnText.textContent = isSending ? "Analizando…" : "Verificar EPP";
}

function renderResults(data) {
    const resultados = data.resultados || [];

    // Aviso de baja resolución — el backend marca resolucion_baja cuando el
    // lado menor de la imagen es muy pequeño (ver MIN_RESOLUCION en index.py).
    // No bloquea el resultado, solo avisa para interpretar con cautela.
    if (lowResWarning) {
        lowResWarning.classList.toggle("hidden", !data.resolucion_baja);
    }

    // Lista cruda de detecciones del modelo
    if (resultsList) {
        resultsList.innerHTML = resultados
            .map(
                (r, i) => `
                <div class="result-item">
                    <span class="result-rank">${i + 1}</span>
                    <div class="result-info">
                        <span class="result-label">${r.clase_es || r.clase}</span>
                        <div class="result-bar-track">
                            <div class="result-bar" style="width:${Math.min(r.confianza, 100)}%"></div>
                        </div>
                    </div>
                    <span class="result-pct">${r.confianza}%</span>
                </div>`
            )
            .join("");
    }

    if (resultsTime) {
        resultsTime.textContent = `Detecciones: ${resultados.length}`;
    }

    // Estado de EPP: el modelo a veces da detecciones contradictorias sobre
    // la misma persona (ej. "Hardhat" y "NO-Hardhat" ambas presentes, con
    // distinta confianza) — es un comportamiento conocido de este modelo,
    // visible incluso en sus propios datos oficiales de validación. Para
    // evitar mostrar ambas a la vez, nos quedamos solo con la detección de
    // mayor confianza dentro de cada par (casco vs. sin-casco, chaleco vs.
    // sin-chaleco).
    function mejorDetección(clasePositiva, claseNegativa) {
        const positivas = resultados.filter((r) => r.clase === clasePositiva);
        const negativas = resultados.filter((r) => r.clase === claseNegativa);
        const maxPositiva = positivas.length ? Math.max(...positivas.map((r) => r.confianza)) : -1;
        const maxNegativa = negativas.length ? Math.max(...negativas.map((r) => r.confianza)) : -1;

        if (maxPositiva === -1 && maxNegativa === -1) return null; // ninguna detectada
        return maxPositiva >= maxNegativa ? "presente" : "ausente";
    }

    const estadoCasco = mejorDetección("Hardhat", "NO-Hardhat");
    const estadoChaleco = mejorDetección("Safety Vest", "NO-Safety Vest");

    const tieneCasco = estadoCasco === "presente";
    const sinCasco = estadoCasco === "ausente";
    const tieneChaleco = estadoChaleco === "presente";
    const sinChaleco = estadoChaleco === "ausente";

    if (eppAlertTitle && eppAlertMsg && eppAlert) {
        eppAlert.classList.remove("ninguno", "parcial", "completo");

        if (tieneCasco && tieneChaleco) {
            eppAlertTitle.textContent = "✓ ACCESO PERMITIDO";
            eppAlertMsg.textContent = "Casco y chaleco de seguridad detectados correctamente.";
            eppAlert.classList.add("completo");
            if (eppAlertIcon) eppAlertIcon.textContent = "✓";
        } else if (tieneCasco || tieneChaleco) {
            // Falta al menos un elemento obligatorio → acceso denegado,
            // según la rúbrica del proyecto.
            eppAlertTitle.textContent = "✕ ACCESO DENEGADO";
            eppAlertMsg.textContent = `${tieneCasco ? "Casco presente" : "Falta casco"} · ${tieneChaleco ? "chaleco presente" : "falta chaleco"}.`;
            eppAlert.classList.add("ninguno");
            if (eppAlertIcon) eppAlertIcon.textContent = "✕";
        } else if (sinCasco || sinChaleco) {
            eppAlertTitle.textContent = "✕ ACCESO DENEGADO";
            eppAlertMsg.textContent = "Se detectaron personas sin el equipo de protección requerido.";
            eppAlert.classList.add("ninguno");
            if (eppAlertIcon) eppAlertIcon.textContent = "✕";
        } else {
            // No se identificó ni casco ni chaleco en ningún sentido
            // (ni presente ni ausente) — la imagen no permite verificar
            // el EPP, así que por seguridad también se niega el acceso.
            const top = resultados[0];
            eppAlertTitle.textContent = "✕ ACCESO DENEGADO";
            eppAlertMsg.textContent = top
                ? `No se pudo verificar casco ni chaleco (detectado: ${top.clase_es || top.clase}, ${top.confianza}%).`
                : "No se identificó casco ni chaleco en la imagen.";
            eppAlert.classList.add("ninguno");
            if (eppAlertIcon) eppAlertIcon.textContent = "✕";
        }
    }

    if (eppGrid) {
        // Construimos las tarjetas de casco/chaleco a partir del estado ya
        // resuelto (no de la lista cruda), para no repetir el mismo
        // problema de mostrar ambos lados de un par contradictorio.
        const tarjetas = [];
        if (estadoCasco) {
            tarjetas.push({
                nombre: "Casco de seguridad",
                presente: estadoCasco === "presente",
                confianza: Math.max(
                    ...resultados
                        .filter((r) => r.clase === (estadoCasco === "presente" ? "Hardhat" : "NO-Hardhat"))
                        .map((r) => r.confianza)
                ),
            });
        }
        if (estadoChaleco) {
            tarjetas.push({
                nombre: "Chaleco de seguridad",
                presente: estadoChaleco === "presente",
                confianza: Math.max(
                    ...resultados
                        .filter((r) => r.clase === (estadoChaleco === "presente" ? "Safety Vest" : "NO-Safety Vest"))
                        .map((r) => r.confianza)
                ),
            });
        }
        // Completamos con otras detecciones (Person, Mask, etc.) que no sean
        // parte de los pares ya resueltos arriba.
        const otras = resultados.filter(
            (r) => !["Hardhat", "NO-Hardhat", "Safety Vest", "NO-Safety Vest"].includes(r.clase)
        );
        for (const r of otras) {
            if (tarjetas.length >= 4) break;
            tarjetas.push({
                nombre: r.clase_es || r.clase,
                presente: !r.clase.startsWith("NO-"),
                confianza: r.confianza,
            });
        }

        eppGrid.innerHTML = tarjetas
            .slice(0, 4)
            .map(
                (t) => `
                <div class="epp-card ${t.presente ? "presente" : "ausente"}">
                    <span class="epp-card-icon">${t.presente ? "✓" : "⚠"}</span>
                    <div class="epp-card-info">
                        <div class="epp-card-name">${t.nombre}</div>
                        <div class="epp-card-status">${t.presente ? "Detectado" : "No detectado"}</div>
                        <div class="epp-card-conf">${t.confianza.toFixed(2)}%</div>
                    </div>
                </div>`
            )
            .join("");
    }

    window._ultimasDetecciones = resultados;
    // Esperar a que el navegador termine de pintar el layout (por si la
    // imagen acaba de cargar) antes de medir tamaños y dibujar las cajas.
    requestAnimationFrame(() => dibujarBoxes(resultados));

    showResultPanel(resultsState);
}

function setPreview(blob, filename = "captura.jpg") {
    imageBlob = blob;

    if (!blob) return;

    const url = URL.createObjectURL(blob);
    previewImg.src = url;
    previewImg.alt = "Vista previa";
    previewMeta.textContent = filename;

    limpiarBoxes();
    window._ultimasDetecciones = null;

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
// ===============================
// CAMARA
// ===============================

async function startCamera() {

    try {

        stopCameraIfAny();

        // Pedimos máxima resolución sin forzar landscape — en móvil vertical
        // 1920x1080 puede invertir alto/ancho y confundir al modelo.
        const constraints = {
            video: {
                facingMode: { ideal: currentFacingMode },
                width: { ideal: 4096 },
                height: { ideal: 4096 },
            },
            audio: false
        };

        const stream = await navigator.mediaDevices.getUserMedia(constraints);

        mediaStream = stream;
        video.srcObject = stream;

        if (cameraOverlay) cameraOverlay.classList.add("hidden");

        await video.play();

        // Log diagnóstico
        console.log("[cámara] stream activo:",
            video.videoWidth, "x", video.videoHeight);

        captureBtn.disabled = false;

    } catch (error) {

        console.error("ERROR CAMARA:", error);
        showCameraError(error);

    }

}

startCameraBtn.addEventListener("click", startCamera);

if (switchCameraBtn) {
    switchCameraBtn.addEventListener("click", async () => {
        currentFacingMode =
            currentFacingMode === "environment" ? "user" : "environment";
        await startCamera();
    });
}

// Capturar foto
captureBtn.addEventListener("click", () => {
    const w = video.videoWidth;
    const h = video.videoHeight;

    if (!w || !h) {
        alert("La cámara aún no está lista. Espera 1–2 segundos e intenta de nuevo.");
        return;
    }

    // Log de diagnóstico — pega esto en la consola del navegador para comparar
    // la resolución de la cámara vs una foto subida desde galería
    console.log("[captura] video resolución:", w, "x", h);
    console.log("[captura] devicePixelRatio:", window.devicePixelRatio);

    canvas.width = w;
    canvas.height = h;

    console.log("[captura] canvas:", canvas.width, "x", canvas.height);

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Calidad 1.0: sin recompresión JPEG adicional — la cámara del celular
    // ya aplica su propia compresión internamente. Bajar la calidad aquí
    // degrada bordes y texturas que el modelo necesita para detectar el casco.
    canvas.toBlob(
        (blob) => {
            if (!blob) {
                alert("No se pudo capturar la imagen.");
                return;
            }
            console.log("[captura] blob size:", blob.size,
                "canvas:", canvas.width, "x", canvas.height);
            setPreview(blob, "captura.jpg");
        },
        "image/jpeg",
        1.0
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

        limpiarBoxes();
        window._ultimasDetecciones = null;

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

    setSendingState(true);
    showResultPanel(loadingState);

    try {
        const response = await fetch("/detect", {
            method: "POST",
            body: formData
        });

        const data = await response.json();
        console.log(data);

        if (!response.ok) {
            if (errorTitle) errorTitle.textContent = "No se pudo procesar la imagen";
            if (errorMsg) errorMsg.textContent = data?.mensaje || "Error al procesar la imagen.";
            showResultPanel(errorState);
            return;
        }

        if (data.status === "sin_detecciones") {
            let mensaje = data.mensaje || "No se detectaron objetos/rostros.";
            if (data.resolucion_baja) {
                mensaje += " La imagen tiene baja resolución, lo cual puede dificultar la detección — prueba con una foto más grande y nítida.";
            }
            if (noDetectMsg) noDetectMsg.textContent = mensaje;
            showResultPanel(noDetectState);
            return;
        }

        renderResults(data);
    } catch (error) {
        console.error(error);
        if (errorTitle) errorTitle.textContent = "Error de conexión";
        if (errorMsg) errorMsg.textContent = "No se pudo conectar con la API. Verifica que el servidor esté corriendo.";
        showResultPanel(errorState);
    } finally {
        setSendingState(false);
    }
});

if (retryBtn) {
    retryBtn.addEventListener("click", () => {
        showResultPanel(emptyState);
    });
}

// Estado inicial (por si el HTML viene con disabled)
captureBtn.disabled = true;
sendImageBtn.disabled = true;

// ===============================
// HEALTH CHECK — badge de estado del modelo
// ===============================
async function checkHealth() {
    const badge = document.getElementById("modelStatus");
    const dot = badge ? badge.querySelector(".badge-dot") : null;
    const label = badge ? badge.querySelector(".badge-label") : null;
    if (!badge || !label) return;

    try {
        const res = await fetch("/health");
        const data = await res.json();

        if (res.ok && data.status === "ok") {
            label.textContent = `Modelo listo (${data.etiquetas} clases)`;
            if (dot) dot.classList.add("ok");
        } else {
            label.textContent = "Modelo no disponible";
            if (dot) dot.classList.add("error");
        }
    } catch (err) {
        label.textContent = "Sin conexión con la API";
        if (dot) dot.classList.add("error");
    }
}

checkHealth();