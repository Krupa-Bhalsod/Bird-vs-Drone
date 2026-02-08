// JavaScript for upload and webcam capture
document.addEventListener('DOMContentLoaded', () => {
  const fileInput = document.getElementById('file-input');
  const predictBtn = document.getElementById('predict-btn');
  const uploadResult = document.getElementById('upload-result');
  const uploadLabel = document.getElementById('upload-label');
  const uploadProb = document.getElementById('upload-prob');
  const uploadPreview = document.getElementById('upload-preview');
  const confidenceFill = document.getElementById('confidence-fill');

  predictBtn.addEventListener('click', async () => {
    const f = fileInput.files[0];
    if (!f) return alert('Choose an image');
    const threshold = document.getElementById('threshold-input').value || 0.5;
    const fd = new FormData();
    fd.append('file', f);
    fd.append('threshold', threshold);
    const res = await fetch('/predict', { method: 'POST', body: fd });
    const json = await res.json();
    if (json.error) return alert(json.error);

    uploadResult.style.display = 'block';
    uploadLabel.textContent = json.label;
    uploadProb.textContent = 'Confidence: ' + (json.prob * 100).toFixed(2) + '%';
    // preview
    const url = URL.createObjectURL(f);
    uploadPreview.src = url;

    // glowing label
    uploadLabel.classList.remove('glow-label', 'drone', 'bird');
    uploadLabel.classList.add('glow-label', json.label.toLowerCase());

    // animate confidence bar
    if (confidenceFill) {
      confidenceFill.style.width = `${(json.prob * 100).toFixed(1)}%`;
    }

    // small animation
    uploadResult.classList.add('pop');
    setTimeout(() => uploadResult.classList.remove('pop'), 900);
  });

  // Webcam
  const video = document.getElementById('video');
  const startBtn = document.getElementById('start-btn');
  const stopBtn = document.getElementById('stop-btn');
  const webcamResult = document.getElementById('webcam-result');
  let stream = null;
  let intervalId = null;

  startBtn.addEventListener('click', async () => {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      video.srcObject = stream;
      video.play();
      // start sending frames every 700ms
      intervalId = setInterval(async () => {
        const canvas = document.createElement('canvas');
        canvas.width = Math.max(160, Math.floor(video.videoWidth / 2));
        canvas.height = Math.max(120, Math.floor(video.videoHeight / 2));
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const data = canvas.toDataURL('image/jpeg', 0.7);
        const threshold = document.getElementById('threshold-input').value || 0.5;
        try {
          const resp = await fetch('/webcam_predict', {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ image: data, threshold: threshold })
          });
          const j = await resp.json();
          if (!j.error) {
            webcamResult.innerText = `${j.label} (${(j.prob*100).toFixed(1)}%)`;
            webcamResult.classList.remove('glow-label','drone','bird');
            webcamResult.classList.add('glow-label', j.label.toLowerCase());
          }
        } catch (e) {
          console.error(e);
        }
      }, 700);
    } catch (e) {
      alert('Could not open webcam: ' + e.message);
    }
  });

  stopBtn.addEventListener('click', () => {
    if (intervalId) clearInterval(intervalId);
    if (stream) {
      stream.getTracks().forEach(t => t.stop());
      stream = null;
    }
    video.srcObject = null;
    webcamResult.innerText = 'Stopped';
    webcamResult.classList.remove('glow-label','drone','bird');
  });

  // initialize particles (if tsParticles is available)
  try {
    if (window.tsParticles) {
      tsParticles.load('tsparticles', {
        fpsLimit: 60,
        particles: {
          number: { value: 40, density: { enable: true, area: 800 } },
          color: { value: ['#ffffff', '#aee1ff', '#c8ffd6'] },
          shape: { type: 'circle' },
          opacity: { value: 0.6, random: { enable: true, minimumValue: 0.2 } },
          size: { value: { min: 1, max: 6 } },
          move: { enable: true, speed: 0.8, direction: 'none', random: true, outModes: { default: 'out' } },
        },
        interactivity: { events: { onHover: { enable: true, mode: 'repulse' }, onClick: { enable: true, mode: 'push' } }, modes: { repulse: { distance: 100 } } },
        detectRetina: true,
      });
    }
  } catch (e) {
    console.warn('tsParticles init failed', e);
  }

});
