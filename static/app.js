let ecgChart = null;

async function loadSample(type) {
    showLoading(true);
    try {
        const response = await fetch(`/api/sample/${type}`);
        const sample = await response.json();
        
        // Now predict with this sample
        await predict(sample.data);
    } catch (error) {
        console.error("Error loading sample:", error);
        alert("Erreur lors du chargement de l'échantillon.");
    } finally {
        showLoading(false);
    }
}

async function predict(data) {
    showLoading(true);
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ data: data })
        });
        const result = await response.json();
        updateUI(result);
    } catch (error) {
        console.error("Error predicting:", error);
        alert("Erreur lors de l'analyse du signal.");
    } finally {
        showLoading(false);
    }
}

async function handleFileUpload(file) {
    showLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
        let endpoint = '/api/predict/image';
        // If it's a CSV, we should ideally parse it, but for simplicity we'll handle images primarily 
        // as the API handles image uploads. For CSV, the current API endpoint is /api/predict with JSON.
        
        if (file.name.endsWith('.csv')) {
            const text = await file.text();
            const data = text.split(',').map(Number).slice(0, 187);
            await predict(data);
            return;
        }

        const response = await fetch(endpoint, {
            method: 'POST',
            body: formData
        });
        const result = await response.json();
        updateUI(result);
    } catch (error) {
        console.error("Error uploading file:", error);
        alert("Erreur lors du traitement du fichier.");
    } finally {
        showLoading(false);
    }
}

function updateUI(result) {
    // Update Metrics
    document.getElementById('mae-value').textContent = result.reconstruction_error.toFixed(5);
    document.getElementById('threshold-value').textContent = result.threshold.toFixed(5);
    
    // Update Status Badge
    const container = document.getElementById('status-container');
    const isAnomaly = result.is_anomalous;
    container.innerHTML = `<span class="status-badge ${isAnomaly ? 'status-anomaly' : 'status-normal'}">${result.classification}</span>`;
    
    // Update Image if available
    if (result.visualization_url) {
        const img = document.getElementById('viz-image');
        img.src = result.visualization_url + "?t=" + new Date().getTime(); // Anti-cache
        document.getElementById('image-result').style.display = 'block';
    }

    // Update Chart
    renderChart(result.original_signal, result.reconstructed_signal);
}

function renderChart(original, reconstructed) {
    const ctx = document.getElementById('ecg-canvas').getContext('2d');
    
    if (ecgChart) {
        ecgChart.destroy();
    }

    const labels = Array.from({length: original.length}, (_, i) => i);

    ecgChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Signal Original',
                    data: original,
                    borderColor: '#00d2ff',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.4
                },
                {
                    label: 'Reconstruction',
                    data: reconstructed,
                    borderColor: '#9d50bb',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    pointRadius: 0,
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    grid: { color: '#222' },
                    ticks: { color: '#8b949e' }
                },
                x: {
                    grid: { display: false },
                    ticks: { display: false }
                }
            },
            plugins: {
                legend: {
                    labels: { color: '#e6edf3', font: { weight: 'bold' } }
                }
            },
            interaction: {
                intersect: false,
                mode: 'index'
            }
        }
    });
}

function showLoading(show) {
    document.getElementById('loading').style.display = show ? 'flex' : 'none';
}

// Event Listeners
document.getElementById('file-input').addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileUpload(e.target.files[0]);
    }
});

const dropZone = document.getElementById('drop-zone');
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = '#00d2ff';
});

dropZone.addEventListener('dragleave', () => {
    dropZone.style.borderColor = '#30363d';
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = '#30363d';
    if (e.dataTransfer.files.length > 0) {
        handleFileUpload(e.dataTransfer.files[0]);
    }
});

// Initial Load
window.onload = () => {
    loadSample('normal');
};
