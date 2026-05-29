const API_URL = (window.location.origin.includes('127.0.0.1:8000') || window.location.origin.includes('localhost:8000'))
    ? window.location.origin
    : 'http://127.0.0.1:8000';
let shapChartInstance = null;
let fairnessChartInstance = null;

// ── Auth helpers ──────────────────────────────────────────────
function getUsers() {
    return JSON.parse(localStorage.getItem('fairai_users') || '{}');
}
function saveUsers(users) {
    localStorage.setItem('fairai_users', JSON.stringify(users));
}
function showError(id, msg) {
    const el = document.getElementById(id);
    el.textContent = msg;
    el.style.display = 'block';
}
function hideMsg(id) {
    document.getElementById(id).style.display = 'none';
}

// ── Toggle between Login & Register ──────────────────────────
document.getElementById('show-register').addEventListener('click', (e) => {
    e.preventDefault();
    document.getElementById('login-card').style.display = 'none';
    document.getElementById('register-card').style.display = 'block';
    hideMsg('login-error');
});

document.getElementById('show-login').addEventListener('click', (e) => {
    e.preventDefault();
    document.getElementById('register-card').style.display = 'none';
    document.getElementById('login-card').style.display = 'block';
    hideMsg('register-error');
    hideMsg('register-success');
});

// ── Register Form Submit ──────────────────────────────────────
document.getElementById('register-form').addEventListener('submit', (e) => {
    e.preventDefault();
    hideMsg('register-error');
    hideMsg('register-success');

    const name     = document.getElementById('reg-name').value.trim();
    const username = document.getElementById('reg-username').value.trim();
    const password = document.getElementById('reg-password').value;
    const confirm  = document.getElementById('reg-confirm').value;

    if (!name || !username || !password) {
        showError('register-error', 'All fields are required.');
        return;
    }
    if (password.length < 4) {
        showError('register-error', 'Password must be at least 4 characters.');
        return;
    }
    if (password !== confirm) {
        showError('register-error', 'Passwords do not match.');
        return;
    }

    const users = getUsers();
    if (users[username]) {
        showError('register-error', 'Username already exists. Please choose another.');
        return;
    }

    users[username] = { name, password };
    saveUsers(users);

    // Show success and switch to login after 1.5s
    const successEl = document.getElementById('register-success');
    successEl.textContent = `Account created for "${name}"! Redirecting to login…`;
    successEl.style.display = 'block';
    document.getElementById('register-form').reset();

    setTimeout(() => {
        document.getElementById('register-card').style.display = 'none';
        document.getElementById('login-card').style.display = 'block';
        hideMsg('register-success');
    }, 1800);
});

// ── Login Form Submit ─────────────────────────────────────────
document.getElementById('login-form').addEventListener('submit', (e) => {
    e.preventDefault();
    hideMsg('login-error');

    const username = document.getElementById('username').value.trim();
    const password = document.getElementById('password').value;
    const users = getUsers();

    if (!username || !password) {
        showError('login-error', 'Please enter both username and password.');
        return;
    }

    // Allow if user exists with matching password OR if no users registered yet (demo mode)
    const validUser = users[username] && users[username].password === password;
    const demoMode  = Object.keys(users).length === 0;

    if (validUser || demoMode) {
        document.getElementById('login-container').style.display = 'none';
        document.getElementById('main-app').style.display = 'flex';
    } else {
        showError('login-error', 'Incorrect username or password.');
    }
});

// Navigation
document.getElementById('nav-dashboard').addEventListener('click', (e) => {
    e.preventDefault();
    showView('view-dashboard');
    updateNav('nav-dashboard');
});

document.getElementById('nav-fairness').addEventListener('click', (e) => {
    e.preventDefault();
    showView('view-fairness');
    updateNav('nav-fairness');
    loadFairnessMetrics();
});

document.getElementById('nav-upload').addEventListener('click', (e) => {
    e.preventDefault();
    showView('view-upload');
    updateNav('nav-upload');
});

function showView(viewId) {
    ['view-dashboard', 'view-fairness', 'view-upload'].forEach(id => {
        document.getElementById(id).classList.add('hidden');
    });
    document.getElementById(viewId).classList.remove('hidden');
}

function updateNav(navId) {
    ['nav-dashboard', 'nav-fairness', 'nav-upload'].forEach(id => {
        document.getElementById(id).parentElement.classList.remove('active');
    });
    document.getElementById(navId).parentElement.classList.add('active');
}

// ── CSV Upload Handlers ──────────────────────────────────────
const uploadZone = document.getElementById('upload-zone');
const fileInput  = document.getElementById('csv-file-input');

uploadZone.addEventListener('click', () => fileInput.click());
uploadZone.addEventListener('dragover', (e) => { e.preventDefault(); uploadZone.style.borderColor = 'var(--accent)'; });
uploadZone.addEventListener('dragleave', () => { uploadZone.style.borderColor = 'var(--panel-border)'; });
uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.style.borderColor = 'var(--panel-border)';
    const files = e.dataTransfer.files;
    if (files.length) handleFileUpload(files[0]);
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) handleFileUpload(e.target.files[0]);
});

async function handleFileUpload(file) {
    const errorEl  = document.getElementById('upload-error');
    const statusEl = document.getElementById('upload-status');
    const resultArea = document.getElementById('custom-results-area');
    
    errorEl.style.display = 'none';
    statusEl.style.display = 'block';
    statusEl.textContent = `Analyzing "${file.name}"...`;
    resultArea.style.display = 'none';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(`${API_URL}/upload_csv`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Upload failed");
        }

        const data = await response.json();
        statusEl.textContent = `Analysis complete for ${file.name}`;
        document.getElementById('custom-filename').textContent = file.name;
        
        // Show result area and wait for DOM update before rendering chart
        resultArea.style.display = 'block';
        setTimeout(() => {
            renderCustomAnalysis(data.analysis);
        }, 0);

    } catch (error) {
        statusEl.style.display = 'none';
        errorEl.style.display = 'block';
        errorEl.textContent = error.message;
    }
}

let customChartInstance = null;
function renderCustomAnalysis(analysis) {
    const colorMap = { danger: '#f25c6e', warning: '#fbbf24', success: '#10d9a0' };
    const bgMap    = { danger: 'rgba(242,92,110,0.12)', warning: 'rgba(251,191,36,0.1)', success: 'rgba(16,217,160,0.1)' };
    const c = colorMap[analysis.severity_color];
    const bg = bgMap[analysis.severity_color];

    // Severity Bar
    document.getElementById('custom-severity-bar').innerHTML = `
        <div style="display:flex; align-items:center; gap:1rem; flex-wrap:wrap; padding:1.2rem; background:${bg}; border:1px solid ${c}; border-radius:12px;">
            <div style="font-size:1.6rem; font-weight:800; color:${c};">${analysis.severity} Bias Detected</div>
            <div style="flex:1; min-width:200px;">
                <div style="display:flex; gap:1.2rem; flex-wrap:wrap; font-size:0.85rem; color:var(--text-secondary);">
                    <span>⚧ Gender Gap: <strong style="color:var(--text-primary)">${(analysis.gender_gap*100).toFixed(1)}%</strong></span>
                    <span>🌍 Race Gap: <strong style="color:var(--text-primary)">${(analysis.race_gap*100).toFixed(1)}%</strong></span>
                    <span>📏 DP Diff: <strong style="color:var(--text-primary)">${analysis.dp_diff.toFixed(4)}</strong></span>
                </div>
            </div>
        </div>`;

    // Causes
    document.getElementById('custom-causes-list').innerHTML = analysis.causes.map(cause => `
        <div style="padding:1rem; background:rgba(255,255,255,0.03); border-left:4px solid var(--accent); border-radius:8px; margin-bottom:1rem;">
            <div style="font-weight:700; margin-bottom:0.3rem;">${cause.icon} ${cause.type}</div>
            <p style="font-size:0.9rem; color:var(--text-secondary);">${cause.detail}</p>
        </div>`).join('');

    // Chart
    const ctx = document.getElementById('customFairnessChart').getContext('2d');
    if (customChartInstance) customChartInstance.destroy();
    customChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Demographic Parity Difference', 'Equal Opportunity Difference'],
            datasets: [
                { 
                    label: 'Dataset Baseline (Historical)', 
                    data: [analysis.dataset_bias, 0], 
                    backgroundColor: 'rgba(251, 191, 36, 0.8)',
                    minBarLength: 5 
                },
                { 
                    label: 'Standard Model (Current)', 
                    data: [analysis.standard.dp, analysis.standard.eo], 
                    backgroundColor: 'rgba(239, 68, 68, 0.8)',
                    minBarLength: 5
                },
                { 
                    label: 'Mitigated Model (Proposed)', 
                    data: [analysis.mitigated.dp, analysis.mitigated.eo], 
                    backgroundColor: 'rgba(16, 185, 129, 0.8)',
                    minBarLength: 5
                }
            ]
        },
        options: { 
            responsive: true, 
            maintainAspectRatio: false, 
            plugins: { 
                legend: { labels: { color: '#f8fafc' } } 
            },
            scales: { 
                y: { 
                    beginAtZero: true, 
                    grid: { color: 'rgba(255,255,255,0.1)' }, 
                    ticks: { color: '#94a3b8' } 
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#94a3b8' }
                }
            }
        }
    });
}



// Toggle Model
const toggleInput = document.getElementById('mitigation-toggle');
const toggleLabel = document.getElementById('toggle-label');

toggleInput.addEventListener('change', () => {
    if(toggleInput.checked) {
        toggleLabel.textContent = "Mitigated Model";
        toggleLabel.style.color = "var(--success)";
    } else {
        toggleLabel.textContent = "Standard Model";
        toggleLabel.style.color = "var(--danger)";
    }
    // Retrigger prediction if a result already exists
    if(!document.getElementById('outcome-display').classList.contains('pending')) {
        document.getElementById('prediction-form').dispatchEvent(new Event('submit'));
    }
});

// Prediction Form Submit
document.getElementById('prediction-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const useMitigated = toggleInput.checked;
    
    const formData = {
        age: parseInt(document.getElementById('age').value),
        workclass: document.getElementById('workclass').value,
        fnlwgt: parseFloat(document.getElementById('fnlwgt').value),
        education: document.getElementById('education').value,
        education_num: parseFloat(document.getElementById('education_num').value),
        marital_status: document.getElementById('marital_status').value,
        occupation: document.getElementById('occupation').value,
        relationship: document.getElementById('relationship').value,
        race: document.getElementById('race').value,
        sex: document.getElementById('sex').value,
        capital_gain: parseFloat(document.getElementById('capital_gain').value),
        capital_loss: parseFloat(document.getElementById('capital_loss').value),
        hours_per_week: parseFloat(document.getElementById('hours_per_week').value),
        native_country: document.getElementById('native_country').value,
        use_mitigated_model: useMitigated
    };

    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });

        if(!response.ok) throw new Error("API Request Failed");

        const data = await response.json();
        
        // Update Outcome
        const outcomeEl = document.getElementById('outcome-display');
        outcomeEl.textContent = data.prediction;
        outcomeEl.className = 'outcome ' + (data.prediction.includes("Approved") ? 'approved' : 'rejected');
        
        document.getElementById('probability-display').textContent = `Confidence: ${(data.probability * 100).toFixed(1)}%`;

        // Prediction Bias Insight
        const insightEl = document.getElementById('prediction-bias-insight');
        if (!useMitigated) {
            insightEl.style.display = 'block';
            insightEl.innerHTML = `
                <h4><i class="fas fa-exclamation-triangle"></i> Why this might be biased?</h4>
                <p>You are using the <strong>Standard Model</strong>. This model was trained on historical data where certain groups (like Females or Non-White applicants) had lower approval rates. SHAP analysis shows that features like 'Sex' and 'Race' are influencing this decision.</p>
                <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(251, 191, 36, 0.2);">
                    <strong style="color: var(--success);">Solution:</strong> Switch to the <strong>"Mitigated Model"</strong> (toggle in sidebar) to remove sensitive attributes and ensure a blind, fairer decision.
                </div>
            `;
        } else {
            insightEl.style.display = 'block';
            insightEl.className = 'bias-alert safe';
            insightEl.innerHTML = `
                <h4><i class="fas fa-check-circle"></i> Fair Decision Active</h4>
                <p>The <strong>Mitigated Model</strong> is currently active. It has been mathematically blinded to 'Sex' and 'Race', ensuring these sensitive attributes did not influence this specific outcome.</p>
            `;
        }

        // Render SHAP chart
        renderShapChart(data.shap_values);
        
    } catch (error) {
        console.error(error);
        alert("Error connecting to backend. Make sure FastAPI server is running.");
    }
});

// Download Report
document.getElementById('download-report').addEventListener('click', async () => {
    try {
        const response = await fetch(`${API_URL}/report`);
        const data = await response.json();
        
        const blob = new Blob([data.report], { type: 'text/plain' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'Fairness_Report.txt';
        a.click();
    } catch (error) {
        alert("Failed to download report.");
    }
});


// Render SHAP Chart
function renderShapChart(shapData) {
    const ctx = document.getElementById('shapChart').getContext('2d');
    
    const labels = shapData.map(d => d.feature.replace('num__', '').replace('cat__', ''));
    const values = shapData.map(d => d.value);
    const colors = values.map(v => v > 0 ? 'rgba(239, 68, 68, 0.8)' : 'rgba(59, 130, 246, 0.8)');

    if(shapChartInstance) shapChartInstance.destroy();

    shapChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'SHAP Value (Impact on Output)',
                data: values,
                backgroundColor: colors,
                borderWidth: 1
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.1)' },
                    ticks: { color: '#94a3b8' }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#94a3b8' }
                }
            }
        }
    });
}

// Load Fairness Metrics
async function loadFairnessMetrics() {
    try {
        const response = await fetch(`${API_URL}/metrics`);
        const metrics = await response.json();
        
        const ctx = document.getElementById('fairnessChart').getContext('2d');
        
        if(fairnessChartInstance) fairnessChartInstance.destroy();

        fairnessChartInstance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Demographic Parity Difference', 'Equal Opportunity Difference'],
                datasets: [
                    {
                        label: 'Standard Model (Biased)',
                        data: [metrics.standard.demographic_parity_diff, metrics.standard.equal_opportunity_diff],
                        backgroundColor: 'rgba(239, 68, 68, 0.8)',
                        minBarLength: 5
                    },
                    {
                        label: 'Mitigated Model',
                        data: [metrics.mitigated.demographic_parity_diff, metrics.mitigated.equal_opportunity_diff],
                        backgroundColor: 'rgba(16, 185, 129, 0.8)',
                        minBarLength: 5
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#f8fafc' } }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: { color: 'rgba(255,255,255,0.1)' },
                        ticks: { color: '#94a3b8' }
                    },
                    x: {
                        grid: { display: false },
                        ticks: { color: '#94a3b8' }
                    }
                }
            }
        });

        // Bias alert banner
        const alertEl = document.getElementById('bias-alert');
        const standardDP = Math.abs(metrics.standard.demographic_parity_diff);
        if (standardDP > 0.1) {
            alertEl.className = 'bias-alert';
            alertEl.innerHTML = `<h4>⚠️ Significant Bias Detected</h4>
                <p>The Standard Model has a Demographic Parity Difference of <strong>${standardDP.toFixed(4)}</strong>.
                Scroll down to see root causes and recommended solutions.</p>`;
        } else {
            alertEl.className = 'bias-alert safe';
            alertEl.innerHTML = `<h4>✅ Model is Fair</h4>
                <p>The Standard Model's Demographic Parity Difference is within acceptable limits.</p>`;
        }

        // Render bias analysis panels
        if (metrics.bias_analysis) {
            renderBiasAnalysis(metrics.bias_analysis);
        }

    } catch (error) {
        console.error(error);
    }
}

function renderBiasAnalysis(analysis) {
    const colorMap = { danger: '#f25c6e', warning: '#fbbf24', success: '#10d9a0' };
    const bgMap    = { danger: 'rgba(242,92,110,0.12)', warning: 'rgba(251,191,36,0.1)', success: 'rgba(16,217,160,0.1)' };
    const c = colorMap[analysis.severity_color];
    const bg = bgMap[analysis.severity_color];

    // ── Severity bar ──────────────────────────────────────────
    document.getElementById('bias-severity-bar').innerHTML = `
        <div style="display:flex; align-items:center; gap:1rem; flex-wrap:wrap; padding:1rem; background:${bg};
             border:1px solid ${c}; border-radius:10px;">
            <div style="font-size:1.8rem; font-weight:800; color:${c};">${analysis.severity} Bias</div>
            <div style="flex:1; min-width:200px;">
                <div style="display:flex; gap:1.5rem; flex-wrap:wrap; font-size:0.88rem; color:var(--text-secondary);">
                    <span>📐 Demographic Parity Gap: <strong style="color:var(--text-primary)">${(analysis.dp_diff*100).toFixed(2)}%</strong></span>
                    <span>🎯 Equal Opportunity Gap: <strong style="color:var(--text-primary)">${(analysis.eo_diff*100).toFixed(2)}%</strong></span>
                    <span>⚧ Gender Gap: <strong style="color:var(--text-primary)">${(analysis.gender_gap*100).toFixed(1)}%</strong></span>
                    <span>🌍 Race Gap: <strong style="color:var(--text-primary)">${(analysis.race_gap*100).toFixed(1)}%</strong></span>
                </div>
            </div>
        </div>`;

    // ── Causes ────────────────────────────────────────────────
    const causeColors = ['rgba(242,92,110,0.1)', 'rgba(251,191,36,0.08)', 'rgba(79,142,247,0.1)', 'rgba(139,92,246,0.1)'];
    const causeBorder = ['rgba(242,92,110,0.4)', 'rgba(251,191,36,0.4)', 'rgba(79,142,247,0.4)', 'rgba(139,92,246,0.4)'];
    document.getElementById('bias-causes-list').innerHTML = analysis.causes.map((cause, i) => `
        <div style="padding:1rem 1.2rem; background:${causeColors[i%4]}; border-left:4px solid ${causeBorder[i%4]};
             border-radius:8px; margin-bottom:1rem;">
            <div style="font-size:1.1rem; font-weight:700; margin-bottom:0.4rem;">
                ${cause.icon} ${cause.type}
            </div>
            <p style="font-size:0.9rem; color:var(--text-secondary); line-height:1.6;">${cause.detail}</p>
        </div>`).join('');

    // ── Solutions ─────────────────────────────────────────────
    const tagColors  = { 'Active via Toggle':'#10d9a0', 'Recommended':'#4f8ef7', 'Post-processing':'#fbbf24', 'Long-term Fix':'#a78bfa' };
    const effColors  = { 'Very High':'#10d9a0', 'High':'#4f8ef7', 'Medium':'#fbbf24', 'Low':'#f25c6e' };
    document.getElementById('bias-solutions-list').innerHTML = analysis.solutions.map(s => `
        <div style="padding:1.2rem; background:rgba(79,142,247,0.06); border:1px solid rgba(79,142,247,0.15);
             border-radius:10px; margin-bottom:1rem; display:flex; gap:1rem; align-items:flex-start;">
            <div style="font-size:1.5rem; font-weight:800; color:var(--accent); min-width:32px; padding-top:2px;">#${s.rank}</div>
            <div style="flex:1;">
                <div style="display:flex; align-items:center; gap:0.6rem; flex-wrap:wrap; margin-bottom:0.5rem;">
                    <span style="font-size:1rem; font-weight:700;">${s.name}</span>
                    <span style="font-size:0.75rem; padding:2px 8px; border-radius:20px; background:${tagColors[s.tag]||'#4f8ef7'}22;
                          color:${tagColors[s.tag]||'#4f8ef7'}; border:1px solid ${tagColors[s.tag]||'#4f8ef7'}55;">${s.tag}</span>
                    <span style="font-size:0.75rem; padding:2px 8px; border-radius:20px; background:rgba(255,255,255,0.05);
                          color:${effColors[s.effectiveness]||'#fff'};">⚡ ${s.effectiveness} Effectiveness</span>
                    <span style="font-size:0.75rem; padding:2px 8px; border-radius:20px; background:rgba(255,255,255,0.05);
                          color:var(--text-secondary);">${s.type}</span>
                </div>
                <p style="font-size:0.9rem; color:var(--text-secondary); line-height:1.6; margin-bottom:0.4rem;">${s.description}</p>
                <p style="font-size:0.82rem; color:rgba(251,191,36,0.8);">⚠ Trade-off: ${s.tradeoff}</p>
            </div>
        </div>`).join('');

    document.getElementById('bias-causes-card').style.display   = 'block';
    document.getElementById('bias-solutions-card').style.display = 'block';
}
