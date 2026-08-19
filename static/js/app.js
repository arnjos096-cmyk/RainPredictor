/**
 * RainPredictor AI — Frontend Logic & Interactive Meteorological Engine
 */

document.addEventListener('DOMContentLoaded', () => {
    // State management
    const state = {
        mode: 'quick', // 'quick' | 'history' | 'dataset'
        scenario: 'thunderstorm',
        scenarios: {},
        currentWeather: {
            temperature: 21.5,
            humidity: 94.0,
            pressure: 993.0,
            wind_speed: 42.0,
            wind_direction: 225.0,
            soil_moisture: 65.0,
            solar_radiation: 40.0,
            cloud_cover: 98.0,
            dew_point: 20.3,
            evapotranspiration: 0.05,
            rainfall_mm: 1.2
        },
        pressureTrend: 'falling',
        hourOfDay: 14,
        sequence24h: [],
        charts: {
            timeline: null,
            forecast: null
        },
        debounceTimer: null
    };

    // DOM Elements
    const elements = {
        statusPill: document.getElementById('statusPill'),
        statusText: document.getElementById('statusText'),
        scenarioChips: document.getElementById('scenarioChips'),
        gaugeValue: document.getElementById('gaugeValue'),
        gaugeProgress: document.getElementById('gaugeProgress'),
        riskBadge: document.getElementById('riskBadge'),
        riskIcon: document.getElementById('riskIcon'),
        riskLabel: document.getElementById('riskLabel'),
        riskDescription: document.getElementById('riskDescription'),
        probPercent: document.getElementById('probPercent'),
        probFill: document.getElementById('probFill'),
        lastInferenceTime: document.getElementById('lastInferenceTime'),
        insightsList: document.getElementById('insightsList'),
        paramHour: document.getElementById('param_hour_of_day'),
        valHour: document.getElementById('val_hour_of_day'),
        pressureTrendGroup: document.getElementById('pressureTrendGroup'),
        btnRandomize: document.getElementById('btnRandomize'),
        btnReset: document.getElementById('btnReset'),
        quickDeckView: document.getElementById('quickDeckView'),
        matrixDeckView: document.getElementById('matrixDeckView'),
        datasetDeckView: document.getElementById('datasetDeckView'),
        btnModeQuick: document.getElementById('btnModeQuick'),
        btnModeHistory: document.getElementById('btnModeHistory'),
        btnModeDataset: document.getElementById('btnModeDataset'),
        matrixTableBody: document.getElementById('matrixTableBody'),
        datasetScrubber: document.getElementById('datasetScrubber'),
        datasetIdxLabel: document.getElementById('datasetIdxLabel'),
        metaDateStart: document.getElementById('metaDateStart'),
        metaDateEnd: document.getElementById('metaDateEnd'),
        metaActualRain: document.getElementById('metaActualRain'),
        btnLoadSlice: document.getElementById('btnLoadSlice'),
        btnRandomSlice: document.getElementById('btnRandomSlice'),
        panelSubtext: document.getElementById('panelSubtext')
    };

    // Parameter input keys
    const paramKeys = [
        'temperature', 'humidity', 'pressure', 'wind_speed',
        'wind_direction', 'soil_moisture', 'solar_radiation',
        'cloud_cover', 'dew_point', 'evapotranspiration', 'rainfall_mm'
    ];

    // --- 1. Initialize Application ---
    async function init() {
        initWeatherCanvas();
        initCharts();
        setupEventListeners();
        await fetchStatus();
        await fetchScenarios();
        synthesize24hFromCurrent();
        runInference();
    }

    // --- 2. API Communication ---
    async function fetchStatus() {
        try {
            const res = await fetch('/api/status');
            const data = await res.json();
            if (data.model_loaded) {
                elements.statusPill.className = 'status-pill status-online';
                elements.statusText.textContent = `Neural Model Ready (${data.device.toUpperCase()})`;
            }
        } catch (err) {
            elements.statusPill.className = 'status-pill status-offline';
            elements.statusText.textContent = 'API Offline';
            console.error('Status fetch error:', err);
        }
    }

    async function fetchScenarios() {
        try {
            const res = await fetch('/api/scenarios');
            const scenarios = await res.json();
            state.scenarios = {};
            scenarios.forEach(s => {
                state.scenarios[s.id] = s;
            });
        } catch (err) {
            console.error('Scenarios fetch error:', err);
        }
    }

    async function runInference() {
        elements.lastInferenceTime.textContent = 'Running...';
        
        try {
            let res;
            if (state.mode === 'quick') {
                res = await fetch('/api/predict-single', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        current: state.currentWeather,
                        pressure_trend: state.pressureTrend,
                        hour_of_day: state.hourOfDay
                    })
                });
            } else {
                res = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        sequence: state.sequence24h
                    })
                });
            }

            const data = await res.json();
            updatePredictionUI(data);
            
            // Also fetch 12-hour forward projection
            runForwardForecast();
            updateTimelineChart();
        } catch (err) {
            console.error('Inference error:', err);
            elements.lastInferenceTime.textContent = 'Error';
        }
    }

    async function runForwardForecast() {
        try {
            const seqToSend = state.mode === 'quick' ? state.sequence24h : state.sequence24h;
            if (!seqToSend || seqToSend.length !== 24) return;

            const res = await fetch('/api/predict-forecast', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    sequence: seqToSend,
                    steps: 12
                })
            });
            const data = await res.json();
            updateForecastChart(data.forecast);
        } catch (err) {
            console.error('Forecast error:', err);
        }
    }

    // --- 3. UI Update Helpers ---
    function updatePredictionUI(data) {
        const mm = data.predicted_rainfall_mm;
        const cat = data.category;

        // Animate gauge value
        elements.gaugeValue.textContent = mm.toFixed(2);
        
        // Gauge stroke calculation (max 25mm represents 100% circle)
        const circumference = 502; // 2 * pi * 80
        const percentage = Math.min(1.0, mm / 20.0);
        const offset = circumference - (percentage * (circumference * 0.75));
        elements.gaugeProgress.style.strokeDashoffset = offset;
        elements.gaugeProgress.style.stroke = cat.color;

        // Risk Badge
        elements.riskBadge.style.color = cat.color;
        elements.riskBadge.style.borderColor = cat.color + '66';
        elements.riskBadge.style.backgroundColor = cat.color + '22';
        elements.riskLabel.textContent = cat.tier;
        elements.riskDescription.textContent = cat.summary;

        // Icons
        if (cat.code === 'DRY') elements.riskIcon.className = 'fa-solid fa-sun';
        else if (cat.code === 'LIGHT') elements.riskIcon.className = 'fa-solid fa-cloud-rain';
        else if (cat.code === 'MODERATE') elements.riskIcon.className = 'fa-solid fa-cloud-showers-heavy';
        else elements.riskIcon.className = 'fa-solid fa-bolt-lightning';

        // Probability
        elements.probPercent.textContent = `${cat.probability.toFixed(0)}%`;
        elements.probFill.style.width = `${cat.probability}%`;

        // Timestamp
        const now = new Date();
        elements.lastInferenceTime.textContent = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

        // Insights
        if (data.insights && data.insights.length > 0) {
            elements.insightsList.innerHTML = data.insights.map(item => `
                <div class="insight-item">
                    <i class="fa-solid ${item.icon} insight-icon" style="color: ${item.type === 'alert' || item.type === 'warning' ? '#ef4444' : '#10b981'}"></i>
                    <div>
                        <h4 class="insight-title">${item.title}</h4>
                        <p class="insight-detail">${item.detail}</p>
                    </div>
                </div>
            `).join('');
        } else {
            elements.insightsList.innerHTML = `
                <div class="insight-item">
                    <i class="fa-solid fa-circle-check insight-icon text-emerald"></i>
                    <div>
                        <h4 class="insight-title">Atmospheric Equilibrium</h4>
                        <p class="insight-detail">Parameters are within stable baseline operational ranges.</p>
                    </div>
                </div>
            `;
        }
    }

    function synthesize24hFromCurrent() {
        const curr = state.currentWeather;
        const trend = state.pressureTrend;
        const hour = state.hourOfDay;
        
        state.sequence24h = [];
        for (let step = 0; step < 24; step++) {
            const h_offset = 23 - step;
            const past_hour = (hour - h_offset + 24) % 24;
            
            const diurnal_temp = 3.0 * Math.sin(2 * Math.PI * (past_hour - 8) / 24);
            const curr_diurnal = 3.0 * Math.sin(2 * Math.PI * (hour - 8) / 24);
            const t_val = curr.temperature + (diurnal_temp - curr_diurnal);
            
            let p_val = curr.pressure;
            if (trend === 'falling') p_val = curr.pressure + (h_offset * 0.25);
            else if (trend === 'rising') p_val = curr.pressure - (h_offset * 0.25);
            else p_val = curr.pressure + Math.sin(step) * 0.3;
            
            const factor = (24 - h_offset) / 24.0;
            let c_val = curr.cloud_cover;
            let hum_val = curr.humidity;
            let w_val = curr.wind_speed;
            
            if (trend === 'falling') {
                c_val = Math.max(10.0, curr.cloud_cover * (0.4 + 0.6 * factor));
                hum_val = Math.min(100.0, curr.humidity * (0.6 + 0.4 * factor));
                w_val = Math.max(5.0, curr.wind_speed * (0.5 + 0.5 * factor));
            } else {
                c_val = Math.max(5.0, curr.cloud_cover * (1.2 - 0.2 * factor));
                hum_val = Math.max(20.0, curr.humidity * (1.1 - 0.1 * factor));
                w_val = Math.max(5.0, curr.wind_speed * (1.1 - 0.1 * factor));
            }
            
            const daylight = (past_hour > 6) && (past_hour < 18);
            let rad_val = 0;
            if (daylight) {
                rad_val = 800.0 * Math.sin(Math.PI * (past_hour - 6) / 12) * (1.0 - 0.7 * (c_val / 100.0));
            }
            
            const dew_val = t_val - ((100.0 - hum_val) / 5.0);
            const evapo_val = Math.max(0.0, (rad_val * 0.001) + (Math.max(0.0, t_val) * 0.01) - (hum_val * 0.001));
            
            let rain_val = 0.0;
            if (step === 23) {
                rain_val = curr.rainfall_mm;
            } else if (trend === 'falling' && c_val > 80 && step > 18) {
                rain_val = Math.random() * 0.4;
            }
            
            state.sequence24h.push({
                temperature: Number(t_val.toFixed(1)),
                humidity: Number(hum_val.toFixed(1)),
                pressure: Number(p_val.toFixed(1)),
                wind_speed: Number(w_val.toFixed(1)),
                wind_direction: curr.wind_direction,
                soil_moisture: Number(curr.soil_moisture.toFixed(1)),
                solar_radiation: Number(rad_val.toFixed(1)),
                cloud_cover: Number(c_val.toFixed(1)),
                dew_point: Number(dew_val.toFixed(1)),
                evapotranspiration: Number(evapo_val.toFixed(2)),
                rainfall_mm: Number(rain_val.toFixed(2))
            });
        }
        
        renderMatrixTable();
    }

    function renderMatrixTable() {
        if (!elements.matrixTableBody) return;
        elements.matrixTableBody.innerHTML = state.sequence24h.map((row, idx) => `
            <tr>
                <td><strong>t-${23 - idx}h</strong></td>
                <td><input type="number" step="0.5" value="${row.temperature}" data-idx="${idx}" data-field="temperature"></td>
                <td><input type="number" step="1" value="${row.humidity}" data-idx="${idx}" data-field="humidity"></td>
                <td><input type="number" step="0.5" value="${row.pressure}" data-idx="${idx}" data-field="pressure"></td>
                <td><input type="number" step="1" value="${row.wind_speed}" data-idx="${idx}" data-field="wind_speed"></td>
                <td><input type="number" step="1" value="${row.cloud_cover}" data-idx="${idx}" data-field="cloud_cover"></td>
                <td><input type="number" step="10" value="${row.solar_radiation}" data-idx="${idx}" data-field="solar_radiation"></td>
                <td><input type="number" step="0.1" value="${row.rainfall_mm}" data-idx="${idx}" data-field="rainfall_mm"></td>
            </tr>
        `).join('');

        // Attach change listeners to matrix inputs
        elements.matrixTableBody.querySelectorAll('input').forEach(inp => {
            inp.addEventListener('input', (e) => {
                const idx = parseInt(e.target.dataset.idx);
                const field = e.target.dataset.field;
                state.sequence24h[idx][field] = parseFloat(e.target.value) || 0;
                debounceInference();
            });
        });
    }

    function syncSlidersFromState() {
        paramKeys.forEach(k => {
            const slider = document.getElementById(`param_${k}`);
            const valLabel = document.getElementById(`val_${k}`);
            if (slider && valLabel) {
                slider.value = state.currentWeather[k];
                const unit = k === 'temperature' || k === 'dew_point' ? '°C' :
                             k === 'humidity' || k === 'cloud_cover' || k === 'soil_moisture' ? '%' :
                             k === 'pressure' ? 'hPa' :
                             k === 'wind_speed' ? 'km/h' :
                             k === 'wind_direction' ? '°' :
                             k === 'solar_radiation' ? 'W/m²' :
                             k === 'evapotranspiration' ? 'mm/h' : 'mm';
                valLabel.innerHTML = `${state.currentWeather[k]} <span class="unit">${unit}</span>`;
            }
        });
    }

    function debounceInference() {
        clearTimeout(state.debounceTimer);
        state.debounceTimer = setTimeout(() => {
            runInference();
        }, 150);
    }

    // --- 4. Event Listeners ---
    function setupEventListeners() {
        // Slider inputs
        paramKeys.forEach(k => {
            const slider = document.getElementById(`param_${k}`);
            const valLabel = document.getElementById(`val_${k}`);
            if (slider) {
                slider.addEventListener('input', (e) => {
                    const val = parseFloat(e.target.value);
                    state.currentWeather[k] = val;
                    const unit = k === 'temperature' || k === 'dew_point' ? '°C' :
                                 k === 'humidity' || k === 'cloud_cover' || k === 'soil_moisture' ? '%' :
                                 k === 'pressure' ? 'hPa' :
                                 k === 'wind_speed' ? 'km/h' :
                                 k === 'wind_direction' ? '°' :
                                 k === 'solar_radiation' ? 'W/m²' :
                                 k === 'evapotranspiration' ? 'mm/h' : 'mm';
                    valLabel.innerHTML = `${val} <span class="unit">${unit}</span>`;
                    
                    // Re-synthesize 24h timeline
                    synthesize24hFromCurrent();
                    debounceInference();
                });
            }
        });

        // Hour of Day slider
        if (elements.paramHour) {
            elements.paramHour.addEventListener('input', (e) => {
                state.hourOfDay = parseInt(e.target.value);
                const isDay = state.hourOfDay >= 6 && state.hourOfDay <= 18;
                elements.valHour.textContent = `${state.hourOfDay.toString().padStart(2, '0')}:00 (${isDay ? 'Daylight' : 'Night'})`;
                synthesize24hFromCurrent();
                debounceInference();
            });
        }

        // Pressure Trend Buttons
        if (elements.pressureTrendGroup) {
            elements.pressureTrendGroup.querySelectorAll('.trend-btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    elements.pressureTrendGroup.querySelectorAll('.trend-btn').forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');
                    state.pressureTrend = btn.dataset.trend;
                    synthesize24hFromCurrent();
                    debounceInference();
                });
            });
        }

        // Preset Scenario Chips
        if (elements.scenarioChips) {
            elements.scenarioChips.querySelectorAll('.scenario-chip').forEach(chip => {
                chip.addEventListener('click', () => {
                    elements.scenarioChips.querySelectorAll('.scenario-chip').forEach(c => c.classList.remove('active'));
                    chip.classList.add('active');
                    const scId = chip.dataset.scenario;
                    loadScenario(scId);
                });
            });
        }

        // Mode Switching
        const modeButtons = [elements.btnModeQuick, elements.btnModeHistory, elements.btnModeDataset];
        modeButtons.forEach(btn => {
            if (!btn) return;
            btn.addEventListener('click', () => {
                modeButtons.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                state.mode = btn.dataset.mode;

                elements.quickDeckView.classList.add('hidden');
                elements.matrixDeckView.classList.add('hidden');
                elements.datasetDeckView.classList.add('hidden');

                if (state.mode === 'quick') {
                    elements.quickDeckView.classList.remove('hidden');
                    elements.panelSubtext.textContent = 'Tune live weather variables to trigger real-time neural inference.';
                } else if (state.mode === 'history') {
                    elements.matrixDeckView.classList.remove('hidden');
                    elements.panelSubtext.textContent = 'Inspect and modify each of the 24 hourly sequence timesteps directly.';
                    renderMatrixTable();
                } else if (state.mode === 'dataset') {
                    elements.datasetDeckView.classList.remove('hidden');
                    elements.panelSubtext.textContent = 'Scrub through the 5-year dataset and validate LSTM against true recorded rainfall.';
                    loadHistoricalSlice(parseInt(elements.datasetScrubber.value));
                }

                runInference();
            });
        });

        // Dataset scrubber
        if (elements.datasetScrubber) {
            elements.datasetScrubber.addEventListener('input', (e) => {
                const idx = parseInt(e.target.value);
                elements.datasetIdxLabel.textContent = idx;
            });
            elements.datasetScrubber.addEventListener('change', (e) => {
                loadHistoricalSlice(parseInt(e.target.value));
            });
        }

        if (elements.btnLoadSlice) {
            elements.btnLoadSlice.addEventListener('click', () => {
                loadHistoricalSlice(parseInt(elements.datasetScrubber.value));
            });
        }

        if (elements.btnRandomSlice) {
            elements.btnRandomSlice.addEventListener('click', () => {
                const randIdx = Math.floor(Math.random() * 40000);
                elements.datasetScrubber.value = randIdx;
                elements.datasetIdxLabel.textContent = randIdx;
                loadHistoricalSlice(randIdx);
            });
        }

        // Randomize & Reset
        if (elements.btnRandomize) {
            elements.btnRandomize.addEventListener('click', () => {
                state.currentWeather.temperature = Number((Math.random() * 35).toFixed(1));
                state.currentWeather.humidity = Number((30 + Math.random() * 70).toFixed(0));
                state.currentWeather.pressure = Number((980 + Math.random() * 45).toFixed(1));
                state.currentWeather.wind_speed = Number((Math.random() * 60).toFixed(1));
                state.currentWeather.cloud_cover = Number((Math.random() * 100).toFixed(0));
                state.currentWeather.soil_moisture = Number((Math.random() * 100).toFixed(0));
                state.currentWeather.solar_radiation = Number((Math.random() * 900).toFixed(0));
                syncSlidersFromState();
                synthesize24hFromCurrent();
                debounceInference();
            });
        }

        if (elements.btnReset) {
            elements.btnReset.addEventListener('click', () => {
                loadScenario('thunderstorm');
            });
        }
    }

    function loadScenario(scId) {
        const sc = state.scenarios[scId];
        if (!sc) return;

        state.currentWeather = { ...sc.weather };
        state.pressureTrend = sc.pressure_trend || 'falling';

        // Update trend buttons
        if (elements.pressureTrendGroup) {
            elements.pressureTrendGroup.querySelectorAll('.trend-btn').forEach(b => {
                b.classList.toggle('active', b.dataset.trend === state.pressureTrend);
            });
        }

        syncSlidersFromState();
        synthesize24hFromCurrent();
        debounceInference();
    }

    async function loadHistoricalSlice(idx) {
        try {
            const res = await fetch(`/api/historical?start_idx=${idx}&length=24`);
            const data = await res.json();
            
            elements.metaDateStart.textContent = data.date_start;
            elements.metaDateEnd.textContent = data.date_end;
            elements.metaActualRain.textContent = `${data.next_actual_rainfall_mm} mm`;

            state.sequence24h = data.sequence.map(row => ({
                temperature: row.temperature,
                humidity: row.humidity,
                pressure: row.pressure,
                wind_speed: row.wind_speed,
                wind_direction: row.wind_direction,
                soil_moisture: row.soil_moisture,
                solar_radiation: row.solar_radiation,
                cloud_cover: row.cloud_cover,
                dew_point: row.dew_point,
                evapotranspiration: row.evapotranspiration,
                rainfall_mm: row.rainfall_mm
            }));

            // Sync current weather from the last row
            state.currentWeather = { ...state.sequence24h[23] };
            syncSlidersFromState();
            renderMatrixTable();
            runInference();
        } catch (err) {
            console.error('Historical slice error:', err);
        }
    }

    // --- 5. Chart.js Visualizations ---
    function initCharts() {
        const fontConfig = {
            family: "'Outfit', sans-serif",
            size: 11
        };

        // Timeline Multi-Metric Chart
        const ctxTimeline = document.getElementById('timelineChart').getContext('2d');
        state.charts.timeline = new Chart(ctxTimeline, {
            type: 'line',
            data: {
                labels: Array.from({ length: 24 }, (_, i) => `t-${23 - i}h`),
                datasets: [
                    {
                        label: 'Pressure (hPa)',
                        data: [],
                        borderColor: '#a855f7',
                        backgroundColor: 'rgba(168, 85, 247, 0.1)',
                        borderWidth: 2,
                        yAxisID: 'yPressure',
                        tension: 0.3,
                        pointRadius: 2
                    },
                    {
                        label: 'Cloud Cover (%)',
                        data: [],
                        borderColor: '#94a3b8',
                        borderWidth: 2,
                        borderDash: [4, 4],
                        yAxisID: 'yPercent',
                        tension: 0.3,
                        pointRadius: 0
                    },
                    {
                        label: 'Rainfall (mm)',
                        data: [],
                        type: 'bar',
                        backgroundColor: 'rgba(56, 189, 248, 0.65)',
                        borderColor: '#38bdf8',
                        borderWidth: 1,
                        yAxisID: 'yRain',
                        borderRadius: 3
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { mode: 'index', intersect: false },
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#64748b', font: fontConfig }
                    },
                    yPressure: {
                        type: 'linear',
                        position: 'left',
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#a855f7', font: fontConfig }
                    },
                    yPercent: {
                        type: 'linear',
                        position: 'right',
                        min: 0,
                        max: 100,
                        grid: { drawOnChartArea: false },
                        ticks: { color: '#94a3b8', font: fontConfig }
                    },
                    yRain: {
                        type: 'linear',
                        position: 'right',
                        min: 0,
                        grid: { drawOnChartArea: false },
                        ticks: { color: '#38bdf8', font: fontConfig }
                    }
                }
            }
        });

        // 12-Hour Forward Horizon Forecast Chart
        const ctxForecast = document.getElementById('forecastChart').getContext('2d');
        state.charts.forecast = new Chart(ctxForecast, {
            type: 'bar',
            data: {
                labels: Array.from({ length: 12 }, (_, i) => `+${i + 1}h`),
                datasets: [{
                    label: 'Predicted Rain (mm)',
                    data: [],
                    backgroundColor: 'rgba(56, 189, 248, 0.6)',
                    borderColor: '#38bdf8',
                    borderWidth: 1.5,
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#64748b', font: fontConfig }
                    },
                    y: {
                        min: 0,
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#94a3b8', font: fontConfig }
                    }
                }
            }
        });
    }

    function updateTimelineChart() {
        if (!state.charts.timeline || !state.sequence24h.length) return;
        
        const pressures = state.sequence24h.map(r => r.pressure);
        const clouds = state.sequence24h.map(r => r.cloud_cover);
        const rains = state.sequence24h.map(r => r.rainfall_mm);

        state.charts.timeline.data.datasets[0].data = pressures;
        state.charts.timeline.data.datasets[1].data = clouds;
        state.charts.timeline.data.datasets[2].data = rains;
        state.charts.timeline.update();
    }

    function updateForecastChart(forecastList) {
        if (!state.charts.forecast || !forecastList) return;
        
        const rainVals = forecastList.map(f => f.predicted_rainfall_mm);
        const colors = forecastList.map(f => f.category.color);
        
        state.charts.forecast.data.datasets[0].data = rainVals;
        state.charts.forecast.data.datasets[0].backgroundColor = colors.map(c => c + '99');
        state.charts.forecast.data.datasets[0].borderColor = colors;
        state.charts.forecast.update();
    }

    // --- 6. Ambient Canvas Particle Engine ---
    function initWeatherCanvas() {
        const canvas = document.getElementById('weatherCanvas');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');

        let width = (canvas.width = window.innerWidth);
        let height = (canvas.height = window.innerHeight);

        window.addEventListener('resize', () => {
            width = canvas.width = window.innerWidth;
            height = canvas.height = window.innerHeight;
        });

        const drops = [];
        const dropCount = 90;

        for (let i = 0; i < dropCount; i++) {
            drops.push({
                x: Math.random() * width,
                y: Math.random() * height,
                length: Math.random() * 20 + 10,
                speed: Math.random() * 12 + 8,
                opacity: Math.random() * 0.4 + 0.1
            });
        }

        function render() {
            ctx.clearRect(0, 0, width, height);

            ctx.strokeStyle = 'rgba(56, 189, 248, 0.4)';
            ctx.lineWidth = 1;

            drops.forEach(d => {
                ctx.beginPath();
                ctx.moveTo(d.x, d.y);
                ctx.lineTo(d.x - 2, d.y + d.length);
                ctx.stroke();

                d.y += d.speed;
                d.x -= 1;

                if (d.y > height) {
                    d.y = -20;
                    d.x = Math.random() * width;
                }
            });

            requestAnimationFrame(render);
        }

        render();
    }

    // Run initialization
    init();
});
