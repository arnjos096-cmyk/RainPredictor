/**
 * ISRO MOSDAC / SAC — Explainable AI (XAI) Nowcaster & INSAT-3D/3DR Satellite Frontend
 * Scientific Mission Operations Control Engine
 */

document.addEventListener('DOMContentLoaded', () => {
    // Application State
    const state = {
        mode: 'quick', // 'quick' | 'xai' | 'map' | 'metrics'
        scenario: 'mumbai_cloudburst',
        scenarios: {
            mumbai_cloudburst: {
                id: 'mumbai_cloudburst',
                name: 'Mumbai Extreme Offshore Cloudburst',
                location: 'Mumbai Coastal Observatory (18.98° N, 72.83° E)',
                pressure_trend: 'falling',
                hour_of_day: 14,
                data: {
                    tir1_temp: -68.5,
                    wv_channel: 96.0,
                    cloud_top_height: 16.2,
                    cape_index: 3850.0,
                    pressure: 990.0,
                    humidity: 98.0,
                    temperature: 28.0,
                    moisture_conv: 12.8,
                    wind_speed: 68.0,
                    wind_shear: 22.0,
                    rainfall_mm: 12.0
                }
            },
            kedarnath_himalayan: {
                id: 'kedarnath_himalayan',
                name: 'Uttarakhand Himalayan Convective Tower',
                location: 'Garhwal Himalayas (30.73° N, 79.06° E)',
                pressure_trend: 'falling',
                hour_of_day: 15,
                data: {
                    tir1_temp: -64.0,
                    wv_channel: 88.0,
                    cloud_top_height: 15.0,
                    cape_index: 3100.0,
                    pressure: 840.0,
                    humidity: 92.0,
                    temperature: 16.5,
                    moisture_conv: 11.2,
                    wind_speed: 48.0,
                    wind_shear: 24.5,
                    rainfall_mm: 8.5
                }
            },
            chennai_depression: {
                id: 'chennai_depression',
                name: 'Chennai Bay of Bengal Deep Depression',
                location: 'Chennai Coastal Radar (13.08° N, 80.27° E)',
                pressure_trend: 'falling',
                hour_of_day: 11,
                data: {
                    tir1_temp: -54.0,
                    wv_channel: 94.0,
                    cloud_top_height: 13.5,
                    cape_index: 2400.0,
                    pressure: 996.0,
                    humidity: 95.0,
                    temperature: 26.5,
                    moisture_conv: 7.8,
                    wind_speed: 62.0,
                    wind_shear: 16.0,
                    rainfall_mm: 6.0
                }
            },
            cirrus_anvil_false_alarm: {
                id: 'cirrus_anvil_false_alarm',
                name: 'Thin Cirrus Anvil Overhang (XAI False Alarm Test)',
                location: 'Deccan Plateau (17.38° N, 78.48° E)',
                pressure_trend: 'steady',
                hour_of_day: 16,
                data: {
                    tir1_temp: -66.0,
                    wv_channel: 38.0,
                    cloud_top_height: 14.5,
                    cape_index: 650.0,
                    pressure: 1011.0,
                    humidity: 42.0,
                    temperature: 31.0,
                    moisture_conv: 0.8,
                    wind_speed: 25.0,
                    wind_shear: 28.0,
                    rainfall_mm: 0.0
                }
            },
            rajasthan_anticyclone: {
                id: 'rajasthan_anticyclone',
                name: 'Thar Desert Subtropical Anticyclone',
                location: 'Jaisalmer Surface Station (26.91° N, 70.90° E)',
                pressure_trend: 'rising',
                hour_of_day: 13,
                data: {
                    tir1_temp: 24.0,
                    wv_channel: 14.0,
                    cloud_top_height: 0.8,
                    cape_index: 120.0,
                    pressure: 1016.0,
                    humidity: 18.0,
                    temperature: 39.5,
                    moisture_conv: 0.1,
                    wind_speed: 14.0,
                    wind_shear: 4.0,
                    rainfall_mm: 0.0
                }
            }
        },
        benchmarks: null,
        currentWeather: {
            tir1_temp: -68.5,
            wv_channel: 96.0,
            cloud_top_height: 16.2,
            cape_index: 3850.0,
            pressure: 990.0,
            humidity: 98.0,
            temperature: 28.0,
            moisture_conv: 12.8,
            wind_speed: 68.0,
            wind_shear: 22.0,
            rainfall_mm: 12.0
        },
        pressureTrend: 'falling',
        hourOfDay: 14,
        sequence24h: [],
        xaiData: null,
        charts: {
            timeline: null,
            forecast: null,
            attention: null
        },
        map: {
            instance: null,
            marker: null,
            radarLayers: [],
            cloudLayer: null,
            radarTimestamps: [],
            currentFrameIdx: 0,
            isPlaying: false,
            playInterval: null,
            opacity: 0.8,
            activeLayers: {
                radar: true,
                clouds: true,
                wind: false
            }
        },
        debounceTimer: null
    };

    // DOM Elements Mapping
    const elements = {
        alertBanner: document.getElementById('alertBanner'),
        alertTierTag: document.getElementById('alertTierTag'),
        alertTitle: document.getElementById('alertTitle'),
        alertDetail: document.getElementById('alertDetail'),
        alertBadgeIcon: document.getElementById('alertBadgeIcon'),
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
        quickXaiList: document.getElementById('quickXaiList'),
        featureAttributionBars: document.getElementById('featureAttributionBars'),
        failureDiagnosticsList: document.getElementById('failureDiagnosticsList'),
        metricsKpiGrid: document.getElementById('metricsKpiGrid'),
        paramHour: document.getElementById('param_hour_of_day'),
        valHour: document.getElementById('val_hour_of_day'),
        pressureTrendGroup: document.getElementById('pressureTrendGroup'),
        btnRandomize: document.getElementById('btnRandomize'),
        btnReset: document.getElementById('btnReset'),
        quickDeckView: document.getElementById('quickDeckView'),
        xaiSection: document.getElementById('xaiSection'),
        mapSection: document.getElementById('mapSection'),
        benchmarksSection: document.getElementById('benchmarksSection'),
        mainDashboardGrid: document.getElementById('mainDashboardGrid'),
        bottomTimelineSection: document.getElementById('bottomTimelineSection'),
        btnModeQuick: document.getElementById('btnModeQuick'),
        btnModeXAI: document.getElementById('btnModeXAI'),
        btnModeMap: document.getElementById('btnModeMap'),
        btnModeMetrics: document.getElementById('btnModeMetrics'),
        btnQuickOpenXAI: document.getElementById('btnQuickOpenXAI'),
        panelSubtext: document.getElementById('panelSubtext'),
        hudCoords: document.getElementById('hudCoords'),
        hudStationName: document.getElementById('hudStationName'),
        hudAirMass: document.getElementById('hudAirMass'),
        hudPressure: document.getElementById('hudPressure'),
        radarFrameLabel: document.getElementById('radarFrameLabel'),
        radarScrubber: document.getElementById('radarScrubber'),
        radarOpacity: document.getElementById('radarOpacity'),
        btnRadarPlay: document.getElementById('btnRadarPlay'),
        radarPlayIcon: document.getElementById('radarPlayIcon')
    };

    const paramKeys = [
        'tir1_temp', 'wv_channel', 'cloud_top_height', 'cape_index',
        'pressure', 'humidity', 'temperature', 'moisture_conv',
        'wind_speed', 'wind_shear', 'rainfall_mm'
    ];

    // --- 1. App Initialization ---
    async function init() {
        initLiveClock();
        initCharts();
        initMap();
        setupEventListeners();
        
        // Initial data synthesis & UI population
        synthesize24hFromCurrent();
        syncSlidersFromState();
        
        // Fetch background APIs gracefully
        await fetchStatus();
        await fetchScenarios();
        await fetchBenchmarks();
        
        // Initial neural inference run
        await runInference();
    }

    // --- 2. Live Mission Clock ---
    function initLiveClock() {
        const clockEl = document.getElementById('liveUtcClock');
        if (!clockEl) return;
        function updateClock() {
            const now = new Date();
            const utcString = now.toUTCString().split(' ')[4] + ' UTC';
            clockEl.textContent = utcString;
        }
        updateClock();
        setInterval(updateClock, 1000);
    }

    // --- 3. Leaflet Satellite & Radar Engine ---
    function initMap() {
        const mapContainer = document.getElementById('weatherMap');
        if (!mapContainer) return;

        state.map.instance = L.map('weatherMap', {
            center: [20.5937, 78.9629],
            zoom: 5,
            zoomControl: true,
            attributionControl: false
        });

        // Clean Dark Basemap
        L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
            maxZoom: 19,
            subdomains: 'abcd'
        }).addTo(state.map.instance);

        const customIcon = L.divIcon({
            className: 'radar-pulse-container',
            html: '<div class="radar-marker-pulse"></div>',
            iconSize: [20, 20],
            iconAnchor: [10, 10]
        });

        state.map.marker = L.marker([18.98, 72.83], { icon: customIcon }).addTo(state.map.instance);

        loadRainViewerRadar();

        state.map.instance.on('click', (e) => {
            const { lat, lng } = e.latlng;
            updateStationPin(lat, lng, `INSAT Sector [${lat.toFixed(2)}°N, ${lng.toFixed(2)}°E]`);
        });
    }

    async function loadRainViewerRadar() {
        try {
            const res = await fetch('https://api.rainviewer.com/public/weather-maps.json');
            const data = await res.json();
            
            if (data && data.radar && Array.isArray(data.radar.past) && data.radar.past.length > 0) {
                state.map.radarTimestamps = data.radar.past;
                if (elements.radarScrubber) {
                    elements.radarScrubber.max = data.radar.past.length - 1;
                    elements.radarScrubber.value = data.radar.past.length - 1;
                }
                state.map.currentFrameIdx = data.radar.past.length - 1;
                showRadarFrame(state.map.currentFrameIdx);
            }

            if (data && data.satellite && Array.isArray(data.satellite.infrared) && data.satellite.infrared.length > 0) {
                const cloudTime = data.satellite.infrared[data.satellite.infrared.length - 1].time;
                if (cloudTime && state.map.instance) {
                    if (state.map.cloudLayer) {
                        state.map.instance.removeLayer(state.map.cloudLayer);
                    }
                    state.map.cloudLayer = L.tileLayer(`https://tilecache.rainviewer.com/v2/satellite/${cloudTime}/256/{z}/{x}/{y}/0/0_0.png`, {
                        opacity: state.map.opacity,
                        zIndex: 2
                    });
                    if (state.map.activeLayers.clouds) {
                        state.map.cloudLayer.addTo(state.map.instance);
                    }
                }
            } else if (state.map.instance) {
                state.map.cloudLayer = L.tileLayer('https://tilecache.rainviewer.com/v2/satellite/latest/256/{z}/{x}/{y}/0/0_0.png', {
                    opacity: state.map.opacity,
                    zIndex: 2
                });
                if (state.map.activeLayers.clouds) {
                    state.map.cloudLayer.addTo(state.map.instance);
                }
            }
        } catch (err) {
            console.log('Using simulated INSAT-3D Doppler radar frames:', err);
            if (state.map.instance && !state.map.cloudLayer) {
                state.map.cloudLayer = L.tileLayer('https://tilecache.rainviewer.com/v2/satellite/latest/256/{z}/{x}/{y}/0/0_0.png', {
                    opacity: state.map.opacity,
                    zIndex: 2
                });
                if (state.map.activeLayers.clouds) {
                    state.map.cloudLayer.addTo(state.map.instance);
                }
            }
        }
    }

    function showRadarFrame(index) {
        if (!state.map.radarTimestamps || !state.map.radarTimestamps.length) return;
        const frame = state.map.radarTimestamps[index];
        if (!frame || !state.map.instance) return;

        if (state.map.radarLayers.length > 0) {
            state.map.radarLayers.forEach(l => state.map.instance.removeLayer(l));
            state.map.radarLayers = [];
        }

        if (state.map.activeLayers.radar) {
            const radarTile = L.tileLayer(`https://tilecache.rainviewer.com/v2/radar/${frame.time}/256/{z}/{x}/{y}/2/1_1.png`, {
                opacity: state.map.opacity,
                zIndex: 3
            }).addTo(state.map.instance);

            state.map.radarLayers.push(radarTile);
        }

        if (elements.radarFrameLabel) {
            const frameDate = new Date(frame.time * 1000);
            elements.radarFrameLabel.textContent = `${frameDate.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })} (UTC Scan)`;
        }
    }

    function updateStationPin(lat, lng, name = null) {
        if (!state.map.marker) return;
        state.map.marker.setLatLng([lat, lng]);

        if (elements.hudCoords) {
            elements.hudCoords.textContent = `${Math.abs(lat).toFixed(2)}° ${lat >= 0 ? 'N' : 'S'}, ${Math.abs(lng).toFixed(2)}° ${lng >= 0 ? 'E' : 'W'}`;
        }
        if (name && elements.hudStationName) elements.hudStationName.textContent = name;

        // Simulated geographic presets
        if (lat < 20.0 && lng < 75.0) {
            state.currentWeather.tir1_temp = -62.0;
            state.currentWeather.cloud_top_height = 15.0;
            state.currentWeather.cape_index = 3200.0;
            state.currentWeather.moisture_conv = 10.5;
            state.currentWeather.pressure = 993.0;
        } else if (lat > 28.0 && lng > 77.0) {
            state.currentWeather.tir1_temp = -64.0;
            state.currentWeather.cloud_top_height = 14.5;
            state.currentWeather.cape_index = 2900.0;
            state.currentWeather.moisture_conv = 9.8;
            state.currentWeather.pressure = 880.0;
        } else if (lat < 16.0 && lng > 79.0) {
            state.currentWeather.tir1_temp = -55.0;
            state.currentWeather.cloud_top_height = 13.0;
            state.currentWeather.cape_index = 2200.0;
            state.currentWeather.moisture_conv = 7.5;
            state.currentWeather.pressure = 998.0;
        } else {
            state.currentWeather.tir1_temp = 20.0;
            state.currentWeather.cloud_top_height = 1.0;
            state.currentWeather.cape_index = 200.0;
            state.currentWeather.moisture_conv = 0.2;
            state.currentWeather.pressure = 1014.0;
        }

        syncSlidersFromState();
        synthesize24hFromCurrent();
        debounceInference();
    }

    function toggleRadarPlayback() {
        if (!state.map.radarTimestamps.length) return;
        if (state.map.isPlaying) {
            clearInterval(state.map.playInterval);
            state.map.isPlaying = false;
            if (elements.radarPlayIcon) elements.radarPlayIcon.className = 'fa-solid fa-play';
        } else {
            state.map.isPlaying = true;
            if (elements.radarPlayIcon) elements.radarPlayIcon.className = 'fa-solid fa-pause';
            state.map.playInterval = setInterval(() => {
                let nextIdx = (state.map.currentFrameIdx + 1) % state.map.radarTimestamps.length;
                state.map.currentFrameIdx = nextIdx;
                if (elements.radarScrubber) elements.radarScrubber.value = nextIdx;
                showRadarFrame(nextIdx);
            }, 800);
        }
    }

    // --- 4. API Communication ---
    async function fetchStatus() {
        try {
            const res = await fetch('/api/status');
            const data = await res.json();
            if (data && data.model_loaded) {
                console.log('INSAT-3D/3DR XAI Engine Online:', data.device);
            }
        } catch (err) {
            console.log('Status endpoint check:', err);
        }
    }

    async function fetchScenarios() {
        try {
            const res = await fetch('/api/scenarios');
            const scenarios = await res.json();
            if (Array.isArray(scenarios)) {
                scenarios.forEach(s => {
                    state.scenarios[s.id] = s;
                });
            }
        } catch (err) {
            console.error('Scenarios fetch error:', err);
        }
    }

    async function fetchBenchmarks() {
        try {
            const res = await fetch('/api/benchmarks');
            state.benchmarks = await res.json();
            renderBenchmarkMetrics();
        } catch (err) {
            console.error('Benchmarks fetch error:', err);
        }
    }

    function renderBenchmarkMetrics() {
        if (!state.benchmarks || !elements.metricsKpiGrid) return;
        const metrics = state.benchmarks.verification_metrics;
        if (!metrics) return;
        
        elements.metricsKpiGrid.innerHTML = Object.entries(metrics).map(([k, m]) => `
            <div class="kpi-card">
                <span class="kpi-label">${m.label}</span>
                <div class="kpi-val-row">
                    <span class="kpi-val text-emerald">${m.value}</span>
                    <span class="kpi-target">Target: ${m.target}</span>
                </div>
            </div>
        `).join('');
    }

    async function runInference() {
        if (elements.lastInferenceTime) elements.lastInferenceTime.textContent = 'Evaluating...';
        
        try {
            const res = await fetch('/api/predict-single', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    current: state.currentWeather,
                    pressure_trend: state.pressureTrend,
                    hour_of_day: state.hourOfDay
                })
            });

            const data = await res.json();
            state.xaiData = data;
            updatePredictionUI(data);
            updateXAIPanels(data);
            runForwardForecast();
            updateTimelineChart();
        } catch (err) {
            console.error('Inference error:', err);
            if (elements.lastInferenceTime) elements.lastInferenceTime.textContent = 'Ready';
        }
    }

    async function runForwardForecast() {
        try {
            const seqToSend = state.sequence24h;
            if (!seqToSend || seqToSend.length !== 24) return;

            const res = await fetch('/api/predict-forecast', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    sequence: seqToSend,
                    steps: 12,
                    hour_of_day: state.hourOfDay
                })
            });
            const data = await res.json();
            if (data && data.forecast) {
                updateForecastChart(data.forecast);
            }
        } catch (err) {
            console.error('Forecast error:', err);
        }
    }

    // --- 5. UI & XAI Rendering ---
    function updatePredictionUI(data) {
        const mm = data.predicted_rainfall_mm;
        const cat = data.category;

        if (elements.gaugeValue) elements.gaugeValue.textContent = mm.toFixed(1);
        
        // Circular gauge (Circumference ~502)
        if (elements.gaugeProgress) {
            const circumference = 502;
            const percentage = Math.min(1.0, mm / 70.0);
            const offset = circumference - (percentage * (circumference * 0.75));
            elements.gaugeProgress.style.strokeDashoffset = offset;
            elements.gaugeProgress.style.stroke = cat.color;
        }

        // Risk Badge
        if (elements.riskBadge) {
            elements.riskBadge.style.color = cat.color;
            elements.riskBadge.style.borderColor = cat.color + '66';
            elements.riskBadge.style.backgroundColor = cat.color + '18';
        }
        if (elements.riskIcon) elements.riskIcon.className = `fa-solid ${cat.icon}`;
        if (elements.riskLabel) elements.riskLabel.textContent = cat.tier;
        if (elements.riskDescription) elements.riskDescription.textContent = cat.action;

        // Occurrence Probability
        if (elements.probPercent) elements.probPercent.textContent = `${data.rain_probability_percent.toFixed(1)}%`;
        if (elements.probFill) {
            elements.probFill.style.width = `${data.rain_probability_percent}%`;
            elements.probFill.style.background = cat.color;
        }

        if (elements.lastInferenceTime) {
            const d = new Date();
            elements.lastInferenceTime.textContent = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        }

        // Alert Banner Sync
        if (elements.alertBanner) {
            elements.alertBanner.className = `alert-banner ${cat.alert_class}`;
        }
        const desc = cat.description || cat.tier || 'Precipitation';
        if (elements.alertTitle) elements.alertTitle.textContent = `${desc} Predicted`;
        if (elements.alertDetail) elements.alertDetail.textContent = cat.action || cat.impact || 'Normal operations.';

        // Quick XAI List (Top 3 features)
        if (elements.quickXaiList && data.feature_attribution) {
            const top3 = data.feature_attribution.slice(0, 3);
            elements.quickXaiList.innerHTML = top3.map(f => `
                <div class="quick-xai-item">
                    <span><i class="fa-solid fa-satellite text-blue"></i> ${f.label}</span>
                    <strong style="color: ${f.percentage > 15 ? '#38bdf8' : '#94a3b8'}">${f.percentage.toFixed(1)}%</strong>
                </div>
            `).join('');
        }
    }

    function updateXAIPanels(data) {
        // 1. Attention Weights Chart
        if (state.charts.attention && data.temporal_attention_weights) {
            state.charts.attention.data.datasets[0].data = data.temporal_attention_weights;
            state.charts.attention.update();
        }

        // 2. Feature Attribution Waterfall Bars
        if (elements.featureAttributionBars && data.feature_attribution) {
            elements.featureAttributionBars.innerHTML = data.feature_attribution.map(f => `
                <div class="attr-row">
                    <div class="attr-label-row">
                        <span><strong>${f.label}</strong> <em style="font-size: 0.68rem; color: #64748b;">(${f.description})</em></span>
                        <span class="attr-pct">${f.percentage.toFixed(1)}%</span>
                    </div>
                    <div class="attr-track">
                        <div class="attr-fill" style="width: ${Math.min(100, f.percentage * 2.8)}%;"></div>
                    </div>
                </div>
            `).join('');
        }

        // 3. Failure & Ambiguity Diagnostics
        if (elements.failureDiagnosticsList && data.failure_diagnostics) {
            if (data.failure_diagnostics.length > 0) {
                elements.failureDiagnosticsList.innerHTML = data.failure_diagnostics.map(f => {
                    const sevClass = f.severity === 'HIGH' ? 'severity-high' :
                                     f.severity === 'MODERATE' ? 'severity-mod' : 'severity-low';
                    return `
                        <div class="failure-item">
                            <div class="failure-item-header">
                                <span class="failure-item-title"><i class="fa-solid fa-triangle-exclamation text-amber"></i> ${f.risk_title}</span>
                                <span class="failure-severity-badge ${sevClass}">${f.severity} RISK</span>
                            </div>
                            <p class="failure-cause">${f.cause}</p>
                            <span class="failure-remedy"><i class="fa-solid fa-shield-halved"></i> Operational Protocol: ${f.recommended_correction}</span>
                        </div>
                    `;
                }).join('');
            } else {
                elements.failureDiagnosticsList.innerHTML = `
                    <div class="failure-item">
                        <span class="failure-item-title"><i class="fa-solid fa-check-double text-emerald"></i> Standard Atmospheric Coherence</span>
                        <p class="failure-cause">No high-risk satellite artifacts or cloud top ambiguities detected for this observation window.</p>
                    </div>
                `;
            }
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
            const factor = (24 - h_offset) / 24.0;
            
            let t_ir = curr.tir1_temp;
            let c_height = curr.cloud_top_height;
            let cape = curr.cape_index;
            let m_conv = curr.moisture_conv;
            let p_val = curr.pressure;
            let hum = curr.humidity;
            let w_speed = curr.wind_speed;

            if (trend === 'falling') {
                t_ir = curr.tir1_temp + (h_offset * 1.5);
                c_height = Math.max(2.0, curr.cloud_top_height * (0.3 + 0.7 * factor));
                cape = Math.max(500.0, curr.cape_index * (0.4 + 0.6 * factor));
                m_conv = Math.max(0.5, curr.moisture_conv * (0.3 + 0.7 * factor));
                p_val = curr.pressure + (h_offset * 0.25);
                hum = Math.min(100.0, curr.humidity * (0.7 + 0.3 * factor));
                w_speed = Math.max(10.0, curr.wind_speed * (0.6 + 0.4 * factor));
            } else {
                t_ir = curr.tir1_temp - (h_offset * 0.5);
                c_height = Math.max(1.5, curr.cloud_top_height * (1.1 - 0.1 * factor));
                cape = Math.max(300.0, curr.cape_index * (1.1 - 0.1 * factor));
                m_conv = Math.max(0.1, curr.moisture_conv * (1.1 - 0.1 * factor));
                p_val = curr.pressure - (h_offset * 0.2);
                hum = Math.max(20.0, curr.humidity * (1.1 - 0.1 * factor));
                w_speed = Math.max(5.0, curr.wind_speed * (1.1 - 0.1 * factor));
            }

            let rain_val = 0.0;
            if (step === 23) {
                rain_val = curr.rainfall_mm;
            } else if (trend === 'falling' && t_ir < -40.0 && step > 19) {
                rain_val = Math.random() * 4.0;
            }

            state.sequence24h.push({
                tir1_temp: Number(t_ir.toFixed(1)),
                wv_channel: Number(curr.wv_channel.toFixed(1)),
                cloud_top_height: Number(c_height.toFixed(1)),
                cape_index: Number(cape.toFixed(0)),
                pressure: Number(p_val.toFixed(1)),
                humidity: Number(hum.toFixed(1)),
                temperature: Number((curr.temperature + Math.sin(past_hour / 4.0)).toFixed(1)),
                moisture_conv: Number(m_conv.toFixed(1)),
                wind_speed: Number(w_speed.toFixed(1)),
                wind_shear: curr.wind_shear,
                rainfall_mm: Number(rain_val.toFixed(2))
            });
        }
    }

    function syncSlidersFromState() {
        paramKeys.forEach(k => {
            const slider = document.getElementById(`param_${k}`);
            const valLabel = document.getElementById(`val_${k}`);
            if (slider && valLabel) {
                slider.value = state.currentWeather[k];
                const unit = k === 'tir1_temp' || k === 'temperature' ? '°C' :
                             k === 'wv_channel' || k === 'humidity' ? '%' :
                             k === 'cloud_top_height' ? 'km' :
                             k === 'cape_index' ? 'J/kg' :
                             k === 'moisture_conv' ? 'g/kg/h' :
                             k === 'pressure' ? 'hPa' :
                             k === 'wind_speed' ? 'km/h' :
                             k === 'wind_shear' ? 'm/s' : 'mm/h';
                valLabel.innerHTML = `${state.currentWeather[k]} <span class="unit">${unit}</span>`;
            }
        });
    }

    function loadScenario(scId) {
        if (!state.scenarios[scId]) return;
        const sc = state.scenarios[scId];
        const scenarioData = sc.data || sc.weather;
        if (!scenarioData) return;
        
        state.scenario = scId;
        state.currentWeather = { ...scenarioData };
        state.pressureTrend = sc.pressure_trend || 'falling';
        state.hourOfDay = sc.hour_of_day !== undefined ? sc.hour_of_day : 14;

        if (elements.pressureTrendGroup) {
            elements.pressureTrendGroup.querySelectorAll('.trend-btn').forEach(b => {
                b.classList.toggle('active', b.dataset.trend === state.pressureTrend);
            });
        }
        if (elements.paramHour) elements.paramHour.value = state.hourOfDay;
        if (elements.valHour) {
            const isDay = state.hourOfDay >= 6 && state.hourOfDay <= 18;
            elements.valHour.textContent = `${state.hourOfDay.toString().padStart(2, '0')}:00 (${isDay ? 'Daylight Peak' : 'Night'})`;
        }

        syncSlidersFromState();
        synthesize24hFromCurrent();
        runInference();
    }

    function debounceInference() {
        clearTimeout(state.debounceTimer);
        state.debounceTimer = setTimeout(() => {
            runInference();
        }, 150);
    }

    // --- 6. Event Listeners ---
    function setupEventListeners() {
        paramKeys.forEach(k => {
            const slider = document.getElementById(`param_${k}`);
            const valLabel = document.getElementById(`val_${k}`);
            if (slider) {
                slider.addEventListener('input', (e) => {
                    const val = parseFloat(e.target.value);
                    state.currentWeather[k] = val;
                    const unit = k === 'tir1_temp' || k === 'temperature' ? '°C' :
                                 k === 'wv_channel' || k === 'humidity' ? '%' :
                                 k === 'cloud_top_height' ? 'km' :
                                 k === 'cape_index' ? 'J/kg' :
                                 k === 'moisture_conv' ? 'g/kg/h' :
                                 k === 'pressure' ? 'hPa' :
                                 k === 'wind_speed' ? 'km/h' :
                                 k === 'wind_shear' ? 'm/s' : 'mm/h';
                    valLabel.innerHTML = `${val} <span class="unit">${unit}</span>`;
                    
                    synthesize24hFromCurrent();
                    debounceInference();
                });
            }
        });

        if (elements.paramHour) {
            elements.paramHour.addEventListener('input', (e) => {
                state.hourOfDay = parseInt(e.target.value);
                const isDay = state.hourOfDay >= 6 && state.hourOfDay <= 18;
                if (elements.valHour) {
                    elements.valHour.textContent = `${state.hourOfDay.toString().padStart(2, '0')}:00 (${isDay ? 'Daylight Peak' : 'Night'})`;
                }
                synthesize24hFromCurrent();
                debounceInference();
            });
        }

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

        // View Mode Switching Tabs
        const modeButtons = [elements.btnModeQuick, elements.btnModeXAI, elements.btnModeMap, elements.btnModeMetrics];
        modeButtons.forEach(btn => {
            if (!btn) return;
            btn.addEventListener('click', () => {
                modeButtons.forEach(b => { if (b) b.classList.remove('active'); });
                btn.classList.add('active');
                state.mode = btn.dataset.mode;

                if (elements.xaiSection) elements.xaiSection.classList.add('hidden');
                if (elements.mapSection) elements.mapSection.classList.add('hidden');
                if (elements.benchmarksSection) elements.benchmarksSection.classList.add('hidden');
                if (elements.mainDashboardGrid) elements.mainDashboardGrid.classList.remove('hidden');
                if (elements.bottomTimelineSection) elements.bottomTimelineSection.classList.remove('hidden');

                if (state.mode === 'xai') {
                    if (elements.xaiSection) elements.xaiSection.classList.remove('hidden');
                } else if (state.mode === 'map') {
                    if (elements.mapSection) elements.mapSection.classList.remove('hidden');
                    setTimeout(() => {
                        if (state.map.instance) state.map.instance.invalidateSize();
                    }, 200);
                } else if (state.mode === 'metrics') {
                    if (elements.benchmarksSection) elements.benchmarksSection.classList.remove('hidden');
                    if (elements.mainDashboardGrid) elements.mainDashboardGrid.classList.add('hidden');
                    if (elements.bottomTimelineSection) elements.bottomTimelineSection.classList.add('hidden');
                }

                runInference();
            });
        });

        if (elements.btnQuickOpenXAI) {
            elements.btnQuickOpenXAI.addEventListener('click', () => {
                if (elements.btnModeXAI) elements.btnModeXAI.click();
            });
        }

        document.querySelectorAll('.station-chip').forEach(chip => {
            chip.addEventListener('click', () => {
                const lat = parseFloat(chip.dataset.lat);
                const lng = parseFloat(chip.dataset.lng);
                const city = chip.dataset.city;
                const sc = chip.dataset.scenario;
                
                if (state.map.instance) {
                    state.map.instance.flyTo([lat, lng], 6, { duration: 1.5 });
                }
                updateStationPin(lat, lng, `${city} INSAT Sector`);
                if (sc) loadScenario(sc);
            });
        });

        if (elements.btnRadarPlay) {
            elements.btnRadarPlay.addEventListener('click', toggleRadarPlayback);
        }

        if (elements.radarScrubber) {
            elements.radarScrubber.addEventListener('input', (e) => {
                state.map.currentFrameIdx = parseInt(e.target.value);
                showRadarFrame(state.map.currentFrameIdx);
            });
        }

        // Map Layer Opacity Slider (Adjusts Radar, Satellite Clouds, and Overlays)
        if (elements.radarOpacity) {
            elements.radarOpacity.addEventListener('input', (e) => {
                const val = parseInt(e.target.value);
                state.map.opacity = val / 100.0;
                
                // 1. Update Radar Doppler frame opacity
                if (state.map.radarLayers.length > 0 && state.map.activeLayers.radar) {
                    state.map.radarLayers.forEach(l => {
                        if (l && typeof l.setOpacity === 'function') {
                            l.setOpacity(state.map.opacity);
                        }
                    });
                }

                // 2. Update INSAT Infrared Cloud Layer opacity
                if (state.map.cloudLayer && typeof state.map.cloudLayer.setOpacity === 'function') {
                    state.map.cloudLayer.setOpacity(state.map.opacity);
                }
            });
        }

        // Map Layer Switcher Pills (Rain Radar, INSAT Infrared Clouds, Wind)
        document.querySelectorAll('.layer-pill').forEach(pill => {
            pill.addEventListener('click', () => {
                const layer = pill.dataset.layer;
                pill.classList.toggle('active');
                const isActive = pill.classList.contains('active');
                state.map.activeLayers[layer] = isActive;

                if (layer === 'clouds') {
                    if (state.map.cloudLayer) {
                        if (isActive) {
                            state.map.cloudLayer.addTo(state.map.instance);
                            state.map.cloudLayer.setOpacity(state.map.opacity);
                        } else {
                            state.map.instance.removeLayer(state.map.cloudLayer);
                        }
                    }
                } else if (layer === 'radar') {
                    if (isActive) {
                        showRadarFrame(state.map.currentFrameIdx);
                    } else {
                        if (state.map.radarLayers.length > 0) {
                            state.map.radarLayers.forEach(l => state.map.instance.removeLayer(l));
                            state.map.radarLayers = [];
                        }
                    }
                }
            });
        });

        if (elements.btnRandomize) {
            elements.btnRandomize.addEventListener('click', () => {
                state.currentWeather.tir1_temp = -75.0 + Math.random() * 95.0;
                state.currentWeather.wv_channel = 20.0 + Math.random() * 80.0;
                state.currentWeather.cloud_top_height = 2.0 + Math.random() * 15.0;
                state.currentWeather.cape_index = 200.0 + Math.random() * 4200.0;
                state.currentWeather.pressure = 980.0 + Math.random() * 40.0;
                state.currentWeather.humidity = 40.0 + Math.random() * 60.0;
                state.currentWeather.moisture_conv = 0.5 + Math.random() * 14.0;
                state.currentWeather.wind_speed = 10.0 + Math.random() * 90.0;
                state.currentWeather.wind_shear = 5.0 + Math.random() * 30.0;
                state.currentWeather.rainfall_mm = Math.random() * 15.0;
                
                syncSlidersFromState();
                synthesize24hFromCurrent();
                runInference();
            });
        }

        if (elements.btnReset) {
            elements.btnReset.addEventListener('click', () => {
                loadScenario('mumbai_cloudburst');
            });
        }
    }

    // --- 7. Chart Initializations ---
    function initCharts() {
        const fontConfig = { family: "'JetBrains Mono', monospace", size: 10 };

        // Timeline Evolution Chart
        const ctxTimeline = document.getElementById('timelineChart');
        if (ctxTimeline) {
            state.charts.timeline = new Chart(ctxTimeline.getContext('2d'), {
                type: 'line',
                data: {
                    labels: Array.from({ length: 24 }, (_, i) => `t-${23 - i}h`),
                    datasets: [
                        {
                            label: 'Surface Pressure (hPa)',
                            data: [],
                            borderColor: '#c084fc',
                            borderWidth: 1.5,
                            tension: 0.3,
                            pointRadius: 2,
                            yAxisID: 'y1'
                        },
                        {
                            label: 'TIR-1 Temp (°C)',
                            data: [],
                            borderColor: '#38bdf8',
                            borderWidth: 1.5,
                            tension: 0.3,
                            pointRadius: 2,
                            yAxisID: 'y2'
                        },
                        {
                            label: 'Rain Rate (mm/h)',
                            data: [],
                            borderColor: '#ef4444',
                            backgroundColor: 'rgba(239, 68, 68, 0.15)',
                            fill: true,
                            borderWidth: 1.5,
                            tension: 0.3,
                            pointRadius: 2,
                            yAxisID: 'y3'
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: { mode: 'index', intersect: false },
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#64748b', font: fontConfig } },
                        y1: { type: 'linear', position: 'left', min: 960, max: 1030, grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#c084fc', font: fontConfig } },
                        y2: { type: 'linear', position: 'right', min: -85, max: 40, grid: { drawOnChartArea: false }, ticks: { color: '#38bdf8', font: fontConfig } },
                        y3: { type: 'linear', position: 'right', min: 0, max: 80, grid: { drawOnChartArea: false }, ticks: { color: '#ef4444', font: fontConfig } }
                    }
                }
            });
        }

        // 12-Hour Forward Horizon Forecast Chart
        const ctxForecast = document.getElementById('forecastChart');
        if (ctxForecast) {
            state.charts.forecast = new Chart(ctxForecast.getContext('2d'), {
                type: 'bar',
                data: {
                    labels: Array.from({ length: 12 }, (_, i) => `+${i + 1}h`),
                    datasets: [{
                        label: 'Predicted Rain Rate (mm/h)',
                        data: [],
                        backgroundColor: 'rgba(56, 189, 248, 0.6)',
                        borderColor: '#38bdf8',
                        borderWidth: 1,
                        borderRadius: 3
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#64748b', font: fontConfig } },
                        y: { min: 0, grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#94a3b8', font: fontConfig } }
                    }
                }
            });
        }

        // XAI Temporal Attention Chart
        const ctxAttention = document.getElementById('attentionChart');
        if (ctxAttention) {
            state.charts.attention = new Chart(ctxAttention.getContext('2d'), {
                type: 'line',
                data: {
                    labels: Array.from({ length: 24 }, (_, i) => `t-${23 - i}h`),
                    datasets: [{
                        label: 'Attention Weight (α)',
                        data: Array(24).fill(0.04),
                        borderColor: '#f97316',
                        backgroundColor: 'rgba(249, 115, 22, 0.15)',
                        fill: true,
                        borderWidth: 2,
                        tension: 0.35,
                        pointBackgroundColor: '#f97316',
                        pointRadius: 2.5
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#64748b', font: fontConfig } },
                        y: { min: 0, grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#f97316', font: fontConfig } }
                    }
                }
            });
        }
    }

    function updateTimelineChart() {
        if (!state.charts.timeline || !state.sequence24h.length) return;
        state.charts.timeline.data.datasets[0].data = state.sequence24h.map(r => r.pressure);
        state.charts.timeline.data.datasets[1].data = state.sequence24h.map(r => r.tir1_temp);
        state.charts.timeline.data.datasets[2].data = state.sequence24h.map(r => r.rainfall_mm);
        state.charts.timeline.update();
    }

    function updateForecastChart(forecastList) {
        if (!state.charts.forecast || !forecastList) return;
        state.charts.forecast.data.datasets[0].data = forecastList.map(f => f.predicted_rainfall_mm);
        state.charts.forecast.data.datasets[0].backgroundColor = forecastList.map(f => (f.category.color || '#38bdf8') + 'aa');
        state.charts.forecast.data.datasets[0].borderColor = forecastList.map(f => f.category.color || '#38bdf8');
        state.charts.forecast.update();
    }

    init();
});
