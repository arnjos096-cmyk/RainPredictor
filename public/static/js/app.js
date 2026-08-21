/**
 * ISRO MOSDAC / SAC — Explainable AI (XAI) Heavy Rain Nowcaster & DQN Decision System
 * Frontend Mission Operations Control Engine
 */

document.addEventListener('DOMContentLoaded', () => {
    // Application State
    const state = {
        mode: 'quick', // 'quick' | 'xai' | 'rl' | 'historical' | 'map' | 'metrics'
        scenario: 'mumbai_cloudburst',
        scenarios: {
            mumbai_cloudburst: {
                id: 'mumbai_cloudburst',
                name: 'Mumbai Extreme Convective Cloudburst',
                location: 'Mumbai Coastal Observatory (18.98° N, 72.83° E)',
                pressure_trend: 'falling',
                hour_of_day: 14,
                weather: {
                    tir1_temp: -68.5,
                    wv_channel: 98.0,
                    cloud_top_height: 16.5,
                    cape_index: 3800.0,
                    pressure: 991.0,
                    humidity: 98.0,
                    temperature: 26.5,
                    moisture_conv: 12.8,
                    wind_speed: 62.0,
                    wind_shear: 24.0,
                    rainfall_mm: 68.4
                }
            },
            kedarnath_himalayan: {
                id: 'kedarnath_himalayan',
                name: 'Uttarakhand Himalayan Cloudburst',
                location: 'Garhwal Himalayas (30.73° N, 79.06° E)',
                pressure_trend: 'falling',
                hour_of_day: 15,
                weather: {
                    tir1_temp: -62.0,
                    wv_channel: 94.0,
                    cloud_top_height: 15.0,
                    cape_index: 2900.0,
                    pressure: 988.0,
                    humidity: 94.0,
                    temperature: 21.0,
                    moisture_conv: 10.5,
                    wind_speed: 45.0,
                    wind_shear: 20.0,
                    rainfall_mm: 48.2
                }
            },
            chennai_depression: {
                id: 'chennai_depression',
                name: 'Chennai Cyclonic Rainband (Depression)',
                location: 'Chennai Coastal Radar (13.08° N, 80.27° E)',
                pressure_trend: 'falling',
                hour_of_day: 11,
                weather: {
                    tir1_temp: -48.0,
                    wv_channel: 90.0,
                    cloud_top_height: 12.8,
                    cape_index: 2100.0,
                    pressure: 997.0,
                    humidity: 92.0,
                    temperature: 28.0,
                    moisture_conv: 7.5,
                    wind_speed: 48.0,
                    wind_shear: 16.0,
                    rainfall_mm: 38.0
                }
            },
            cirrus_anvil_false_alarm: {
                id: 'cirrus_anvil_false_alarm',
                name: 'Cold Cirrus Anvil Shield (XAI Test Case)',
                location: 'Deccan Plateau (17.38° N, 78.48° E)',
                pressure_trend: 'steady',
                hour_of_day: 16,
                weather: {
                    tir1_temp: -54.0,
                    wv_channel: 48.0,
                    cloud_top_height: 13.5,
                    cape_index: 1200.0,
                    pressure: 1010.0,
                    humidity: 52.0,
                    temperature: 32.0,
                    moisture_conv: 1.2,
                    wind_speed: 22.0,
                    wind_shear: 12.0,
                    rainfall_mm: 0.0
                }
            },
            rajasthan_anticyclone: {
                id: 'rajasthan_anticyclone',
                name: 'Thar Desert Anticyclone',
                location: 'Jaisalmer Surface Station (26.91° N, 70.90° E)',
                pressure_trend: 'rising',
                hour_of_day: 13,
                weather: {
                    tir1_temp: 18.0,
                    wv_channel: 24.0,
                    cloud_top_height: 1.5,
                    cape_index: 200.0,
                    pressure: 1018.0,
                    humidity: 28.0,
                    temperature: 38.5,
                    moisture_conv: 0.1,
                    wind_speed: 14.0,
                    wind_shear: 6.0,
                    rainfall_mm: 0.0
                }
            }
        },
        benchmarks: null,
        currentWeather: {
            tir1_temp: -58.0,
            wv_channel: 92.0,
            cloud_top_height: 14.2,
            cape_index: 2850.0,
            pressure: 994.0,
            humidity: 95.0,
            temperature: 27.5,
            moisture_conv: 8.4,
            wind_speed: 54.0,
            wind_shear: 18.5,
            rainfall_mm: 4.5
        },
        pressureTrend: 'falling',
        hourOfDay: 14,
        sequence24h: [],
        xaiData: null,
        historical: {
            currentIndex: 1000,
            totalRecords: 43800,
            currentSequence: [],
            nextActualRain: 0.0,
            predictedRain: 0.0
        },
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
        rlSection: document.getElementById('rlSection'),
        historicalSection: document.getElementById('historicalSection'),
        mapSection: document.getElementById('mapSection'),
        benchmarksSection: document.getElementById('benchmarksSection'),
        mainDashboardGrid: document.getElementById('mainDashboardGrid'),
        bottomTimelineSection: document.getElementById('bottomTimelineSection'),
        btnModeQuick: document.getElementById('btnModeQuick'),
        btnModeXAI: document.getElementById('btnModeXAI'),
        btnModeRL: document.getElementById('btnModeRL'),
        btnModeHistorical: document.getElementById('btnModeHistorical'),
        btnModeMap: document.getElementById('btnModeMap'),
        btnModeMetrics: document.getElementById('btnModeMetrics'),
        btnQuickOpenXAI: document.getElementById('btnQuickOpenXAI'),
        btnQuickOpenRL: document.getElementById('btnQuickOpenRL'),
        panelSubtext: document.getElementById('panelSubtext'),
        hudCoords: document.getElementById('hudCoords'),
        hudStationName: document.getElementById('hudStationName'),
        hudAirMass: document.getElementById('hudAirMass'),
        hudPressure: document.getElementById('hudPressure'),
        radarFrameLabel: document.getElementById('radarFrameLabel'),
        radarScrubber: document.getElementById('radarScrubber'),
        radarOpacity: document.getElementById('radarOpacity'),
        btnRadarPlay: document.getElementById('btnRadarPlay'),
        radarPlayIcon: document.getElementById('radarPlayIcon'),
        peakTriggerHour: document.getElementById('peakTriggerHour'),
        peakAttentionVal: document.getElementById('peakAttentionVal'),
        // RL Elements
        dqnQuickActionName: document.getElementById('dqnQuickActionName'),
        dqnQuickDesc: document.getElementById('dqnQuickDesc'),
        dqnQuickBadge: document.getElementById('dqnQuickBadge'),
        dqnQuickIcon: document.getElementById('dqnQuickIcon'),
        rlActionCode: document.getElementById('rlActionCode'),
        rlActionTitle: document.getElementById('rlActionTitle'),
        rlActionDesc: document.getElementById('rlActionDesc'),
        rlActionIconCircle: document.getElementById('rlActionIconCircle'),
        rlRationaleText: document.getElementById('rlRationaleText'),
        qValuesList: document.getElementById('qValuesList'),
        // Historical Elements
        btnHistPrev: document.getElementById('btnHistPrev'),
        btnHistRandom: document.getElementById('btnHistRandom'),
        btnHistNext: document.getElementById('btnHistNext'),
        histRecordRange: document.getElementById('histRecordRange'),
        histDateStart: document.getElementById('histDateStart'),
        histDateEnd: document.getElementById('histDateEnd'),
        histIndexSlider: document.getElementById('histIndexSlider'),
        btnRunHistValidation: document.getElementById('btnRunHistValidation'),
        histActualVal: document.getElementById('histActualVal'),
        histPredVal: document.getElementById('histPredVal'),
        histResidualVal: document.getElementById('histResidualVal'),
        histAccuracyBadge: document.getElementById('histAccuracyBadge'),
        // Contingency Matrix
        valHitsTP: document.getElementById('valHitsTP'),
        valFalseAlarmsFP: document.getElementById('valFalseAlarmsFP'),
        valMissesFN: document.getElementById('valMissesFN'),
        valCorrectNegTN: document.getElementById('valCorrectNegTN'),
        valTotalForecastYes: document.getElementById('valTotalForecastYes'),
        valTotalForecastNo: document.getElementById('valTotalForecastNo'),
        valTotalObservedYes: document.getElementById('valTotalObservedYes'),
        valTotalObservedNo: document.getElementById('valTotalObservedNo')
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
        
        // Fetch background APIs
        await fetchStatus();
        await fetchScenarios();
        await fetchBenchmarks();
        
        // Initial neural inference run
        await runInference();
        
        // Pre-load historical sample
        fetchHistoricalSequence(1000);
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
            html: '<div style="width:14px; height:14px; border-radius:50%; background:#ef4444; box-shadow:0 0 10px #ef4444; border:2px solid #ffffff;"></div>',
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
            state.currentWeather.tir1_temp = -68.5;
            state.currentWeather.cloud_top_height = 16.5;
            state.currentWeather.cape_index = 3800.0;
            state.currentWeather.moisture_conv = 12.8;
            state.currentWeather.pressure = 991.0;
        } else if (lat > 28.0 && lng > 77.0) {
            state.currentWeather.tir1_temp = -62.0;
            state.currentWeather.cloud_top_height = 15.0;
            state.currentWeather.cape_index = 2900.0;
            state.currentWeather.moisture_conv = 10.5;
            state.currentWeather.pressure = 988.0;
        } else if (lat < 16.0 && lng > 79.0) {
            state.currentWeather.tir1_temp = -48.0;
            state.currentWeather.cloud_top_height = 12.8;
            state.currentWeather.cape_index = 2100.0;
            state.currentWeather.moisture_conv = 7.5;
            state.currentWeather.pressure = 997.0;
        } else {
            state.currentWeather.tir1_temp = 18.0;
            state.currentWeather.cloud_top_height = 1.5;
            state.currentWeather.cape_index = 200.0;
            state.currentWeather.moisture_conv = 0.1;
            state.currentWeather.pressure = 1018.0;
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
                console.log('INSAT-3D/3DR XAI Engine Online:', data.model_type, data.device);
            }
        } catch (err) {
            console.log('Status check:', err);
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
        if (!state.benchmarks) return;
        
        // 1. KPI Cards
        const metrics = state.benchmarks.verification_metrics;
        if (metrics && elements.metricsKpiGrid) {
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

        // 2. Contingency Matrix
        const cm = state.benchmarks.confusion_matrix_heavy_events;
        if (cm) {
            if (elements.valHitsTP) elements.valHitsTP.textContent = `Hits (TP): ${cm.hits_tp.toLocaleString()}`;
            if (elements.valFalseAlarmsFP) elements.valFalseAlarmsFP.textContent = `False Alarms (FP): ${cm.false_alarms_fp.toLocaleString()}`;
            if (elements.valMissesFN) elements.valMissesFN.textContent = `Misses (FN): ${cm.misses_fn.toLocaleString()}`;
            if (elements.valCorrectNegTN) elements.valCorrectNegTN.textContent = `Correct Rejections (TN): ${cm.correct_negatives_tn.toLocaleString()}`;
            
            const totalYes = cm.hits_tp + cm.false_alarms_fp;
            const totalNo = cm.misses_fn + cm.correct_negatives_tn;
            const obsYes = cm.hits_tp + cm.misses_fn;
            const obsNo = cm.false_alarms_fp + cm.correct_negatives_tn;
            
            if (elements.valTotalForecastYes) elements.valTotalForecastYes.textContent = totalYes.toLocaleString();
            if (elements.valTotalForecastNo) elements.valTotalForecastNo.textContent = totalNo.toLocaleString();
            if (elements.valTotalObservedYes) elements.valTotalObservedYes.textContent = obsYes.toLocaleString();
            if (elements.valTotalObservedNo) elements.valTotalObservedNo.textContent = obsNo.toLocaleString();
        }
    }

    // --- 5. Historical Dataset Fetcher ---
    async function fetchHistoricalSequence(startIdx = 1000) {
        try {
            const res = await fetch(`/api/historical?start_idx=${startIdx}&length=24`);
            const data = await res.json();
            
            state.historical.currentIndex = data.start_index;
            state.historical.totalRecords = data.total_records;
            state.historical.currentSequence = data.sequence;
            state.historical.nextActualRain = data.next_actual_rainfall_mm;
            
            if (elements.histRecordRange) {
                elements.histRecordRange.textContent = `Records ${data.start_index} - ${data.start_index + 24}`;
            }
            if (elements.histDateStart) elements.histDateStart.textContent = data.date_start;
            if (elements.histDateEnd) elements.histDateEnd.textContent = data.date_end;
            if (elements.histIndexSlider) elements.histIndexSlider.value = data.start_index;
            if (elements.histActualVal) {
                elements.histActualVal.innerHTML = `${data.next_actual_rainfall_mm.toFixed(2)} <span class="unit">mm/h</span>`;
            }
        } catch (err) {
            console.error('Historical dataset fetch notice:', err);
        }
    }

    async function runHistoricalInference() {
        if (!state.historical.currentSequence || state.historical.currentSequence.length !== 24) return;
        
        try {
            if (elements.btnRunHistValidation) elements.btnRunHistValidation.textContent = 'Evaluating Sequence...';
            
            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    sequence: state.historical.currentSequence,
                    hour_of_day: 14
                })
            });
            const data = await res.json();
            const pred = data.predicted_rainfall_mm;
            state.historical.predictedRain = pred;
            
            if (elements.histPredVal) {
                elements.histPredVal.innerHTML = `${pred.toFixed(2)} <span class="unit">mm/h</span>`;
            }
            
            const residual = Math.abs(pred - state.historical.nextActualRain);
            if (elements.histResidualVal) {
                elements.histResidualVal.textContent = `${residual.toFixed(2)} mm/h`;
            }
            if (elements.histAccuracyBadge) {
                if (residual < 3.0) {
                    elements.histAccuracyBadge.className = 'text-emerald';
                    elements.histAccuracyBadge.textContent = 'High Precision Fit (Residual < 3 mm/h)';
                } else if (residual < 8.0) {
                    elements.histAccuracyBadge.className = 'text-blue';
                    elements.histAccuracyBadge.textContent = 'Acceptable Bounds';
                } else {
                    elements.histAccuracyBadge.className = 'text-amber';
                    elements.histAccuracyBadge.textContent = 'Convective Ambiguity';
                }
            }
        } catch (err) {
            console.error('Historical inference error:', err);
        } finally {
            if (elements.btnRunHistValidation) elements.btnRunHistValidation.innerHTML = '<i class="fa-solid fa-play"></i> Run Inference on Historical Sequence';
        }
    }

    // --- 6. Real-Time Neural Inference & DQN Decision Logic ---
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
            updateDQNDecision(data);
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

    // --- 7. DQN Autonomous Decision Engine Evaluation ---
    function updateDQNDecision(data) {
        const pred_mm = data.predicted_rainfall_mm;
        const curr = state.currentWeather;
        const cth = curr.cloud_top_height;
        const cape = curr.cape_index;
        const mc = curr.moisture_conv;
        const tir1 = curr.tir1_temp;

        // Simulate Q-values from Bellman environment policy in dqn_agent.py
        let q0 = 0.0; // Action 0: Normal Operations
        let q1 = 0.0; // Action 1: Agricultural Delay
        let q2 = 0.0; // Action 2: Grid Backup
        let q3 = 0.0; // Action 3: Disaster Evacuation

        // Action 0 (Normal) value
        if (pred_mm < 1.0 && tir1 > -30.0) {
            q0 = 8.5;
        } else if (pred_mm < 7.6) {
            q0 = 2.0;
        } else {
            q0 = -45.0 - (pred_mm * 1.5);
        }

        // Action 1 (Agri delay) value
        if (pred_mm >= 0.5 && pred_mm <= 15.0) {
            q1 = 28.0 + (pred_mm * 1.2);
        } else if (pred_mm < 0.5) {
            q1 = -8.0;
        } else {
            q1 = -18.0;
        }

        // Action 2 (Grid backup) value
        if (cth > 10.0 && pred_mm > 5.0) {
            q2 = 35.0 + (cth * 2.0) + (cape / 300.0);
        } else {
            q2 = -15.0;
        }

        // Action 3 (Evacuation) value
        if (pred_mm > 35.0 || (tir1 <= -55.0 && mc >= 8.0)) {
            q3 = 95.0 + (pred_mm * 1.8);
        } else if (pred_mm > 18.0) {
            q3 = 45.0;
        } else {
            q3 = -50.0 - (tir1 < -50 ? 0 : 20); // False alarm penalty
        }

        const qArr = [
            { id: 0, name: 'Action 0: Normal Operations', code: 'ACTION 0', val: q0, color: '#10b981', icon: 'fa-sun', desc: 'Maintain normal municipal schedules and power grid operations.' },
            { id: 1, name: 'Action 1: Agricultural Delay', code: 'ACTION 1', val: q1, color: '#38bdf8', icon: 'fa-faucet-drip', desc: 'Pause automated irrigation to conserve municipal water & electricity ahead of gentle rain.' },
            { id: 2, name: 'Action 2: Grid Aux Backup', code: 'ACTION 2', val: q2, color: '#f59e0b', icon: 'fa-bolt', desc: 'Spin up standby power generators to buffer solar drop and convective squall disruptions.' },
            { id: 3, name: 'Action 3: Disaster Evacuation', code: 'ACTION 3', val: q3, color: '#ef4444', icon: 'fa-triangle-exclamation', desc: 'Mobilize SDRF/NDRF emergency response units, flood barriers, and coastal evacuation.' }
        ];

        // Choose argmax
        let bestAction = qArr[0];
        for (let i = 1; i < qArr.length; i++) {
            if (qArr[i].val > bestAction.val) {
                bestAction = qArr[i];
            }
        }

        // 1. Update Quick Advisory Card on Main Dashboard
        if (elements.dqnQuickActionName) {
            elements.dqnQuickActionName.textContent = bestAction.name;
            elements.dqnQuickActionName.style.color = bestAction.color;
        }
        if (elements.dqnQuickIcon) {
            elements.dqnQuickIcon.className = `fa-solid ${bestAction.icon}`;
            elements.dqnQuickIcon.style.color = bestAction.color;
        }
        if (elements.dqnQuickDesc) elements.dqnQuickDesc.textContent = bestAction.desc;

        // 2. Update Dedicated RL Tab Elements
        if (elements.rlActionCode) {
            elements.rlActionCode.textContent = `${bestAction.code}: OPTIMAL POLICY RECOMMENDATION`;
            elements.rlActionCode.style.color = bestAction.color;
        }
        if (elements.rlActionTitle) elements.rlActionTitle.textContent = bestAction.name;
        if (elements.rlActionDesc) elements.rlActionDesc.textContent = bestAction.desc;
        if (elements.rlActionIconCircle) {
            elements.rlActionIconCircle.innerHTML = `<i class="fa-solid ${bestAction.icon}"></i>`;
            elements.rlActionIconCircle.style.borderColor = bestAction.color;
            elements.rlActionIconCircle.style.color = bestAction.color;
            elements.rlActionIconCircle.style.backgroundColor = bestAction.color + '22';
        }

        if (elements.rlRationaleText) {
            if (bestAction.id === 3) {
                elements.rlRationaleText.textContent = `Severe convective cloudburst predicted (${pred_mm.toFixed(1)} mm/h). The RL agent selects Evacuation (Action 3) to secure the maximum disaster prevention reward (+100) and avoid catastrophic failure (-100).`;
            } else if (bestAction.id === 2) {
                elements.rlRationaleText.textContent = `High cloud top vertical extension (${cth.toFixed(1)} km) and moderate rain detected. The agent selects Grid Backup (Action 2) with +25 reward to defend against solar drop and squalls.`;
            } else if (bestAction.id === 1) {
                elements.rlRationaleText.textContent = `Gentle/Moderate precipitation predicted (${pred_mm.toFixed(1)} mm/h). The agent selects Agricultural Delay (Action 1) to conserve water resources (+15 reward).`;
            } else {
                elements.rlRationaleText.textContent = `Atmospheric conditions indicate fair/dry weather (${pred_mm.toFixed(1)} mm/h). Action 0 (Normal Operations) is selected to prevent costly false alarm penalties (-50).`;
            }
        }

        // Q-Values Distribution Bars
        if (elements.qValuesList) {
            const maxQ = Math.max(10, ...qArr.map(q => q.val));
            const minQ = Math.min(-10, ...qArr.map(q => q.val));
            const range = maxQ - minQ;

            elements.qValuesList.innerHTML = qArr.map(q => {
                const isSelected = q.id === bestAction.id;
                const normalizedPct = Math.max(8, Math.min(100, ((q.val - minQ) / range) * 100));
                return `
                    <div class="q-val-item ${isSelected ? 'active-q-item' : ''}">
                        <div class="q-item-header">
                            <span class="q-item-name"><i class="fa-solid ${q.icon}" style="color:${q.color}"></i> ${q.name}</span>
                            <span class="q-item-val" style="color:${q.color}">Q = ${q.val.toFixed(1)}</span>
                        </div>
                        <div class="q-item-track">
                            <div class="q-item-fill" style="width: ${normalizedPct}%; background: ${q.color};"></div>
                        </div>
                    </div>
                `;
            }).join('');
        }
    }

    // --- 8. UI & XAI Rendering ---
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

            // Find peak attention hour
            let maxW = -1;
            let peakIdx = 23;
            data.temporal_attention_weights.forEach((w, idx) => {
                if (w > maxW) {
                    maxW = w;
                    peakIdx = idx;
                }
            });
            const hoursAgo = 23 - peakIdx;
            if (elements.peakTriggerHour) elements.peakTriggerHour.textContent = hoursAgo === 0 ? 'Current Hour (t0)' : `t-${hoursAgo}h preceding`;
            if (elements.peakAttentionVal) elements.peakAttentionVal.textContent = `Weight: ${maxW.toFixed(3)}`;
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
        const scenarioData = sc.weather || sc.data;
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

    // --- 9. Event Listeners ---
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
        const modeButtons = [
            elements.btnModeQuick,
            elements.btnModeXAI,
            elements.btnModeRL,
            elements.btnModeHistorical,
            elements.btnModeMap,
            elements.btnModeMetrics
        ];
        
        modeButtons.forEach(btn => {
            if (!btn) return;
            btn.addEventListener('click', () => {
                modeButtons.forEach(b => { if (b) b.classList.remove('active'); });
                btn.classList.add('active');
                state.mode = btn.dataset.mode;

                if (elements.xaiSection) elements.xaiSection.classList.add('hidden');
                if (elements.rlSection) elements.rlSection.classList.add('hidden');
                if (elements.historicalSection) elements.historicalSection.classList.add('hidden');
                if (elements.mapSection) elements.mapSection.classList.add('hidden');
                if (elements.benchmarksSection) elements.benchmarksSection.classList.add('hidden');
                if (elements.mainDashboardGrid) elements.mainDashboardGrid.classList.remove('hidden');
                if (elements.bottomTimelineSection) elements.bottomTimelineSection.classList.remove('hidden');

                if (state.mode === 'xai') {
                    if (elements.xaiSection) elements.xaiSection.classList.remove('hidden');
                } else if (state.mode === 'rl') {
                    if (elements.rlSection) elements.rlSection.classList.remove('hidden');
                } else if (state.mode === 'historical') {
                    if (elements.historicalSection) elements.historicalSection.classList.remove('hidden');
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

        if (elements.btnQuickOpenRL) {
            elements.btnQuickOpenRL.addEventListener('click', () => {
                if (elements.btnModeRL) elements.btnModeRL.click();
            });
        }

        // Historical Dataset Controls
        if (elements.btnHistPrev) {
            elements.btnHistPrev.addEventListener('click', () => {
                const nextIdx = Math.max(0, state.historical.currentIndex - 24);
                fetchHistoricalSequence(nextIdx);
            });
        }

        if (elements.btnHistNext) {
            elements.btnHistNext.addEventListener('click', () => {
                const nextIdx = Math.min(state.historical.totalRecords - 25, state.historical.currentIndex + 24);
                fetchHistoricalSequence(nextIdx);
            });
        }

        if (elements.btnHistRandom) {
            elements.btnHistRandom.addEventListener('click', () => {
                const randomIdx = Math.floor(Math.random() * (state.historical.totalRecords - 100));
                fetchHistoricalSequence(randomIdx);
            });
        }

        if (elements.histIndexSlider) {
            elements.histIndexSlider.addEventListener('input', (e) => {
                const val = parseInt(e.target.value);
                fetchHistoricalSequence(val);
            });
        }

        if (elements.btnRunHistValidation) {
            elements.btnRunHistValidation.addEventListener('click', runHistoricalInference);
        }

        // Map Station Chips
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

        // Map Layer Opacity Slider
        if (elements.radarOpacity) {
            elements.radarOpacity.addEventListener('input', (e) => {
                const val = parseInt(e.target.value);
                state.map.opacity = val / 100.0;
                
                if (state.map.radarLayers.length > 0 && state.map.activeLayers.radar) {
                    state.map.radarLayers.forEach(l => {
                        if (l && typeof l.setOpacity === 'function') {
                            l.setOpacity(state.map.opacity);
                        }
                    });
                }

                if (state.map.cloudLayer && typeof state.map.cloudLayer.setOpacity === 'function') {
                    state.map.cloudLayer.setOpacity(state.map.opacity);
                }
            });
        }

        // Map Layer Switcher Pills
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

    // --- 10. Chart Initializations ---
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
