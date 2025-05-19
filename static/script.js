document.addEventListener('DOMContentLoaded', () => {
    // Modern Slideshow functionality
    let slideIndex = 1;
    let slideInterval;
    const slideDuration = 5000; // 5 seconds per slide
    let progressBar;
    let startTime;
    let animationFrame;

    function initSlideshow() {
        showSlides(slideIndex);
        startAutoSlide();
        initProgressBar();
    }

    function showSlides(n) {
        const slides = document.getElementsByClassName("slide");
        const indicators = document.getElementsByClassName("indicator");
        
        if (n > slides.length) {slideIndex = 1}
        if (n < 1) {slideIndex = slides.length}
        
        Array.from(slides).forEach(slide => {
            slide.style.display = "none";
            slide.classList.remove("active");
        });
        
        Array.from(indicators).forEach(indicator => {
            indicator.classList.remove("active");
        });
        
        slides[slideIndex-1].style.display = "block";
        slides[slideIndex-1].classList.add("active");
        indicators[slideIndex-1].classList.add("active");

        resetProgressBar();
    }

    function startAutoSlide() {
        if (slideInterval) {
            clearInterval(slideInterval);
        }
        slideInterval = setInterval(() => {
            plusSlides(1);
        }, slideDuration);
    }

    function initProgressBar() {
        progressBar = document.querySelector('.progress-bar');
        startTime = Date.now();
        updateProgressBar();
    }

    function updateProgressBar() {
        if (!progressBar) return;
        const elapsed = Date.now() - startTime;
        const progress = (elapsed / slideDuration) * 100;
        
        if (progress <= 100) {
            progressBar.style.width = `${progress}%`;
            animationFrame = requestAnimationFrame(updateProgressBar);
        }
    }

    function resetProgressBar() {
        progressBar = document.querySelector('.progress-bar');
        if (!progressBar) return;
        if (animationFrame) {
            cancelAnimationFrame(animationFrame);
        }
        progressBar.style.width = '0%';
        startTime = Date.now();
        updateProgressBar();
    }

    window.plusSlides = function(n) {
        showSlides(slideIndex += n);
        startAutoSlide();
    }

    window.currentSlide = function(n) {
        showSlides(slideIndex = n);
        startAutoSlide();
    }

    initSlideshow();

    const slideshow = document.querySelector('.modern-slideshow');
    slideshow.addEventListener('mouseenter', () => {
        clearInterval(slideInterval);
        if (animationFrame) {
            cancelAnimationFrame(animationFrame);
        }
    });

    slideshow.addEventListener('mouseleave', () => {
        startAutoSlide();
        resetProgressBar();
    });

    const themeToggle = document.getElementById('themeToggle');
    const html = document.documentElement;
    const themeIcon = themeToggle.querySelector('i');

    const savedTheme = localStorage.getItem('theme') || 'light';
    html.setAttribute('data-theme', savedTheme);
    updateThemeIcon(savedTheme);

    themeToggle.addEventListener('click', () => {
        const currentTheme = html.getAttribute('data-theme');
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';
        
        html.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
        updateThemeIcon(newTheme);
    });

    function updateThemeIcon(theme) {
        themeIcon.className = theme === 'light' ? 'fas fa-moon' : 'fas fa-sun';
    }

    const authFormsContainer = document.getElementById('auth-forms');
    const mainContent = document.getElementById('main-content');
    const authTabs = document.querySelectorAll('.auth-tab');
    const authForms = document.querySelectorAll('.auth-form');
    const loginForm = document.getElementById('loginForm');
    const registerForm = document.getElementById('registerForm');

    const token = localStorage.getItem('token');
    if (token) {
        showMainContent();
    }

    authTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const targetForm = tab.dataset.tab;
            
            authTabs.forEach(t => t.classList.remove('active'));
            authForms.forEach(f => f.classList.remove('active'));
            
            tab.classList.add('active');
            document.getElementById(`${targetForm}Form`).classList.add('active');
        });
    });

    loginForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const username = document.getElementById('loginUsername').value;
        const password = document.getElementById('loginPassword').value;

        try {
            const response = await fetch('/token', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: new URLSearchParams({
                    'username': username,
                    'password': password,
                }),
            });

            const data = await response.json();

            if (response.ok) {
                localStorage.setItem('token', data.access_token);
                showMainContent();
            } else {
                throw new Error(data.detail || 'Login failed');
            }
        } catch (error) {
            alert('Error: ' + error.message);
        }
    });

    registerForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const username = document.getElementById('registerUsername').value;
        const email = document.getElementById('registerEmail').value;
        const password = document.getElementById('registerPassword').value;

        try {
            const response = await fetch('/register', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    username,
                    email,
                    password,
                }),
            });

            const data = await response.json();

            if (response.ok) {
                alert('Registration successful! Please login.');
                document.querySelector('[data-tab="login"]').click();
            } else {
                throw new Error(data.detail || 'Registration failed');
            }
        } catch (error) {
            alert('Error: ' + error.message);
        }
    });

    function showMainContent() {
        authFormsContainer.classList.add('hidden');
        mainContent.classList.remove('hidden');
        updateUserInfo();
    }

    async function updateUserInfo() {
        try {
            const token = localStorage.getItem('token');
            if (!token) {
                console.error('No token found');
                return;
            }

            const response = await fetch('/users/me', {
                headers: {
                    'Authorization': `Bearer ${token}`,
                    'Content-Type': 'application/json'
                },
            });

            if (response.ok) {
                const user = await response.json();
                const userInfo = document.createElement('div');
                userInfo.className = 'user-info';
                userInfo.innerHTML = `
                    <span class="user-name">Welcome, ${user.username}!</span>
                    <button class="logout-btn" onclick="logout()">Logout</button>
                `;
                mainContent.insertBefore(userInfo, mainContent.firstChild);
            } else if (response.status === 401) {
                localStorage.removeItem('token');
                location.reload();
            } else {
                throw new Error('Failed to fetch user info');
            }
        } catch (error) {
            console.error('Error fetching user info:', error);
            localStorage.removeItem('token');
            location.reload();
        }
    }

    window.logout = function() {
        localStorage.removeItem('token');
        location.reload();
    };

    const originalFetch = window.fetch;
    window.fetch = function(url, options = {}) {
        const token = localStorage.getItem('token');
        if (token && url.startsWith('/')) { // Only add token for relative paths (API calls)
            options.headers = {
                ...options.headers,
                'Authorization': `Bearer ${token}`,
            };
        }
        return originalFetch(url, options);
    };

    const navLinks = document.querySelectorAll('.nav-link');
    const sections = document.querySelectorAll('.section');

    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = link.getAttribute('href').substring(1);
            
            navLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');
            
            sections.forEach(section => {
                section.classList.remove('active');
                if (section.id === targetId) {
                    section.classList.add('active');
                }
            });
        });
    });

    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const fileInfo = document.getElementById('fileInfo');
    const uploadForm = document.getElementById('uploadForm');
    const predictBtn = document.querySelector('.predict-btn');
    const result = document.getElementById('result');
    const loader = document.getElementById('loader');
    const predictionHistory = document.getElementById('predictionHistory');

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        dropZone.classList.add('highlight');
    }

    function unhighlight(e) {
        dropZone.classList.remove('highlight');
    }

    dropZone.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    fileInput.addEventListener('change', (e) => {
        handleFiles(e.target.files);
    });

    function handleFiles(files) {
        if (files.length > 0) {
            const file = files[0];
            if (validateFile(file)) {
                fileInput.files = files; 
                displayFileInfo(file);
                predictBtn.disabled = false;
            }
        }
    }

    function validateFile(file) {
        const validTypes = ['audio/wav', 'audio/mpeg', 'audio/mp3', 'audio/ogg', 'audio/flac', 'audio/aac', 'audio/x-m4a', 'audio/m4a'];
        const maxSize = 10 * 1024 * 1024; // 10MB

        if (!file.type.startsWith('audio/') && !validTypes.some(type => file.name.toLowerCase().endsWith(type.split('/')[1]))) {
             if (!validTypes.includes(file.type)) {
                alert('Please upload a valid audio file (e.g., WAV, MP3, OGG, FLAC, AAC, M4A).');
                return false;
            }
        }

        if (file.size > maxSize) {
            alert('File size should be less than 10MB');
            return false;
        }

        return true;
    }

    function displayFileInfo(file) {
        const fileSize = (file.size / (1024 * 1024)).toFixed(2);
        fileInfo.innerHTML = `
            <p><strong>File:</strong> ${file.name}</p>
            <p><strong>Size:</strong> ${fileSize} MB</p>
            <p><strong>Type:</strong> ${file.type || 'N/A'}</p>
        `;
        fileInfo.style.display = 'block';
    }

    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        if (!fileInput.files || fileInput.files.length === 0) {
            alert('Please choose a file first.');
            return;
        }

        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        loader.classList.remove('hidden');
        result.classList.add('hidden');
        predictBtn.disabled = true;

        try {
            console.log('=== Starting prediction request ===');
            console.log('File being sent:', fileInput.files[0].name);
            
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            console.log('Response status:', response.status);
            const responseHeaders = {};
            for(let pair of response.headers.entries()) { responseHeaders[pair[0]] = pair[1]; }
            console.log('Response headers:', responseHeaders);
            
            const data = await response.json();
            console.log('Raw response data:', data);

            if (response.ok) {
                // More robust check for the new response structure
                if (!data.prediction || 
                    !data.prediction.hasOwnProperty('genre') || 
                    !data.prediction.hasOwnProperty('confidence') ||
                    data.prediction.genre === null || data.prediction.genre === '' || 
                    typeof data.prediction.confidence !== 'number'
                    ) {
                    // Detailed logging for why this condition is true
                    console.error('Robust check failed. Details:');
                    console.log('data.prediction exists?:', !!data.prediction);
                    if (data.prediction) {
                        console.log('data.prediction.hasOwnProperty(\'genre\')?:', data.prediction.hasOwnProperty('genre'));
                        console.log('data.prediction.hasOwnProperty(\'confidence\')?:', data.prediction.hasOwnProperty('confidence'));
                        console.log('data.prediction.genre value:', data.prediction.genre);
                        console.log('data.prediction.genre === null?:', data.prediction.genre === null);
                        console.log('data.prediction.genre === \'\'?:', data.prediction.genre === '');
                        console.log('typeof data.prediction.confidence:', typeof data.prediction.confidence);
                        console.log('typeof data.prediction.confidence !== \'number\'?:', typeof data.prediction.confidence !== 'number');
                    }
                    console.error('Invalid response format (robust check failed - full data):', data);
                    throw new Error('Invalid response format from server (robust check details in console)');
                }
                
                if (typeof data.prediction.genre === 'string' && data.prediction.genre.startsWith('Error:')) {
                    console.error('Backend prediction error:', data.prediction.genre);
                    throw new Error(data.prediction.genre);
                }

                console.log('Processing prediction (new structure):', {
                    prediction: data.prediction
                });
                
                console.log('Result container:', result);
                console.log('Genre result element:', result.querySelector('.genre-result'));
                console.log('Confidence meter:', result.querySelector('.confidence-meter'));
                console.log('Predictions list:', result.querySelector('.predictions-list'));
                
                displayResult(data.prediction, [data.prediction]); 
                addToHistory(data.prediction.genre, fileInput.files[0].name);
            } else {
                console.error('Server error details:', data.detail); 
                throw new Error(data.detail || `Server error: ${response.status}`);
            }
        } catch (error) {
            console.error('Prediction error (in catch block):', error);
            console.error('Error stack:', error.stack);
            displayResult({ genre: "Unknown", confidence: 0 }, []);
            alert('Prediction failed: ' + error.message);
        } finally {
            loader.classList.add('hidden');
            predictBtn.disabled = false;
        }
    });

    function displayResult(topPrediction, allPredictions) {
        console.log('=== Displaying result ===');
        console.log('Top prediction (for display):', topPrediction);
        console.log('All predictions (for display - should be array with one item):', allPredictions);
        
        const genreResult = result.querySelector('.genre-result');
        const confidenceMeter = result.querySelector('.confidence-meter');
        const predictionsList = result.querySelector('.predictions-list');
        
        if (!genreResult || !confidenceMeter || !predictionsList) {
            console.error('One or more result DOM elements not found!');
            return;
        }
        
        if (!topPrediction || typeof topPrediction.genre === 'undefined' || typeof topPrediction.confidence === 'undefined') {
            console.error('Invalid topPrediction object for displayResult:', topPrediction);
            genreResult.textContent = "Error: Invalid prediction data received";
            genreResult.style.color = "red";
            confidenceMeter.style.setProperty('--confidence', '0%');
            predictionsList.innerHTML = '';
            result.classList.remove('hidden');
            return;
        }
        
        if (topPrediction.genre === "Unknown" || (typeof topPrediction.genre === 'string' && topPrediction.genre.startsWith("Error:"))) {
            console.log('Displaying error/unknown prediction:', topPrediction.genre);
            genreResult.textContent = topPrediction.genre; 
            genreResult.style.color = "red";
            confidenceMeter.style.setProperty('--confidence', '0%');
            predictionsList.innerHTML = '';
        } else {
            console.log('Displaying successful genre result:', topPrediction.genre);
            genreResult.textContent = topPrediction.genre;
            genreResult.style.color = "var(--primary-color)";
            
            const confidence = topPrediction.confidence * 100;
            console.log('Setting confidence for display:', confidence);
            confidenceMeter.style.setProperty('--confidence', `${confidence}%`);
            
            if (Array.isArray(allPredictions)) {
                console.log('Setting predictions list for display:', allPredictions);
                const predictionsHTML = allPredictions
                    .filter(pred => pred && typeof pred.genre !== 'undefined' && typeof pred.confidence !== 'undefined') 
                    .map(pred => `
                        <div class="prediction-item">
                            <span class="genre">${pred.genre}</span>
                            <span class="confidence">${(pred.confidence * 100).toFixed(1)}%</span>
                        </div>
                    `)
                    .join('');
                predictionsList.innerHTML = predictionsHTML;
            } else {
                console.error('Invalid allPredictions array for display:', allPredictions);
                predictionsList.innerHTML = '';
            }
        }
        
        result.classList.remove('hidden');
        console.log('Result container shown.');
    }

    function addToHistory(prediction, fileName) {
        if (!predictionHistory) {
            console.warn('predictionHistory element not found. Cannot add to history.');
            return;
        }
        const historyItem = document.createElement('div');
        historyItem.className = 'history-item';
        historyItem.innerHTML = `
            <p><strong>File:</strong> ${fileName}</p>
            <p><strong>Genre:</strong> ${prediction || 'N/A'}</p> 
            <p><strong>Time:</strong> ${new Date().toLocaleTimeString()}</p>
        `;
        
        predictionHistory.insertBefore(historyItem, predictionHistory.firstChild);
        
        if (predictionHistory.children.length > 5) {
            predictionHistory.removeChild(predictionHistory.lastChild);
        }
    }

    const translations = {
        en: {
            site_title: 'Music Genre Classifier',
            nav_home: 'Home',
            nav_about: 'About',
            nav_contact: 'Contact',
            nav_likes: 'Likes',
            nav_recommendations: 'Recommendations',
            hero_title: 'AI Music Genre Classifier',
            hero_subtitle: 'Identify your music genre in seconds. Just upload a file and get the result!',
            hero_try: 'Try Now',
            main_title: 'Discover Your Music Genre',
            main_subtitle: 'Upload your music file and let our AI classify its genre',
            drop_text: 'Drag & Drop your audio file here',
            drop_or: 'or',
            drop_choose: 'Choose File',
            predict_btn: 'Predict Genre',
            loader_text: 'Analyzing your music...',
            result_title: 'Prediction Result',
            history_title: 'Recent Predictions',
            about_title: 'About Our Classifier',
            about_desc: 'Our AI-powered music genre classifier uses advanced machine learning to analyze your music and determine its genre. We support various music genres including:',
            contact_title: 'Contact Us',
            likes_title: 'Liked Tracks',
            recommendations_title: 'Recommendations',
            feature_instant: 'Instant',
            feature_instant_desc: 'Get results in a couple of seconds',
            feature_ai: 'AI Inside',
            feature_ai_desc: 'Modern algorithms',
            feature_safe: 'Safe',
            feature_safe_desc: 'Your files are not saved',
        },
        ru: {
            site_title: 'Классификатор жанров музыки',
            nav_home: 'Главная',
            nav_about: 'О сервисе',
            nav_contact: 'Контакты',
            nav_likes: 'Избранное',
            nav_recommendations: 'Рекомендации',
            hero_title: 'AI-классификатор жанров музыки',
            hero_subtitle: 'Узнай жанр своей музыки за секунды. Просто загрузи файл — и получи результат!',
            hero_try: 'Попробовать',
            main_title: 'Определи жанр своей музыки',
            main_subtitle: 'Загрузите аудиофайл — наш ИИ определит его жанр',
            drop_text: 'Перетащите аудиофайл сюда',
            drop_or: 'или',
            drop_choose: 'Выбрать файл',
            predict_btn: 'Определить жанр',
            loader_text: 'Анализируем вашу музыку...',
            result_title: 'Результат',
            history_title: 'История предсказаний',
            about_title: 'О сервисе',
            about_desc: 'Наш ИИ-классификатор анализирует музыку и определяет её жанр с помощью современных алгоритмов машинного обучения. Мы поддерживаем множество жанров, включая:',
            contact_title: 'Связаться с нами',
            likes_title: 'Избранные треки',
            recommendations_title: 'Рекомендации',
            feature_instant: 'Мгновенно',
            feature_instant_desc: 'Результат за пару секунд',
            feature_ai: 'ИИ внутри',
            feature_ai_desc: 'Современные алгоритмы',
            feature_safe: 'Безопасно',
            feature_safe_desc: 'Ваши файлы не сохраняются',
        }
    };

    function setLanguage(lang) {
        localStorage.setItem('lang', lang);
        document.documentElement.setAttribute('lang', lang);
        document.getElementById('langLabel').textContent = lang.toUpperCase();
        const elements = document.querySelectorAll('[data-i18n]');
        elements.forEach(el => {
            const key = el.getAttribute('data-i18n');
            if (translations[lang] && translations[lang][key]) {
                el.textContent = translations[lang][key];
            } else {
                console.warn(`Missing translation for key: ${key} in language: ${lang}`);
            }
        });
    }

    const savedLang = localStorage.getItem('lang') || 'en';
    setLanguage(savedLang);

    document.getElementById('langToggle').addEventListener('click', () => {
        const current = localStorage.getItem('lang') || 'en';
        const next = current === 'en' ? 'ru' : 'en';
        setLanguage(next);
    });

    const burgerBtn = document.getElementById('burgerBtn');
    const sidebar = document.querySelector('.sidebar');
    burgerBtn.addEventListener('click', () => {
        sidebar.classList.toggle('open');
    });

    const searchInput = document.getElementById('searchInput');
    if (searchInput) {
        searchInput.addEventListener('input', (e) => {
            const query = e.target.value.trim().toLowerCase();
            console.log('Search query:', query);
        });
    }
});